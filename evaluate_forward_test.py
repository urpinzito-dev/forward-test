"""
Evaluate the forward test — Daniel / Modelo de Predicciones de Fútbol
========================================================================
~/workspace/lightgbm-model/evaluate_forward_test.py

Reads the snapshots saved by forward_test.py
(data/forward_test/YYYY-MM-DD.csv), settles them, and computes the same
financial metrics as financial_backtest.py — EV, CLV, ROI, hit rate —
per focus market. Focus markets ONLY:

  * corners_away_over45  (best log loss 0.645)
  * corners_home_over45  (0.667)
  * cards_home_over15    (0.673)
  * cards_over35         (0.676)

Shots / shots on target are OFF COMPLETELY (log loss at the
always-favourite baseline) and never appear here.

DEFINITIONS (identical to financial_backtest.py):
  * EV = (model probability * morning odds) - 1. Morning odds are the
    10:00 capture; they are the "taken" odds.
  * CLV = morning odds / closing odds - 1. Closing odds are the
    ~15-min-before-kickoff capture. No Pinnacle: this is
    morning-vs-closing CLV from the same bookmaker average.
  * Flat 1u vs Quarter Kelly (0.25x, 2% of CURRENT bankroll cap,
    reusing kelly.kelly_stake; dispersion = 0 here because the
    forward test stores one model probability per market, not the
    three base models — the Kelly result is therefore an upper bound
    on what the dispersion-penalised Kelly would stake).
  * A market is only reported as a result once it has n >= 30 settled
    bets with both odds present. Below that the script says honestly
    "not enough sample yet" and reports only the capture progress
    (days captured, matches, rows) — no ROI is quoted from tiny n.

SETTLEMENT (how `won` gets filled):
  * --results FILE.csv — manual results file with columns:
    match_id, corners_home, corners_away, cards_home, cards_total.
    (Corners/cards are public after the match; FotMob matchDetails
    and API-Football /fixtures/statistics both publish them.)
  * --settle — fetch results from API-Football /fixtures/statistics
    (needs API_FOOTBALL_KEY; "Corner Kicks", "Yellow Cards" +
    "Red Cards" per team). One request per unsettled fixture, counted
    against the same daily quota guard as forward_test.py.
  * Rows with no result stay unsettled and are excluded, never guessed.

USAGE:
  cd ~/workspace/lightgbm-model
  ./venv/bin/python evaluate_forward_test.py
  ./venv/bin/python evaluate_forward_test.py --results results.csv
  ./venv/bin/python evaluate_forward_test.py --settle
  ./venv/bin/python evaluate_forward_test.py --ev-threshold 0.03
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
OUT_DIR = os.path.join(BASE, "data", "forward_test")

from kelly import kelly_stake  # noqa: E402

FOCUS_MARKETS = [
    "corners_away_over45",
    "corners_home_over45",
    "cards_home_over15",
    "cards_over35",
]
MARKET_LABELS = {
    "corners_away_over45": "Corners away > 4.5 (log loss 0.645)",
    "corners_home_over45": "Corners home > 4.5 (log loss 0.667)",
    "cards_home_over15": "Cards home > 1.5 (log loss 0.673)",
    "cards_over35": "Cards total > 3.5 (log loss 0.676)",
}
MIN_SAMPLE = 30  # per market, before any ROI/CLV is quoted
EV_THRESHOLDS = [0.00, 0.03, 0.05]
EV_WINDOW = (0.02, 0.08)  # from market_shrinkage.py: the EV ceiling


def _won(row):
    """1/0 from the settled result columns, or None if unsettled."""
    try:
        ch = float(row["corners_home"])
        ca = float(row["corners_away"])
        csh = float(row["cards_home"])
        cst = float(row["cards_total"])
    except (KeyError, TypeError, ValueError):
        return None
    m = row["market"]
    if m == "corners_home_over45":
        return int(ch > 4.5)
    if m == "corners_away_over45":
        return int(ca > 4.5)
    if m == "cards_home_over15":
        return int(csh > 1.5)
    if m == "cards_over35":
        return int(cst > 3.5)
    return None


def load_all_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "????-??-??.csv"))):
        with open(path, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def settle_from_results(rows, results_path):
    results = {}
    with open(results_path, newline="") as f:
        for r in csv.DictReader(f):
            results[str(r["match_id"])] = r
    n = 0
    for row in rows:
        res = results.get(str(row["match_id"]))
        if not res:
            continue
        for col in ("corners_home", "corners_away", "cards_home", "cards_total"):
            if res.get(col) not in (None, ""):
                row[col] = res[col]
        w = _won(row)
        if w is not None:
            row["won"] = str(w)
            n += 1
    return n


def settle_via_api(rows):
    """Fetch match statistics from API-Football (key-gated)."""
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        print("skipped, no key: --settle needs API_FOOTBALL_KEY — "
              "0 requests made")
        return 0
    pending = {}
    for row in rows:
        if row.get("won") in ("0", "1"):
            continue
        pending.setdefault(row["match_id"], row)
    print(f"  fixtures to settle via API: {len(pending)}")
    n = 0
    for fid in sorted(pending):
        req = urllib.request.Request(
            f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fid}",
            headers={"x-apisports-key": key, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} on statistics fixture={fid} — skipped")
            continue
        except Exception as e:
            print(f"  statistics fixture={fid} failed: {e} — skipped")
            continue
        time.sleep(0.6)
        stats = {}
        for team_block in data.get("response", []):
            team = team_block.get("team", {}).get("name", "")
            d = {s.get("type"): s.get("value")
                 for s in team_block.get("statistics", [])}
            stats[team] = d
        # home/away by team name, from the pending row
        ref = pending[fid]
        home_stats = stats.get(ref["home"], {})
        away_stats = stats.get(ref["away"], {})
        try:
            ch = int(home_stats.get("Corner Kicks") or 0)
            ca = int(away_stats.get("Corner Kicks") or 0)
            csh = (int(home_stats.get("Yellow Cards") or 0)
                   + int(home_stats.get("Red Cards") or 0))
            csa = (int(away_stats.get("Yellow Cards") or 0)
                   + int(away_stats.get("Red Cards") or 0))
        except (TypeError, ValueError):
            continue
        for row in rows:
            if row["match_id"] != fid:
                continue
            row["corners_home"] = str(ch)
            row["corners_away"] = str(ca)
            row["cards_home"] = str(csh)
            row["cards_total"] = str(csh + csa)
            w = _won(row)
            if w is not None:
                row["won"] = str(w)
                n += 1
    return n


def save_all_rows(rows):
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    cols = list(rows[0].keys()) if rows else []
    for date_str, sub in by_date.items():
        path = os.path.join(OUT_DIR, f"{date_str}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(sub)


def _summarise(bets):
    if not bets:
        return {"n_bets": 0, "hit_rate": None, "mean_clv_pct": None,
                "pct_beating_close": None, "roi_at_taken_odds_pct": None,
                "profit_units_flat_1u": 0.0}
    pay = [b["odds"] if b["won"] else 0.0 for b in bets]
    clv = [b["clv"] for b in bets]
    return {
        "n_bets": len(bets),
        "hit_rate": sum(b["won"] for b in bets) / len(bets),
        "mean_clv_pct": 100 * sum(clv) / len(clv),
        "pct_beating_close": 100 * sum(1 for c in clv if c > 0) / len(clv),
        "roi_at_taken_odds_pct": 100 * (sum(pay) / len(pay) - 1.0),
        "profit_units_flat_1u": sum(pay) - len(bets),
    }


def _kelly(bets, start_bankroll=100.0):
    """Quarter Kelly, dispersion = 0 (single model probability stored).
    See module docstring: this is an upper bound on the
    dispersion-penalised Kelly stake."""
    bank, peak, max_dd = start_bankroll, start_bankroll, 0.0
    staked, n_skipped = 0.0, 0
    for b in bets:
        ks = kelly_stake(b["p"], b["odds"], fraction=0.25, dispersion=0.0)
        s = bank * ks["stake_frac"]
        if s <= 0:
            n_skipped += 1
            continue
        staked += s
        bank += s * (b["odds"] - 1.0) if b["won"] else -s
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak if peak > 0 else 0.0)
    return {
        "final_bankroll_units": bank,
        "profit_units": bank - start_bankroll,
        "roi_on_staked_pct": (100 * (bank - start_bankroll) / staked
                              if staked else 0.0),
        "total_staked_units": staked,
        "max_drawdown_pct": 100 * max_dd,
        "n_bets": len(bets) - n_skipped,
        "n_bets_skipped_zero_stake": n_skipped,
        "note": ("Quarter Kelly 0.25x, 2% of CURRENT bankroll cap, "
                 "dispersion = 0 (single stored probability). Fractional "
                 "Kelly REDUCES ruin risk; it does NOT guarantee never "
                 "going bust."),
    }


def evaluate(rows):
    settled = []
    for r in rows:
        if r.get("won") not in ("0", "1"):
            continue
        try:
            p = float(r["model_p"])
            morning = float(r["odds_morning"])
            closing = float(r["odds_closing"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0.0 < p < 1.0 and morning > 1.0 and closing > 1.0):
            continue
        settled.append({
            "date": r["date"], "market": r["market"], "p": p,
            "odds": morning, "ev": p * morning - 1.0,
            "clv": morning / closing - 1.0,
            "won": int(r["won"]),
        })
    out = {"markets": {}, "progress": {
        "snapshot_rows": len(rows),
        "settled_rows_with_both_odds_and_model_p": len(settled),
        "days_captured": len({r["date"] for r in rows}),
        "matches_captured": len({r["match_id"] for r in rows}),
    }}
    for market in FOCUS_MARKETS:
        bets = sorted([b for b in settled if b["market"] == market],
                      key=lambda b: b["date"])
        entry = {"market": MARKET_LABELS[market], "n_settled": len(bets)}
        if len(bets) < MIN_SAMPLE:
            entry["status"] = (f"not enough sample yet: {len(bets)}/{MIN_SAMPLE} "
                               "settled bets — no ROI/CLV quoted")
            out["markets"][market] = entry
            continue
        entry["status"] = "enough sample"
        entry["by_ev_threshold"] = {}
        for thr in EV_THRESHOLDS:
            sub = [b for b in bets if b["ev"] >= thr]
            s = _summarise(sub)
            s["staking"] = {"quarter_kelly_dispersion_0": _kelly(sub)}
            entry["by_ev_threshold"][f"ev>={thr:.2f}"] = s
        lo, hi = EV_WINDOW
        sub = [b for b in bets if lo <= b["ev"] <= hi]
        s = _summarise(sub)
        s["staking"] = {"quarter_kelly_dispersion_0": _kelly(sub)}
        entry["ev_window_0.02_0.08"] = s
        out["markets"][market] = entry
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate the forward test")
    ap.add_argument("--results", default=None,
                    help="CSV with match_id, corners_home, corners_away, "
                         "cards_home, cards_total")
    ap.add_argument("--settle", action="store_true",
                    help="fetch results from API-Football statistics")
    ap.add_argument("--ev-threshold", type=float, default=0.03)
    args = ap.parse_args()

    rows = load_all_rows()
    print(f"Snapshot rows: {len(rows)} | days: "
          f"{len({r['date'] for r in rows})} | matches: "
          f"{len({r['match_id'] for r in rows})}")
    if args.results:
        n = settle_from_results(rows, args.results)
        print(f"Settled from {args.results}: {n} rows")
        save_all_rows(rows)
    if args.settle:
        n = settle_via_api(rows)
        print(f"Settled via API-Football: {n} rows")
        save_all_rows(rows)

    result = evaluate(rows)
    out_path = os.path.join(OUT_DIR, "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print()
    print(f"Progress: {result['progress']['matches_captured']} matches, "
          f"{result['progress']['days_captured']} days captured, "
          f"{result['progress']['settled_rows_with_both_odds_and_model_p']} "
          "settled rows with both odds + model_p.")
    print("Target: 150-200 matches in 3-4 weeks before any market is "
          "called profitable.")
    for market in FOCUS_MARKETS:
        e = result["markets"][market]
        print(f"\n{e['market']}: {e['status']}")
        if e["status"] == "enough sample":
            t = e["by_ev_threshold"][f"ev>={args.ev_threshold:.2f}"]
            print(f"  EV>={args.ev_threshold:.2f}: n={t['n_bets']} "
                  f"hit={t['hit_rate']:.3f} CLV={t['mean_clv_pct']:.2f}% "
                  f"beat-close={t['pct_beating_close']:.1f}% "
                  f"ROI={t['roi_at_taken_odds_pct']:.1f}% "
                  f"Kelly final={t['staking']['quarter_kelly_dispersion_0']['final_bankroll_units']:.1f}u")
    print(f"\nSaved: {out_path}")
    print("Shots / shots on target: excluded by design (see FORWARD-TEST.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
