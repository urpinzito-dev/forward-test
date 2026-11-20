"""
Forward-test / paper trading — Daniel / Modelo de Predicciones de Fútbol
=========================================================================
~/workspace/lightgbm-model/forward_test.py

WHY THIS EXISTS
---------------
The free tiers of API-Football / The Odds API cannot backfill 7,400 past
matches of odds history (odds history is a paid feature). The professional
fix used by quants with no free odds history is a FORWARD TEST: capture
real odds live, every day, and evaluate after 3-4 weeks (150-200 matches)
with zero look-ahead bias and $0 of cost.

WHAT IT DOES
------------
* Morning capture (10:00 America/Caracas): today's fixtures in the 10
  modelled leagues + morning odds for the 4 focus markets.
* Closing capture (every 30 min, only for matches kicking off in the
  next 60 min): the same odds again, ~15 min before kickoff. This is
  the "closing" line for CLV.
* Saves one row per match x market in data/forward_test/YYYY-MM-DD.csv.

FOCUS MARKETS ONLY (Daniel's decision, 2026-11-20):
  * corners_away_over45  — Corners away > 4.5   (best log loss 0.645)
  * corners_home_over45  — Corners home > 4.5   (best log loss 0.667)
  * cards_home_over15    — Cards home > 1.5     (best log loss 0.673)
  * cards_over35         — Cards total > 3.5     (best log loss 0.676)
Shots / shots on target are OFF COMPLETELY: their log loss is at the
always-favourite baseline, so they only add complexity and noise.
They do not appear anywhere in this file.

KEYS (never hardcoded, never invented):
  * API_FOOTBALL_KEY — free account at api-football.com
    (dashboard.api-football.com, no credit card). 100 requests/day,
    10/min. Header: x-apisports-key.
  * ODDS_API_KEY — optional. The Odds API v4 has NO corners/cards
    markets (only h2h / spreads / totals), so for these 4 markets it
    is always skipped with 0 requests, key or no key. It is checked
    only to document that honestly.
If API_FOOTBALL_KEY is not set, the script prints "skipped, no key"
and exits 0 having made 0 requests.

MODEL PROBABILITIES (honest gap):
  The feature-builder bridge that scores FUTURE matches does not
  exist yet (predict.py scores LightGBM-only models from a feature
  row; it does not build features from raw matches). So this script
  reads model probabilities from an optional file
  data/forward_test/model_probs_YYYY-MM-DD.json, format:
      {"<fixture_id>": {"corners_home_over45": 0.58, ...}, ...}
  If that file (or a market) is missing, the odds are still saved
  with model_p EMPTY. Probabilities are NEVER invented. Once the
  bridge exists, it should write that JSON every morning before
  this script runs.

QUOTA GUARD: free API-Football = 100 req/day. The script counts its
own requests in data/forward_test/_quota_YYYY-MM-DD.json and stops
at 90, so the morning + closing passes never burn the whole quota.
Morning pass: 1 /fixtures?date= call + odds calls (bulk /odds?date=
first, per-fixture fallback capped). Closing pass: only fixtures
kicking off in the next 60 minutes, so usually 0-6 calls per run.

Usage:
  cd ~/workspace/lightgbm-model
  API_FOOTBALL_KEY=... ./venv/bin/python forward_test.py --mode morning
  API_FOOTBALL_KEY=... ./venv/bin/python forward_test.py --mode closing
  ./venv/bin/python forward_test.py --mode morning --dry-run   # no requests
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data", "forward_test")
CARACAS = ZoneInfo("America/Caracas") if ZoneInfo else timezone(timedelta(hours=-4))

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
# API-Football league ids for the 10 leagues in the model (verified in
# SOURCES-AUDIT.md as the league-id mapping to use; re-check at signup).
LEAGUES = {
    39: "E0",    # Premier League
    140: "SP1",  # LaLiga
    135: "I1",   # Serie A
    78: "D1",    # Bundesliga
    61: "F1",    # Ligue 1
    88: "N1",    # Eredivisie
    94: "P1",    # Liga Portugal
    203: "T1",   # Super Lig
    144: "B1",   # Jupiler Pro League
    197: "G1",   # Super League 1 (Greece)
}
FOCUS_MARKETS = [
    "corners_away_over45",
    "corners_home_over45",
    "cards_home_over15",
    "cards_over35",
]
MARKET_LABELS = {
    "corners_away_over45": "Corners away > 4.5",
    "corners_home_over45": "Corners home > 4.5",
    "cards_home_over15": "Cards home > 1.5",
    "cards_over35": "Cards total > 3.5",
}
BEST_LOG_LOSS = {
    "corners_away_over45": 0.645,
    "corners_home_over45": 0.667,
    "cards_home_over15": 0.673,
    "cards_over35": 0.676,
}
CSV_COLUMNS = [
    "date", "match_id", "source", "home", "away", "league", "kickoff",
    "market", "model_p", "odds_morning", "odds_closing",
    "won", "corners_home", "corners_away", "cards_home", "cards_total",
    "captured_morning_at", "captured_closing_at",
]
DAILY_REQUEST_CAP = 90  # free tier is 100/day; keep a 10-request margin


# ------------------------------------------------------------------ helpers
def now_caracas():
    return datetime.now(CARACAS)


def quota_path(date_str):
    return os.path.join(OUT_DIR, f"_quota_{date_str}.json")


def quota_used(date_str):
    try:
        return int(json.load(open(quota_path(date_str))).get("requests", 0))
    except (OSError, ValueError, KeyError):
        return 0


def quota_bump(date_str):
    n = quota_used(date_str) + 1
    with open(quota_path(date_str), "w") as f:
        json.dump({"date": date_str, "requests": n}, f)
    return n


class ApiClient:
    """Minimal API-Football client (stdlib only). Counts every request."""

    def __init__(self, key, date_str, dry_run=False):
        self.key = key
        self.date_str = date_str
        self.dry_run = dry_run
        self.requests_made = 0

    def get(self, path):
        if self.dry_run:
            print(f"  [dry-run] would call GET {path}")
            return None
        if quota_used(self.date_str) >= DAILY_REQUEST_CAP:
            print(f"  quota cap reached ({DAILY_REQUEST_CAP}/day) — stopping, "
                  "remaining fixtures skipped, not failed")
            return None
        url = API_FOOTBALL_BASE + path
        req = urllib.request.Request(url, headers={
            "x-apisports-key": self.key,
            "Accept": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} on {path} — logged, continuing")
            return None
        except Exception as e:  # network error, timeout, bad JSON
            print(f"  request failed on {path}: {e} — logged, continuing")
            return None
        quota_bump(self.date_str)
        self.requests_made += 1
        time.sleep(0.6)  # free tier: 10 req/min; stay well under it
        return data


# ------------------------------------------------------------------ odds parsing
def _norm(s):
    return str(s or "").strip().lower()


def _over_odds_for(bets, want):
    """Average decimal odds across bookmakers for one selection.

    `want` is a predicate over (bet_name, value_label). API-Football
    bet/value labels vary by bookmaker, so several spellings are
    tried; if none match, returns None (market not offered — logged,
    never invented). Returns (avg_odds, n_bookmakers, matched_label).
    """
    found = []
    for bet in bets or []:
        name = _norm(bet.get("name"))
        for val in bet.get("values") or []:
            label = _norm(val.get("value"))
            try:
                odd = float(val.get("odd"))
            except (TypeError, ValueError):
                continue
            if odd <= 1.0:
                continue
            if want(name, label):
                found.append((odd, f"{bet.get('name')} / {val.get('value')}"))
    if not found:
        return None
    avg = sum(o for o, _ in found) / len(found)
    return round(avg, 3), len(found), found[0][1]


def extract_market_odds(bets, home, away, market):
    """Map one focus market to API-Football odds. Returns the
    (avg_odds, n_bookmakers, matched_label) tuple or None."""
    h, a = _norm(home), _norm(away)
    if market == "corners_home_over45":
        def want(n, v):
            return "corner" in n and "over 4.5" in v and (
                "home" in n or "home" in v or h in v)
    elif market == "corners_away_over45":
        def want(n, v):
            return "corner" in n and "over 4.5" in v and (
                "away" in n or "away" in v or a in v)
    elif market == "cards_home_over15":
        def want(n, v):
            return ("card" in n or "booking" in n) and "over 1.5" in v and (
                "home" in n or "home" in v or h in v)
    elif market == "cards_over35":
        def want(n, v):
            return ("card" in n or "booking" in n) and "over 3.5" in v and (
                "home" not in n and "away" not in n
                and "home" not in v and "away" not in v)
    else:
        return None
    return _over_odds_for(bets, want)


# ------------------------------------------------------------------ snapshots
def load_snapshot(date_str):
    path = os.path.join(OUT_DIR, f"{date_str}.csv")
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    return path, rows


def save_snapshot(path, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})


def load_model_probs(date_str):
    """Optional bridge output. Missing file/market -> model_p empty."""
    path = os.path.join(OUT_DIR, f"model_probs_{date_str}.json")
    if not os.path.exists(path):
        print(f"  no model_probs file ({os.path.basename(path)}): model_p "
              "will be empty — the feature-builder bridge is still needed "
              "to fill it. Odds are still saved; nothing is invented.")
        return {}
    try:
        return json.load(open(path))
    except (OSError, ValueError) as e:
        print(f"  model_probs file unreadable ({e}): model_p empty")
        return {}


def todays_fixtures(client, date_str):
    """All of today's fixtures in the 10 modelled leagues."""
    data = client.get(
        f"/fixtures?date={date_str}&timezone=America/Caracas")
    if not data:
        return []
    out = []
    for item in data.get("response", []):
        league_id = item.get("league", {}).get("id")
        if league_id not in LEAGUES:
            continue
        fx = item.get("fixture", {})
        out.append({
            "match_id": str(fx.get("id")),
            "home": item.get("teams", {}).get("home", {}).get("name", ""),
            "away": item.get("teams", {}).get("away", {}).get("name", ""),
            "league": LEAGUES[league_id],
            "kickoff": str(fx.get("date", "")),
        })
    return out


def fetch_odds_bulk(client, date_str):
    """One /odds?date= call for the whole day. Returns
    {fixture_id: bets} or None if the endpoint does not support it."""
    data = client.get(f"/odds?date={date_str}&timezone=America/Caracas")
    if not data or not data.get("response"):
        return None
    out = {}
    for item in data.get("response", []):
        fid = str(item.get("fixture", {}).get("id"))
        out[fid] = item.get("bookmakers", [])
    return out


def fetch_odds_fixture(client, fixture_id):
    data = client.get(f"/odds?fixture={fixture_id}")
    if not data or not data.get("response"):
        return []
    return data["response"][0].get("bookmakers", [])


# ------------------------------------------------------------------ modes
def run_morning(client, date_str, dry_run):
    print(f"Morning capture — {date_str} (10:00 America/Caracas)")
    fixtures = todays_fixtures(client, date_str)
    print(f"  fixtures today in the 10 leagues: {len(fixtures)}")
    if not fixtures:
        print("  no fixtures — nothing to capture")
        return
    model_probs = load_model_probs(date_str)
    path, rows = load_snapshot(date_str)
    existing = {(r["match_id"], r["market"]) for r in rows}
    stamp = now_caracas().isoformat(timespec="seconds")

    bulk = fetch_odds_bulk(client, date_str)
    if bulk is not None:
        print(f"  bulk /odds?date= worked: odds for {len(bulk)} fixtures")
    else:
        print("  bulk /odds?date= returned nothing — falling back to "
              "per-fixture /odds calls (capped by the daily quota guard)")

    for fx in fixtures:
        bets = bulk.get(fx["match_id"]) if bulk is not None else None
        if bets is None:
            bets = fetch_odds_fixture(client, fx["match_id"])
        for market in FOCUS_MARKETS:
            key = (fx["match_id"], market)
            if key in existing:
                continue
            found = extract_market_odds(bets, fx["home"], fx["away"], market)
            probs = model_probs.get(fx["match_id"], {})
            if isinstance(probs, dict):
                mp = probs.get(market, "")
            else:
                mp = ""
            rows.append({
                "date": date_str, "match_id": fx["match_id"],
                "source": "api-football",
                "home": fx["home"], "away": fx["away"],
                "league": fx["league"], "kickoff": fx["kickoff"],
                "market": market, "model_p": mp,
                "odds_morning": found[0] if found else "",
                "odds_closing": "", "won": "",
                "corners_home": "", "corners_away": "",
                "cards_home": "", "cards_total": "",
                "captured_morning_at": stamp, "captured_closing_at": "",
            })
            if found:
                print(f"  {fx['home']} vs {fx['away']} [{market}]: "
                      f"morning {found[0]} ({found[1]} books, {found[2]})")
            else:
                print(f"  {fx['home']} vs {fx['away']} [{market}]: "
                      "no morning odds found for this market — row saved "
                      "with odds empty, not invented")
    if not dry_run:
        save_snapshot(path, rows)
    print(f"  saved {len(rows)} rows to {path} "
          f"({client.requests_made} API requests this run)")


def run_closing(client, date_str, dry_run):
    print(f"Closing capture — {date_str} (matches kicking off soon)")
    path, rows = load_snapshot(date_str)
    if not rows:
        print("  no morning snapshot for today — run --mode morning first "
              "(closing odds need the morning rows to update)")
        return
    now = now_caracas()
    window_end = now + timedelta(minutes=60)
    todo = {}
    for r in rows:
        if r.get("odds_closing"):
            continue
        try:
            kickoff = datetime.fromisoformat(r["kickoff"])
        except ValueError:
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=CARACAS)
        if now <= kickoff <= window_end:
            todo[r["match_id"]] = r
    print(f"  matches kicking off in the next 60 min: {len(todo)}")
    if not todo:
        print("  nothing to capture — 0 requests made")
        return
    stamp = now.isoformat(timespec="seconds")
    for fid, first in todo.items():
        bets = fetch_odds_fixture(client, fid)
        for r in rows:
            if r["match_id"] != fid:
                continue
            found = extract_market_odds(
                bets, r["home"], r["away"], r["market"])
            if found:
                r["odds_closing"] = found[0]
                r["captured_closing_at"] = stamp
                print(f"  {r['home']} vs {r['away']} [{r['market']}]: "
                      f"closing {found[0]}")
            else:
                print(f"  {r['home']} vs {r['away']} [{r['market']}]: "
                      "no closing odds found — morning odds kept")
    if not dry_run:
        save_snapshot(path, rows)
    print(f"  saved {len(rows)} rows to {path} "
          f"({client.requests_made} API requests this run)")


def main():
    ap = argparse.ArgumentParser(description="Forward-test odds capture")
    ap.add_argument("--mode", choices=["morning", "closing"],
                    required=True)
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD, default today in America/Caracas")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be fetched, make 0 requests")
    args = ap.parse_args()

    date_str = args.date or now_caracas().strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)

    # The Odds API: documented skip, 0 requests. Its v4 markets are
    # h2h / spreads / totals only — no corners, no cards — so it
    # cannot supply odds for any of the 4 focus markets.
    if os.environ.get("ODDS_API_KEY"):
        print("The Odds API: key present, but v4 has no corners/cards "
              "markets — skipped, 0 requests.")
    else:
        print("The Odds API: no key and no corners/cards markets — skipped.")

    key = os.environ.get("API_FOOTBALL_KEY")
    if not key and not args.dry_run:
        print("skipped, no key: set API_FOOTBALL_KEY (free account at "
              "api-football.com) — 0 requests made, exit 0")
        return 0
    if args.dry_run:
        print("[dry-run] no requests will be made, no files written")
        key = key or "dry-run"
    client = ApiClient(key, date_str, dry_run=args.dry_run)
    if args.mode == "morning":
        run_morning(client, date_str, args.dry_run)
    else:
        run_closing(client, date_str, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
