# Forward-Test / Paper Trading en vivo — córners y tarjetas

## Por qué existe esto

Las cuotas de córners y tarjetas **no existen** en los CSVs de
football-data.co.uk, así que el backtest financiero no las pudo medir.
Y las capas gratis de las APIs de cuotas **no dejan descargar el
historial** de 7.400 partidos pasados: el historial de cuotas es de
pago.

La solución profesional es un **forward-test**: capturar las cuotas
reales en vivo, cada día, y evaluar después de 3–4 semanas
(150–200 partidos), con cero sesgo retrospectivo y $0 de costo.

## Los 4 mercados que se miden (y nada más)

| Mercado | Log loss en test |
|---|---|
| Córners visita más de 4,5 | 0,645 |
| Córners local más de 4,5 | 0,667 |
| Tarjetas local más de 1,5 | 0,673 |
| Tarjetas total más de 3,5 | 0,676 |

**Disparos y tiros a puerta están APAGADOS por completo.** Su log
loss está al nivel de apostar siempre al favorito: no aportan nada y
solo añaden complejidad y ruido al enrutador. No aparecen en
`forward_test.py` ni en `evaluate_forward_test.py`.

## Qué se captura cada día

- **10:00 (hora de Caracas):** los partidos del día en las 10 ligas
  del modelo + la cuota de la mañana de los 4 mercados.
- **Cada 30 minutos:** solo los partidos que empiezan en los próximos
  60 minutos → la cuota de cierre (~15 min antes del partido).
- Se guarda en `data/forward_test/AAAA-MM-DD.csv`, una fila por
  partido y mercado: fecha, id del partido, equipos, liga, hora,
  mercado, probabilidad del modelo, cuota de la mañana, cuota de
  cierre, y luego el resultado (córners y tarjetas reales).

Con eso se calcula igual que en `financial_backtest.py`: EV =
(probabilidad × cuota de la mañana) − 1, CLV = cuota de la mañana /
cuota de cierre − 1, ROI y Quarter Kelly.

## Qué necesita Daniel (solo esto)

### 1. Una clave gratis de API-Football

- Crear cuenta gratis en **api-football.com** → panel en
  **dashboard.api-football.com**. No pide tarjeta de crédito.
- Plan gratis: 100 peticiones al día, 10 por minuto. El script tiene
  un tope propio de 90 al día y duerme entre peticiones, así que no
  se pasa.
- **La clave no va en ningún archivo.** Va como secreto del
  repositorio (abajo) o como variable de entorno en tu PC.

### 2. Un repositorio en GitHub (gratis)

1. Crear un repositorio en GitHub y subir esta carpeta
   `lightgbm-model/` (el workflow supone que el repo contiene la
   carpeta `lightgbm-model/`; si el repo **es** directamente esa
   carpeta, quita el `cd lightgbm-model` y el prefijo
   `lightgbm-model/` de las rutas en
   `.github/workflows/forward-test.yml`).
2. En el repo: **Settings → Secrets and variables → Actions → New
   repository secret**. Nombre exacto: `API_FOOTBALL_KEY`. Valor: la
   clave gratis de API-Football.
3. Listo. A partir de ahí corre solo, todos los días.

> Muse no puede crear el repositorio ni el secreto por ti: GitHub
> pide tu cuenta y tu clave. Son los dos únicos pasos manuales.

### 3. (Opcional) The Odds API — no hace falta

The Odds API **no tiene mercados de córners ni de tarjetas**
(solo 1X2, hándicaps y totales de goles), así que para estos 4
mercados siempre se salta, con 0 peticiones, haya clave o no. No
gastes tiempo en esa clave para esto.

## Qué corre automático

| Cuándo | Qué hace |
|---|---|
| 10:00 Caracas (14:00 UTC) | Captura de la mañana: partidos del día + cuotas |
| Cada 30 min, 15:00–02:00 UTC | Captura de cierre: solo partidos por empezar |
| Después de cada captura | Guarda el CSV en el repo (commit automático) |

Costo: $0. GitHub Actions gratis da 2.000 minutos al mes; esto usa
unos 5 minutos al día. API-Football gratis: 100 peticiones al día;
la captura de la mañana usa ~1 + 1 por partido (con tope de 90) y
la de cierre 0–6 por pasada.

## La probabilidad del modelo: el hueco honesto

`forward_test.py` **no inventa probabilidades**. Las lee de un
archivo opcional `data/forward_test/model_probs_AAAA-MM-DD.json`:

```json
{"1234567": {"corners_home_over45": 0.58, "cards_over35": 0.61}}
```

Si ese archivo no existe (hoy no existe), las cuotas se guardan
igual con `model_p` vacío. Para llenarlo hace falta el puente que
construye las 62 variables de un partido futuro y lo puntúa con la
mejor versión por mercado — ese puente todavía no está construido
(es el bloqueador ya conocido). Cuando exista, debe escribir ese
JSON cada mañana **antes** de la captura de las 10:00.

## Cómo evaluar (después de 3–4 semanas)

```bash
cd ~/workspace/lightgbm-model
./venv/bin/python evaluate_forward_test.py --settle
```

- `--settle` busca los resultados (córners y tarjetas reales) en
  API-Football. O pasa un CSV manual:
  `./venv/bin/python evaluate_forward_test.py --results resultados.csv`
  con columnas `match_id, corners_home, corners_away, cards_home,
  cards_total`.
- **Cada mercado se reporta solo con n ≥ 30 apuestas liquidadas.**
  Con menos, el script dice "not enough sample yet" y **no** cita
  ningún ROI. Con 150–200 partidos en total habrá muestra suficiente
  en los 4 mercados.
- El resultado se guarda en `data/forward_test/evaluation.json` y se
  imprime en pantalla: acierto, CLV medio, % que bate el cierre, ROI
  y Kelly por mercado, con EV ≥ 3% y con la ventana de EV 2–8%.

**La regla para sacar un mercado del laboratorio sigue siendo la
misma:** CLV > +2%, más de 55% batiendo el cierre y ROI positivo
fuera de muestra. Si no se cumple, el mercado se queda en
Beta / Laboratorio.

## Archivos nuevos (nada más se tocó)

- `forward_test.py` — captura de cuotas (solo librería estándar,
  sin dependencias nuevas).
- `evaluate_forward_test.py` — liquidación y evaluación financiera.
- `.github/workflows/forward-test.yml` — los dos trabajos
  automáticos de GitHub Actions.
- `FORWARD-TEST.md` — este archivo.
- `data/forward_test/` — los CSV de snapshots (se crean solos).

No se tocó ningún modelo, ningún hiperparámetro, ningún resultado
anterior ni ningún otro script.
