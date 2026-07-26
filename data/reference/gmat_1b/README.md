# Research Core 1B GMAT matrix data

Run `python scripts/prepare_gmat_1b_matrix.py` on the Windows computer that will
run GMAT. This regenerates absolute output paths for that checkout.

- `cases/` contains the ten resolved case configurations.
- `scripts/RUN_ALL_CASES_1B.script` is the preferred one-run GMAT script.
- `scripts/*_TWO_BODY.script` and `scripts/*_J2.script` are safe fallbacks.
- `output/` receives the twenty untouched GMAT STK-TimePosVel ephemerides.
- `GMAT_1B_MATRIX_MANIFEST.json` records configurations, hashes, thresholds,
  and expected outputs.
- `RUN_ORDER_1B.txt` records the individual-script fallback order.

Preparation moves matching old ephemerides into `archive/<UTC timestamp>/`
instead of deleting or silently reusing them. Do not manually edit returned
GMAT ephemerides or change a threshold after viewing results.
