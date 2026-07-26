# Research Core 1D.1 GMAT short arcs

Run `scripts/RUN_GRAVITY_SHORT_ARCS_1D1.script` once in GMAT R2026a after
regenerating the scripts locally with `scripts/prepare_gmat_gravity_1d1.py`.

Expected untouched outputs:

- `output/G20_SHORT_ARC.e`
- `output/G44_SHORT_ARC.e`
- `output/G88_SHORT_ARC.e`
- `output/G2020_SHORT_ARC.e`

Every arc is 1800 seconds with 10-second STK-TimePosVel output. Do not edit the
raw ephemerides or preregistered thresholds after running GMAT.
