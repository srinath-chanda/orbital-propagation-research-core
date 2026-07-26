# Research Core 1D.0 GMAT gravity ladder

This folder is generated locally by `scripts/prepare_gmat_gravity_1d0.py` after
the installed GMAT R2026a `JGM2.cof` has been imported and checksum-recorded.

- `scripts/RUN_GRAVITY_LADDER_1D0.script`: run once in GMAT R2026a.
- `output/GMAT_GRAVITY_LADDER_1D0.csv`: expected untouched raw report.
- `GMAT_GRAVITY_LADDER_1D0_MANIFEST.json`: preparation inventory.

The six levels are 0/0, 2/0, 4/0, 4/4, 8/8, and 20/20. All accelerations are
evaluated at the same 25 spacecraft states. Do not edit the raw output or the
preregistered thresholds after running GMAT.
