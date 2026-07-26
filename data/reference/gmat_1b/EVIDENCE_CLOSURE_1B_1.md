# Research Core 1B.1 evidence closure

The files already present in `cases/`, `scripts/`, and `output/` are the exact
inputs and outputs from the completed GMAT R2026a matrix. They intentionally
retain Research Core 1B.0 identifiers and the original Windows paths.

`GMAT_1B_RESULTS_PACKAGE_MANIFEST.json` is the exact inventory from the
user-created evidence ZIP. It tracks 60 files, including all twenty raw
ephemerides, by byte size and SHA-256. Do not edit any tracked evidence file.

The official aggregate result is under
`results/EXP-GMAT-1B-MULTICASE-001/2026-07-19_123533_125158Z/`. The
machine-readable closure is
`results/EXP-GMAT-1B-MULTICASE-001/GMAT_VALIDATION_CLOSURE_1B_1.json`.

Running `prepare_gmat_1b_matrix.py` begins a new run: it archives matching
current ephemerides and regenerates scripts for the active checkout.
