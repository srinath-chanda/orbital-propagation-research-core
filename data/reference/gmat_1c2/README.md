# Research Core 1C.2 independent GMAT matrix

This folder contains six preregistered holdout configurations and paired GMAT
R2026a point-mass/J2 scripts. None of the complete initial conditions and
durations duplicates a 1B/1C.1 selection case.

The real GMAT execution is complete. All twelve STK-TimePosVel ephemerides are
preserved in `output/`, and the official passed result is stored under
`results/EXP-GMAT-EOP-1C2-INDEPENDENT-001/2026-07-19_191123_017338Z`.

Do not rerun the master script for the closed evidence. Research Core 1C.3
verifies the saved case summaries, all 84 checks, raw source hashes, and 149
manifest records before adopting the full-EOP baseline.

Do not edit the matrix configuration, generated cases, thresholds, EOP file,
scripts, or raw outputs. The closed evidence must remain byte-reproducible.

The tagged EOP source must have SHA-256
`52c95d6871066e892463328424ca34f5e9cedde92c747465a40f7cd0ecd86ed3`.
