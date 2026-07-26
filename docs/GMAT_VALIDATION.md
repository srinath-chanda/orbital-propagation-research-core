# GMAT validation evidence

The repository contains closed comparison evidence prepared with GMAT R2026a.
Routine tests and Paper 1 production do not invoke GMAT.

## Included validation stages

| Stage | Purpose | Closed evidence |
|---|---|---|
| 1B | Ten-case two-body and J2 matrix | `EXP-GMAT-1B-MULTICASE-001` |
| 1C.1 | Earth-orientation diagnostics | `EXP-GMAT-EOP-1C1-001` |
| 1C.2 | Independent EOP multicase check | `EXP-GMAT-EOP-1C2-INDEPENDENT-001` |
| 1D.0 | Higher-order gravity ladder | `EXP-GMAT-GRAVITY-1D0-001` |
| 1D.1 | Higher-order gravity short arc | `EXP-GMAT-GRAVITY-1D1-SHORT-ARC-001` |
| 1D.2 | Higher-order gravity multicase | `EXP-GMAT-GRAVITY-1D2-MULTICASE-001` |
| 1E.0 | Exponential-drag acceleration | `EXP-GMAT-DRAG-1E0-ACCELERATION-001` |

The 1E.0 gate closed with 4 of 4 scenarios and 25 of 25 checks. Its maximum
accepted raw time-grid residual was about 1.397 microseconds against a
5-microsecond synchronization limit.

## Portable GMAT scripts

Generated GMAT script paths in this release are repository-relative. Run
preparation commands from the repository root. GMAT itself is not bundled.

Repeating external validation creates new files and should not overwrite the
closed evidence used by `configs/paper1_baseline_closure.json`. Preserve the
original package and write reruns to a separate working copy.

Reference-file provenance is recorded under `data/reference/gmat_r2026a`.
Third-party terms are summarized in `THIRD_PARTY_NOTICES.md`.
