# GMAT reference data

Regenerate scripts before every local GMAT run so output paths match the active
checkout:

```powershell
python scripts\prepare_gmat_validation.py configs\case_leo400_gmat_matched.json
```

Run the validation ladder in this order:

1. `CASE_LEO400_GMAT_ACCELERATION_DIAGNOSTIC.script`, followed by
   `python scripts\run_gmat_acceleration_validation.py`.
2. `CASE_LEO400_GMAT_J2_SHORT_ARC.script`, followed by
   `python scripts\run_gmat_j2_short_arc_validation.py`.
3. `CASE_LEO400_GMAT_TWO_BODY.script` and `CASE_LEO400_GMAT_J2.script`, followed
   by `python scripts\run_external_validation.py`.

Generated scripts are stored in `scripts/`. Untouched GMAT reports and
ephemerides belong in `output/`. Every Python runner creates timestamped results
under `results/` and records source checksums in the run manifest.

Research Core 1A.8.3 imports the untouched R2026a acceleration report directly,
including repeated headers and the observed single-character `C` delimiter.
Do not create a manually normalized copy for new runs.

The checked-in scripts and preparation metadata keep their 1A.8.2 identifiers
because their exact checksums belong to the archived real run. Regenerating the
scripts starts a new single-case run under the current software version; it does
not alter the historical result.

Do not manually edit a GMAT reference file without recording its origin,
software version, configuration change, and checksum.
