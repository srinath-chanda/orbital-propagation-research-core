# Orbital Propagation Research Core

Research software and frozen evidence for a reproducible comparison of
low-Earth-orbit propagation models.

This repository accompanies the manuscript **“Reproducible Comparison and
Validation of Low-Earth-Orbit Propagation Models.”** It compares:

- analytical two-body propagation;
- numerical two-body propagation;
- numerical propagation with J2;
- numerical propagation with J2 and simplified exponential drag; and
- fixed-TLE SGP4 propagation.

The Paper 1 baseline is frozen. No additional force-model development is part
of this release.

## Recorded Paper 1 baseline

| Item | Recorded result |
|---|---:|
| Production experiments | 11 of 11 completed |
| Failed experiments | 0 |
| Primary model runs | 43 |
| Total executed model runs | 47 |
| Convergence candidates | 36 of 36 passed |
| GMAT drag scenarios | 4 of 4 passed |
| GMAT drag checks | 25 of 25 passed |

The production status is `passed_with_warnings`. These are scientific scope
warnings: the drag law is a controlled sensitivity model, fixed-TLE SGP4 is
not measured-orbit truth, and station passes are geometric.

Selected results include:

- 168 h J2 versus two-body maximum separation of about 5,056 km for LEO400
  and 1,487 km for SSO700;
- 168 h J2-plus-drag versus J2 maximum separation of about 846 km for LEO400
  and 5.65 km for SSO700; and
- at 72 h, final ISS-case separation from fixed-TLE SGP4 of about 3,303 km for
  two-body, 22.7 km for J2, and 10.7 km for J2 plus drag.

The values above are model-to-model differences under the registered test
conditions. They are not universal orbit-prediction errors.

## Install

Python 3.11 through 3.14 is supported.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Detailed setup notes are in [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Verify the repository

Run the complete automated test suite:

```bash
python -m unittest discover -s tests -v
```

Verify the frozen Paper 1 validation baseline:

```bash
python scripts/verify_paper1_baseline.py
```

Verify the public release files and SHA-256 manifest:

```bash
python scripts/verify_release.py
```

The validation commands do not require a new GMAT run. The closed GMAT
evidence used by the tests is included under `data/reference` and `results`.

## Reproduce the production matrix

The full matrix includes LEO400 and SSO700 cases at 6, 24, 72, and 168 h, plus
ISS fixed-TLE cases at 6, 24, and 72 h.

```bash
python scripts/prepare_paper1_production.py
python scripts/run_paper1_production.py
```

The run writes a new timestamped result set. It does not overwrite the frozen
validation evidence. Depending on the machine, the complete run can take
several minutes. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before rerunning it.

## Repository map

| Path | Contents |
|---|---|
| `src/research_core` | Propagation, analysis, comparison, and validation code |
| `configs` | Registered cases, matrices, and baseline closure |
| `scripts` | Command-line entry points |
| `tests` | Unit, integration, and evidence checks |
| `data/reference` | Frozen GMAT, gravity, atmosphere, and Earth-orientation inputs |
| `results` | Selected closed validation evidence required by the tests |
| `paper` | Manuscript preprint and publication figures and tables |
| `environment` | Recorded production environment |
| `docs` | Installation, reproducibility, scope, and validation notes |

## Scientific scope

The software is a research and educational comparison framework. It is not
flight-qualified navigation software and is not intended for operational
conjunction assessment, maneuver planning, or safety-critical decisions.
Model assumptions and frame handling are detailed in
[docs/VALIDATION_SCOPE.md](docs/VALIDATION_SCOPE.md).

## Manuscript and citation

The current author preprint is
[paper/manuscript/PAPER1_MANUSCRIPT_PREPRINT.pdf](paper/manuscript/PAPER1_MANUSCRIPT_PREPRINT.pdf).
Citation metadata is provided in [CITATION.cff](CITATION.cff).

The manuscript is not represented here as accepted or published. Repository
and archive identifiers should be added to the manuscript only after they are
publicly accessible.

## License and external material

Original repository materials are copyright 2026 Srinath Chanda, all rights
reserved. Limited permission to execute an unmodified copy for inspection and
research-result reproduction is stated in [LICENSE](LICENSE).

Third-party reference files and dependencies retain their own terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
