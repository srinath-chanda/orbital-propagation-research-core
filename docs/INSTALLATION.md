# Installation

## Supported Python versions

Use CPython 3.11, 3.12, 3.13, or 3.14. The recorded Paper 1 production
environment used Python 3.14.5 on Windows 11.

## Windows PowerShell

From the repository root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import research_core; print(research_core.RESEARCH_CORE_VERSION)"
```

Expected version:

```text
0.1.0
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

## Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Deterministic Earth-orientation behavior

The repository includes the Earth-orientation record used for the closed
validation evidence. Automated tests do not require a network download.
Applications that request current Earth-orientation data must manage those
updates separately and record the exact input file.

## GMAT

GMAT is not required for normal installation, unit tests, baseline
verification, or production-matrix reproduction. It is needed only to repeat
the independent external-validation generation steps. Those optional steps
were prepared for GMAT R2026a.
