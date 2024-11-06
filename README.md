# P2P Lending

## How to run

Install poetry: `make install_poetry`

Create virtual environment: `python -m venv venv`

Activate virtual environment: `source venv/bin/activate` or `venv\Scripts\Activate` on Windows

Install dependencies: `poetry install --verbose` (verbose outputs the environment used, make sure it's correct)

Run the script: `python p2p_lending/main.py`

## Check mypy

Run mypy: `mypy .`

## Get mypy errors in VSCode

Install mypy extension in VSCode by Microsoft

Change Mypy-type-checker: Import Strategy setting to fromEnvironment
