#!/usr/bin/env bash
set -euo pipefail

rm -rf build dist *.egg-info

python -m venv .venv-build
source .venv-build/bin/activate

python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
ls -lh dist/

# Test the wheel in a clean environment
deactivate
python -m venv .venv-test
source .venv-test/bin/activate

python -m pip install --upgrade pip
python -m pip install dist/*.whl

python -c "import astrologics; print('Import successful'); print(astrologics.__file__)"
python -m pip check