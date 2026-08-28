#!/usr/bin/env bash
set -euo pipefail
python -m unittest discover -s tests -v
python -m threshold_w2sg.cli run --config configs/publication.yaml
