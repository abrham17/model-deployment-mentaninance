#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pip install scikit-learn
pip install -r requirements.frontend.txt
python train_tf.py
