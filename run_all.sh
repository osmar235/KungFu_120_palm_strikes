#!/usr/bin/env bash
# Reproduces every number and figure of the manuscript from the de-identified data.
set -euo pipefail
cd "$(dirname "$0")/code"
python pipeline.py ../data ../results
python pipeline_sensitivity_v1_1.py ../data ../results
python pipeline_sensitivity_v1_2.py ../results
python pipeline_timing_v1_3.py ../data ../results
python pipeline_timing_robust_v1_4.py ../results
python pipeline_rise_v1_3.py ../data ../results
python audit_long_bouts.py ../data ../results/audit_long_bouts
python make_figures.py ../data ../results
python make_figure5_v1_3.py ../results
echo "Done: see results/"
