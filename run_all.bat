@echo off
REM Reproduces every number and figure of the manuscript from the de-identified data.
set PYTHONUTF8=1
cd /d "%~dp0code"
python pipeline.py ..\data ..\results || exit /b 1
python pipeline_sensitivity_v1_1.py ..\data ..\results || exit /b 1
python pipeline_sensitivity_v1_2.py ..\results || exit /b 1
python pipeline_timing_v1_3.py ..\data ..\results || exit /b 1
python pipeline_timing_robust_v1_4.py ..\results || exit /b 1
python pipeline_rise_v1_3.py ..\data ..\results || exit /b 1
python audit_long_bouts.py ..\data ..\results\audit_long_bouts || exit /b 1
python make_figures.py ..\data ..\results || exit /b 1
python make_figure5_v1_3.py ..\results || exit /b 1
echo Done: see results\
