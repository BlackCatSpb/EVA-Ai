@echo off
chcp 65001 >nul
cd /d %~dp0

echo === FMF EVA Container ===
echo.

python -m src.fmf_cli %*
