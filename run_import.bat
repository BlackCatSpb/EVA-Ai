@echo off
chcp 65001 >nul
cd /d C:\Users\black\OneDrive\Desktop\EVA-Ai
set PYTHONPATH=C:\Users\black\OneDrive\Desktop\EVA-Ai
echo === Starting ConceptNet Import ===
venv\Scripts\python.exe import_conceptnet_to_graph.py
echo.
echo === Import Complete (exit code %errorlevel%) ===
pause
