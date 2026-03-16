@echo off
chcp 65001 >nul
title CogniFlex

echo ========================================
echo    CogniFlex AI System
echo ========================================
echo.

cd /d "%~dp0"

if exist ".venv311\Scripts\activate.bat" (
    call .venv311\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found
    pause
    exit /b 1
)

python -m cogniflex.run

pause
