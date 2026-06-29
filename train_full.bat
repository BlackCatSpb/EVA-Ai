@echo off
chcp 65001 >nul
cd /d "C:\Users\black\OneDrive\Desktop\FCF"
echo =============================================
echo  EVA Full Training   ^|   STDP + Collocation
echo =============================================
echo.
echo  Corpus:  full_corpus_ru_clean.txt (1.6 GB)
echo  Log:     train_full.log
echo  Output:  checkpoints\
echo.
echo  * Training runs in background
echo  * Check train_full.log for progress
echo  * Press Ctrl+C to stop
echo.
echo Starting...
echo.
python train_full.py
echo.
echo Finished. Check train_full.log
pause
