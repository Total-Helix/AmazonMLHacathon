@echo off
echo =========================================================
echo      AMAZON ML CHALLENGE: ROLE 2 - ALGORITHMIC BLOCKING
echo =========================================================
echo.
echo Hello! This script runs the high-speed FAISS search engine.
echo Make sure Role 1 gave you the 'data/clean_data/' folder first!
echo.
echo [Step 1] Checking Python...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b
)

echo [Step 2] Installing FAISS and Pandas...
pip install faiss-cpu pandas numpy --quiet
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install libraries. Check your internet.
    pause
    exit /b
)

echo.
echo [Step 3] Booting FAISS Search Engine...
echo Running the algorithmic blocking across millions of records.
echo.
python role2_faiss_blocking.py

echo.
echo =========================================================
echo Blocking Complete! You can close this window.
echo Give the 'output/candidate_pairs.tsv' file to Role 3 (Tanuj)!
echo =========================================================
pause
