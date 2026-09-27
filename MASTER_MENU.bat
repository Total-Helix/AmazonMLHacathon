@echo off
color 0A

:: Check if Python 3.11 is installed to bypass the Python 3.14 PyTorch bug
python3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PY_CMD=python3.11
    set PIP_CMD=python3.11 -m pip
) else (
    set PY_CMD=python
    set PIP_CMD=pip
)

:MENU
cls
echo ========================================================
echo       AMAZON ML HACKATHON - MASTER PIPELINE
echo       Using Python Executable: %PY_CMD%
echo ========================================================
echo.
echo Who is running this script right now?
echo.
echo [1] Role 1 (GPU Preprocessing - Requires RTX 5050)
echo [2] Role 2 (FAISS Algorithmic Blocking)
echo [3] Role 3 (LightGBM Model Training & Inference)
echo [4] Role 4 (Final Zip Packager)
echo [5] Exit
echo.
set /p choice="Enter your Role number (1-5): "

if "%choice%"=="1" goto ROLE1
if "%choice%"=="2" goto ROLE2
if "%choice%"=="3" goto ROLE3
if "%choice%"=="4" goto ROLE4
if "%choice%"=="5" goto EOF

:ROLE1
cls
echo --- RUNNING ROLE 1 (GPU PREPROCESSING) ---
echo [*] Installing PyTorch with NVIDIA CUDA support...
%PIP_CMD% uninstall torch -y --quiet
%PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
%PIP_CMD% install pandas sentence-transformers numpy --quiet
%PY_CMD% role1_gpu_preprocessor.py
pause
goto MENU

:ROLE2
cls
echo --- RUNNING ROLE 2 (FAISS/PYTORCH BLOCKING) ---
echo [*] Installing PyTorch with NVIDIA CUDA support...
%PIP_CMD% uninstall torch -y --quiet
%PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
%PIP_CMD% install faiss-cpu pandas numpy --quiet
%PY_CMD% role2_faiss_blocking.py
pause
goto MENU

:ROLE3
cls
echo --- RUNNING ROLE 3 (LIGHTGBM ML) ---
%PIP_CMD% install lightgbm scikit-learn pandas --quiet
%PIP_CMD% install -r requirements.txt --quiet
%PY_CMD% main.py --use-real-data --train --infer
pause
goto MENU

:ROLE4
cls
echo --- RUNNING ROLE 4 (PACKAGING) ---
%PY_CMD% package.py
pause
goto MENU
