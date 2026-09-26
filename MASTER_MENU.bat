@echo off
color 0A
:MENU
cls
echo ========================================================
echo       AMAZON ML HACKATHON - MASTER PIPELINE
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
pip uninstall torch -y --quiet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install pandas sentence-transformers numpy --quiet
python role1_gpu_preprocessor.py
pause
goto MENU

:ROLE2
cls
echo --- RUNNING ROLE 2 (FAISS BLOCKING) ---
pip install faiss-cpu pandas numpy --quiet
python role2_faiss_blocking.py
pause
goto MENU

:ROLE3
cls
echo --- RUNNING ROLE 3 (LIGHTGBM ML) ---
pip install lightgbm scikit-learn pandas --quiet
python main.py --use-real-data --train --infer
pause
goto MENU

:ROLE4
cls
echo --- RUNNING ROLE 4 (PACKAGING) ---
python package.py
pause
goto MENU
