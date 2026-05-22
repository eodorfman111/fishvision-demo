@echo off
title FishVision Demo Launcher
color 0B

echo.
echo  =========================================
echo   FishVision - Underwater Fish Detection
echo  =========================================
echo.

:: Check if model exists (accept v1.04 or v1.03 for backwards compat)
if exist "%~dp0models\best_v1.04.pt" goto model_ok
if exist "%~dp0models\best_v1.03.pt" goto model_ok

echo  [ERROR] Model not found!
echo  Please copy best_v1.04.pt into the models\ folder.
echo  Expected location: %~dp0models\best_v1.04.pt
echo.
pause
exit /b 1

:model_ok
:: Create static folder for demo video if it doesn't exist
if not exist "%~dp0static\" mkdir "%~dp0static"

:: Install dependencies if needed (silently after first time)
echo  Checking dependencies...
pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing dependencies for the first time, this may take a few minutes...
    pip install streamlit ultralytics opencv-python-headless plotly pandas reportlab torch
    echo  Dependencies installed!
) else (
    echo  Dependencies already installed.
)

echo.
echo  Starting FishVision... your browser will open automatically.
echo  Tip: drop a demo video at static\demo.mp4 to enable the auto-play preview.
echo  To stop the app, close this window or press Ctrl+C
echo.

:: Run from the script's own directory so paths resolve correctly
cd /d "%~dp0"
streamlit run demo.py --server.headless false

pause
