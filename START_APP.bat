@echo off
title CardioGuard AI - Server + ngrok
echo.
echo ============================================
echo   CardioGuard AI - Starting Server...
echo ============================================
echo.

cd /d "c:\Users\CHANDU\OneDrive\Desktop\FULLPROJECTWORK - Copy (3)"

echo [1/2] Starting FastAPI server on port 8000...
start "CardioGuard-Server" cmd /k "cd /d \"c:\Users\CHANDU\OneDrive\Desktop\FULLPROJECTWORK - Copy (3)\" && python -m uvicorn api.predict:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting 15 seconds for server to load ML models...
timeout /t 15 /nobreak

echo.
echo [2/2] Starting ngrok tunnel...
echo.
echo ============================================
echo   Your PUBLIC URL will appear below!
echo   Share it with anyone to access the app.
echo   DO NOT close this window!
echo ============================================
echo.
ngrok http 8000
