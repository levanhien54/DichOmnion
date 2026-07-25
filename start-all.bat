@echo off
echo ========================================================
echo        KHOI DONG HE THONG OMNIVOICE (DICHOMNION)
echo ========================================================
echo.

echo [1/3] Khoi dong Gateway (Cloudflare Worker) tren cong 8787...
start "OmniVoice Gateway" cmd /k "cd apps\gateway && npm run dev"

echo [2/3] Khoi dong Client (Vite React) tren cong 5173...
start "OmniVoice Client" cmd /k "cd apps\client && npm run dev"

echo [3/3] Khoi dong GPU Worker (FastAPI) tren cong 8000...
start "OmniVoice GPU Worker" cmd /k "cd apps\gpu-worker && uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo ========================================================
echo Tat ca cac dich vu dang duoc khoi dong trong cac cua so rieng biet.
echo De tat he thong, hay dong cac cua so do hoac nhan Ctrl+C trong tung cua so.
echo ========================================================
echo.
pause
