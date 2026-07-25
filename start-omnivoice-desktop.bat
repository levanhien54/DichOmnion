@echo off
title OmniVoice Studio (Zero-Trust Edition) - Launcher
color 0b

echo ===================================================
echo      OmniVoice Studio - Khoi chay he thong...
echo ===================================================

echo [1/3] Khoi chay GPU Worker (FastAPI)...
cd apps\gpu-worker
start "OmniVoice GPU Worker" cmd /c "uv run uvicorn src.main:app --host 0.0.0.0 --port 8000"
cd ..\..

echo [2/3] Khoi chay API Gateway (Cloudflare Worker)...
cd apps\gateway
start "OmniVoice Gateway" cmd /c "npx wrangler dev src/index.ts"
cd ..\..

echo [3/3] Cho he thong san sang (5 giay)...
timeout /t 5 /nobreak >nul

echo [4/4] Khoi chay Giao dien Desktop (Tauri)...
cd apps\client\src-tauri\target\release
if exist "OmniVoice.exe" (
    start OmniVoice.exe
) else (
    echo [Loi] Khong tim thay OmniVoice.exe. Vui long build Tauri truoc!
    echo Chay lenh: pnpm tauri build tai thu muc apps\client
    pause
)
cd ..\..\..\..

echo Hoan tat! De dong he thong, hay dong cac cua so cmd.
exit
