@echo off
rem Create the allin1 GPU env (/root/allin1env) inside an ALREADY INSTALLED WSL Ubuntu-22.04.
rem   WSL itself is installed separately, see README.md "allin1 environment".
rem   Usage: double-click, or  setup_allin1env.bat <distro-name>   (default: Ubuntu-22.04, or env VIBRATO_WSL_DISTRO)
rem   Dry run (paths only, no install):  set SETUP_DRY_RUN=1  then run this bat
chcp 65001 >nul
setlocal
set "WSL_UTF8=1"
set "DISTRO=%VIBRATO_WSL_DISTRO%"
if "%DISTRO%"=="" set "DISTRO=Ubuntu-22.04"
if not "%~1"=="" set "DISTRO=%~1"
set "HERE_WIN=%~dp0"
set "HERE_WIN=%HERE_WIN:~0,-1%"

echo == WSL distro: %DISTRO%
wsl -d %DISTRO% -u root -- true
if errorlevel 1 (
  echo [ERROR] WSL distro "%DISTRO%" not found. Install WSL first, see README.md
  echo         Installed distros:
  wsl -l -v
  pause
  exit /b 1
)

for /f "usebackq delims=" %%p in (`wsl -d %DISTRO% -u root -- wslpath -a "%HERE_WIN%"`) do set "HERE=%%p"
echo == setup dir (WSL path): %HERE%
echo == log: %HERE_WIN%\setup_allin1env.log
echo.
wsl -d %DISTRO% -u root -- bash -c "set -o pipefail; tr -d '\r' < '%HERE%/setup_allin1env.sh' > /tmp/setup_allin1env.sh && SETUP_DIR='%HERE%' SETUP_DRY_RUN='%SETUP_DRY_RUN%' bash /tmp/setup_allin1env.sh 2>&1 | tee '%HERE%/setup_allin1env.log'"
if errorlevel 1 (
  echo.
  echo [FAILED] see output above / setup_allin1env.log
  pause
  exit /b 1
)
echo.
echo == Final check, same test p1_allin1_separate.py uses
wsl -d %DISTRO% -u root -- bash -lc "test -x /root/allin1env/bin/allin1 && /root/allin1env/bin/python -c 'import allin1' && echo WSL_ALLIN1_OK"
if errorlevel 1 (
  echo.
  echo [FAILED] final check failed. See the error above.
  pause
  exit /b 1
)
echo.
echo Done. Now run: python run.py ^<song^>
pause
