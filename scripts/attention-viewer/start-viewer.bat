@echo off
setlocal
cd /d "%~dp0"
if exist "D:\Anaconda\python.exe" (
  "D:\Anaconda\python.exe" "%~dp0serve-viewer.py" %*
  goto done
)
where py >nul 2>nul
if not errorlevel 1 (
  py -3 "%~dp0serve-viewer.py" %*
  goto done
)
where python >nul 2>nul
if not errorlevel 1 (
  python "%~dp0serve-viewer.py" %*
  goto done
)
echo Python 3 was not found. Install Python or edit this BAT with its path.
pause
exit /b 1
:done
if errorlevel 1 pause
endlocal
