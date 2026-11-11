@echo off
rem Build the virtual environment and start the application on Windows.
rem Usage: run.bat
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto run

echo Creating virtual environment .venv ...
py -3 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" goto no_python

echo Installing dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

:run
".venv\Scripts\python.exe" main.py
exit /b %errorlevel%

:no_python
echo ERROR: Python 3 was not found. Install it from https://www.python.org/downloads/
exit /b 1

:install_failed
echo ERROR: dependencies were not installed. Check the internet connection and run the script again.
rmdir /s /q .venv
exit /b 1
