@echo off
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0start-webapp.ps1"
if errorlevel 1 pause
