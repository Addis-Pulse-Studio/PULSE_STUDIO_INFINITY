@echo off
rem Upscale a finished Pulse film to 2K/4K with DaVinci Resolve Studio's Super Scale.
rem
rem   Drag a film, or a Pulse run folder, onto this file -- or run:
rem   resolve_superscale.bat <film or run folder> [--target 1440p] [--mode 2x_enhanced]
rem
rem When Resolve is not running this starts it headless (-nogui) and quits it
rem afterwards; a Resolve you already have open is used as it is, and left open.
rem What it does, and the one-time setup: resolve_superscale.py, beside this file.
setlocal
set "RESOLVE_DIR=%ProgramFiles%\Blackmagic Design\DaVinci Resolve"
set "RESOLVE_PY=%RESOLVE_DIR%\ResolvePython\ResolvePython.exe"
if not exist "%RESOLVE_PY%" echo Resolve's own Python is not at "%RESOLVE_PY%". & pause & exit /b 1
set "QUIT="
tasklist /FI "IMAGENAME eq Resolve.exe" 2>nul | find /I "Resolve.exe" >nul
if errorlevel 1 set "QUIT=--quit"
if defined QUIT echo Starting DaVinci Resolve headless, it can take a minute...
if defined QUIT start "" "%RESOLVE_DIR%\Resolve.exe" -nogui
"%RESOLVE_PY%" "%~dp0resolve_superscale.py" %* %QUIT%
pause
endlocal
