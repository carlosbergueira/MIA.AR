@echo off
setlocal

cd /d "%~dp0"

python run_car_rental.py %*

endlocal