@echo off
cd /d "C:\Users\Arpy\OneDrive\Documents\GitHub\Stock_Price_Prediction"
call .venv\Scripts\activate.bat

echo [%date% %time%] Starting daily data fetch... >> update_log.txt
python update_data.py >> update_log.txt 2>&1

echo [%date% %time%] Starting model retraining... >> update_log.txt
python retrain.py >> update_log.txt 2>&1

echo [%date% %time%] Daily automated pipeline finished. >> update_log.txt
