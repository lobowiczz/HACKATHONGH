@echo off
echo Starting Parkinson's AI Server...
cd "C:\Users\lobow\Downloads\EEGENIUS\parkinsons-ai"
uvicorn main:app --reload
pause