@echo off
echo Starting ADHD AI Server...
cd "C:\Users\lobow\Downloads\EEGENIUS\adhd-ai"
uvicorn main:app --reload
pause