@ECHO OFF
rem Create virtual environment
python -m venv .venv
rem Active the created virtual environment
call .venv\Scripts\activate
python.exe -m pip install --upgrade pip

REM pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
pip install flask
pip install flask-cors
pip install cv2
pip install numpy


