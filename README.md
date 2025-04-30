to install:

- pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
- pip install stable-baselines3[extra] protobuf==3.20.*
- pip install mss pydirectinput pytesseract

Also install tesseract ocr https://tesseract-ocr.github.io/tessdoc/Installation.html (installer if youre on windows: https://github.com/UB-Mannheim/tesseract/wiki)

it wont work on linux because of pydirectinput