# DinoAI DQN
Project by [Nicholas Renotte](https://www.youtube.com/watch?v=vahwuupy81A)

AI agent that learns to play Chrome's dinosaur game using Deep Q-Learning.

## Installation

### Prerequisites

- Python 3.x
- [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) 
  - Windows users: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

### Python Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Install Stable Baselines and protobuf
pip install stable-baselines3[extra] protobuf==3.20.*

# Install additional dependencies
pip install mss pydirectinput pytesseract
```

## Compatibility

?? **Note:** This project is currently Windows-only due to its dependency on `pydirectinput`.