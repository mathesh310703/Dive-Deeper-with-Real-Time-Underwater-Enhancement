# Dive-Deeper-with-Real-Time-Underwater-Enhancement
# Dive Deeper with Real-Time Underwater Enhancement

## Project Overview
**Dive Deeper with Real-Time Underwater Enhancement** is a deep learning system designed to enhance video visibility in underwater, foggy, and smoke-filled environments. The system leverages a **Multi-Patch Hierarchical Neural Network** to restore visibility, correct color distortion, and enhance fine details in real-time. The output is optimized for both **clarity and high FPS**, aiding object detection and monitoring in challenging environments.

## Features
- Real-time video enhancement for underwater and low-visibility scenes.
- Multi-scale processing for fine-grained and global scene enhancement.
- LAB color preservation for natural-looking output.
- Optional dehazing and contrast enhancement.
- High FPS output for real-time applications.
- Designed to help human operators or downstream object detection systems (does not detect objects itself).

## Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- GPU with CUDA support recommended for real-time performance

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mathesh310703/Dive-Deeper-with-Real-Time-Underwater-Enhancement.git
cd Dive-Deeper-with-Real-Time-Underwater-Enhancement

python -m venv venv
# Activate venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate

# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Dive-Deeper-with-Real-Time-Underwater-Enhancement/
│
├── main.py                # Main application script
├── encoder.py             # Encoder model
├── decoder.py             # Decoder model
├── checkpoints/           # Pretrained weights (encoder.pth, decoder.pth)
├── data/csvcreate.py, dataset.csv # Optional sample video/images
├── README.md              # Project documentation
└── utils/loader.py , loss.py # Optional dataset reference

## Run command
python main.py --encoder ./checkpoints/current_best_model_encoder.pth --decoder ./checkpoints/current_best_model_decoder.pth --mode color
python main.py --encoder ./checkpoints/current_best_model_3_encoder.pth --decoder ./checkpoints/current_best_model_3_decoder.pth --mode no_color
