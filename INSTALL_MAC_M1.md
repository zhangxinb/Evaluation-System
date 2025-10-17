# Installation Guide for M1/M2 Mac (Apple Silicon)

## Problem Summary

The original `requirements.txt` has a dependency conflict:
- `clip-by-openai==1.0` requires `torch>=1.7.1,<1.7.2` (old version)
- Modern packages require `torch==2.7.1` (latest version)

## Solution

We provide three options:

### Option 1: Minimal Installation (Recommended for M1/M2)

Use the minimal requirements file that only includes essential packages:

```bash
# 1. Verify you're using ARM64 Python (not Rosetta)
python3 -c "import platform; print(platform.machine())"
# Should output: arm64

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install minimal dependencies
pip install -r requirements-mac-m1-minimal.txt
```

### Option 2: Full Installation with Mac-specific fixes

```bash
# Use the Mac-optimized requirements file
pip install -r requirements-mac-m1.txt
```

### Option 3: Manual step-by-step installation

If you encounter issues, install in this order:

```bash
# Step 1: Install PyTorch first (with MPS support)
pip install torch==2.7.1 torchvision==0.22.1

# Step 2: Install TensorFlow
pip install tensorflow==2.20.0 tf-keras==2.20.1

# Step 3: Install CLIP from GitHub (avoids version conflict)
pip install git+https://github.com/openai/CLIP.git
pip install ftfy==6.1.1

# Step 4: Install LPIPS
pip install lpips==0.1.4

# Step 5: Install DeepFace and face detection
pip install deepface==0.0.95 mtcnn==1.0.0 retina-face==0.0.17

# Step 6: Install OpenCV and image processing
pip install opencv-python==4.12.0.88 pillow==11.0.0 scikit-image==0.25.2

# Step 7: Install scientific computing
pip install numpy==2.1.2 scipy==1.16.2 pandas==2.3.2

# Step 8: Install visualization
pip install matplotlib==3.10.6 seaborn==0.13.2

# Step 9: Install Gradio web interface
pip install gradio==5.46.1

# Step 10: Install utilities
pip install tqdm==4.67.1 requests==2.32.5 PyYAML==6.0.2 rich==14.1.0

# Step 11: Install transformers
pip install transformers==4.56.2 huggingface-hub==0.35.1 tokenizers==0.22.1 safetensors==0.6.2

# Step 12: Install system tools
pip install psutil==7.1.0 pydantic==2.11.9
```

## Key Changes for M1/M2 Mac

1. **PyTorch**: Removed `+cpu` suffix and `--extra-index-url` (not needed on Mac)
2. **CLIP**: Changed from `clip-by-openai==1.0` to `git+https://github.com/openai/CLIP.git`
3. **Native GPU**: Can use Metal Performance Shaders (MPS) for acceleration

## Using GPU Acceleration on M1/M2

To use the M1/M2 GPU (via Metal), add this to your code:

```python
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal Performance Shaders (GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")
```

## Common Issues and Solutions

### Issue 1: "ERROR: Cannot install ... conflicting dependencies"

**Solution**: Use `requirements-mac-m1-minimal.txt` instead

### Issue 2: "arm64 architecture not supported"

**Solution**: Make sure you're using ARM64 Python, not x86_64 (Rosetta)

```bash
# Check Python architecture
python3 -c "import platform; print(platform.machine())"

# If it shows x86_64, reinstall Python for ARM64:
# Download from python.org or use:
brew install python@3.11
```

### Issue 3: OpenCV installation fails

**Solution**: Install via Homebrew first:

```bash
brew install opencv
pip install opencv-python==4.12.0.88
```

### Issue 4: TensorFlow not using GPU

**Solution**: TensorFlow on Mac uses Metal automatically, no configuration needed

### Issue 5: "No module named 'clip'"

**Solution**: CLIP from GitHub installs as `import clip`, same as PyPI version

## Verification

After installation, test that everything works:

```bash
python3 << EOF
import torch
import tensorflow as tf
import clip
import lpips
import deepface
import cv2
import gradio

print("✅ PyTorch:", torch.__version__)
print("✅ TensorFlow:", tf.__version__)
print("✅ CLIP: OK")
print("✅ LPIPS: OK")
print("✅ DeepFace: OK")
print("✅ OpenCV:", cv2.__version__)
print("✅ Gradio:", gradio.__version__)

if torch.backends.mps.is_available():
    print("✅ MPS (GPU) available")
EOF
```

## Running the Application

```bash
# Start the evaluation system
python app.py
```

The web interface will be available at: `http://127.0.0.1:7862`

## Performance Notes

- M1/M2 Macs have excellent CPU performance for this workload
- MPS (GPU) acceleration available for PyTorch operations
- TensorFlow automatically uses Metal for GPU acceleration
- Unified memory architecture provides efficient data sharing
- Expected performance: Similar to or better than Windows CPU mode

## Need Help?

If you encounter issues:

1. Check Python architecture: `python3 -c "import platform; print(platform.machine())"`
2. Update pip: `pip install --upgrade pip`
3. Try minimal installation: `pip install -r requirements-mac-m1-minimal.txt`
4. Install packages one by one (see Option 3 above)
5. Check GitHub issues: https://github.com/zhangxinb/Evaluation-System/issues
