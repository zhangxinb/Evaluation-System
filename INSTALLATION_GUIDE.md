# Professional Image Evaluation System - Installation Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/zhangxinb/Evaluation-System.git
cd Evaluation-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the System
```bash
python launch_simple_amd.py
```

The system will automatically download required models (first run may take a few minutes).

## Detailed Setup Instructions

### Prerequisites
- Python 3.8+ 
- 4GB+ RAM (8GB+ recommended)
- Internet connection (for model downloads)
- Windows 10/11, macOS, or Linux

### Installation Steps

#### Step 1: Python Environment Setup
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import torch, tensorflow, gradio; print('All dependencies installed successfully')"
```

#### Step 3: First Run Setup
```bash
# Launch the system
python launch_simple_amd.py
```

**Note**: The first run will download models (~337MB total):
- CLIP ViT-B/32 model for vision-language tasks
- DeepFace models for face recognition
- Additional components as needed

### Model Downloads
Models are automatically downloaded to:
- **Windows**: `C:\Users\{username}\.cache\`
- **macOS**: `~/Library/Caches/`
- **Linux**: `~/.cache/`

## System Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 4GB (system will use ~2-3GB during processing)
- **Storage**: 2GB free space (for models and cache)
- **Network**: Internet connection for initial setup

### Recommended Configuration
- **CPU**: 8+ cores (Intel i5/i7, AMD Ryzen 5/7)
- **RAM**: 8GB+ 
- **Storage**: SSD with 5GB+ free space
- **Graphics**: Integrated graphics supported (no dedicated GPU required)

## Features Overview

### Core Capabilities
- ✅ **Professional Face Recognition**: Multi-model DeepFace integration
- ✅ **Image Quality Assessment**: SSIM, PSNR, MSE metrics
- ✅ **Demographic Analysis**: Age, gender, emotion detection
- ✅ **CPU Optimized**: Works on integrated graphics
- ✅ **Web Interface**: User-friendly Gradio dashboard
- ✅ **Real-time Processing**: 3-5 seconds per image pair

### Supported Formats
- **Images**: JPG, PNG, BMP, TIFF
- **Input**: Drag-and-drop or file upload
- **Output**: Formatted analysis reports

## Usage Guide

### Web Interface
1. Open browser to: `http://127.0.0.1:7861`
2. Upload two images for comparison
3. Click "Analyze Images"
4. Review comprehensive results

### API Usage
```python
from professional_identity_evaluator import ProfessionalIdentityEvaluator

# Initialize evaluator
evaluator = ProfessionalIdentityEvaluator()

# Analyze images
results = evaluator.evaluate_identity("image1.jpg", "image2.jpg")

# Access results
print(f"Identity Similarity: {results['Identity_Similarity']:.4f}")
print(f"Decision: {results['Identity_Decision']}")
```

## Troubleshooting

### Common Issues

#### Model Download Fails
```bash
# Check internet connection
ping github.com

# Clear cache and retry
rm -rf ~/.cache/clip/
python launch_simple_amd.py
```

#### Memory Errors
```bash
# Reduce image size before upload
# Ensure 4GB+ RAM available
# Close other applications
```

#### Import Errors
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

#### Permission Errors (Windows)
- Run Command Prompt as Administrator
- Ensure antivirus allows Python/pip operations

### Performance Optimization

#### For Slower Systems
1. Reduce image resolution before upload
2. Close unnecessary applications
3. Use single-model evaluation mode

#### For Development
```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
python test_evaluation.py
```

## Project Structure

```
Evaluation-System/
├── launch_simple_amd.py           # Main launcher
├── professional_identity_evaluator.py  # Core evaluation engine
├── requirements.txt               # Dependencies
├── models/                        # Model files (auto-downloaded)
│   ├── README.md                 # Model documentation
│   └── .gitkeep                  # Directory preservation
├── docs/                         # Documentation
│   ├── ALGORITHM_DOCUMENTATION.md
│   ├── TECHNICAL_SPECIFICATIONS.md
│   └── API_REFERENCE.md
└── .gitignore                    # Git ignore rules
```

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Submit pull request

### Code Style
- Follow PEP 8 for Python code
- Add docstrings for new functions
- Update documentation for new features

## Support

### Documentation
- [Algorithm Documentation](ALGORITHM_DOCUMENTATION.md)
- [Technical Specifications](TECHNICAL_SPECIFICATIONS.md)
- [API Reference](API_REFERENCE.md)

### Issues
- Report bugs: Create GitHub issue
- Feature requests: Create GitHub issue with [Feature] tag
- Questions: Use GitHub Discussions

## License

This project is provided for educational and research purposes. Please respect the licenses of included models and dependencies.

---

**Last Updated**: September 24, 2025  
**Version**: 1.0.0  
**Compatibility**: Windows 10/11, macOS 10.15+, Linux (Ubuntu 18.04+)