# Professional Image Evaluation System 🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://tensorflow.org/)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.95-green.svg)](https://github.com/serengil/deepface)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B/32-red.svg)](https://github.com/openai/CLIP)
[![LPIPS](https://img.shields.io/badge/LPIPS-AlexNet-orange.svg)](https://github.com/richzhang/PerceptualSimilarity)
[![Gradio](https://img.shields.io/badge/Gradio-5.46.1-purple)](https://gradio.app/)

A state-of-the-art AI-powered image evaluation platform that combines advanced deep learning algorithms (CLIP, LPIPS, multi-model face recognition) with traditional computer vision techniques for professional character consistency analysis and identity verification.

## 🎯 Overview

This professional-grade evaluation system is specifically designed for analyzing character consistency in AI-generated images, particularly for applications involving face recognition, identity verification, and semantic similarity assessment. The system integrates cutting-edge algorithms with robust fallback mechanisms to ensure reliable results across different hardware configurations.

## ✨ Key Features

### 🧠 Advanced Algorithm Integration
- **CLIP (ViT-B/32)**: OpenAI's Vision Transformer for semantic similarity analysis
- **LPIPS (AlexNet)**: Learned Perceptual Image Patch Similarity for human-aligned perception metrics
- **Multi-Model Face Recognition**: DeepFace integration with VGG-Face, FaceNet, OpenFace, and DeepFace models
- **Enhanced Fallback System**: Traditional CV methods (SSIM, LBP, Template Matching, ORB) for robustness

### 🏗️ System Architecture
- **CPU Optimized**: Designed for integrated graphics (AMD 780M) and CPU-only processing
- **Modular Design**: Three core components with clear separation of concerns
- **Compatibility Layer**: Graceful handling of dependency conflicts and hardware limitations
- **Professional Reporting**: Six-section English analysis reports with detailed interpretations

### 🎨 User Experience
- **Web Interface**: Modern Gradio-based dashboard running on port 7862
- **Real-time Processing**: Optimized for 5-15 second evaluation cycles
- **Comprehensive Reports**: Professional-grade analysis with scoring and recommendations
- **Multiple Format Support**: JPG, PNG, BMP, TIFF image formats

## 🔬 Algorithm Details

### 1. CLIP Semantic Analysis
**Technical Specifications:**
- **Model**: OpenAI ViT-B/32 (Vision Transformer)
- **Processing Mode**: CPU-optimized with 30-second timeout protection
- **Evaluation Range**: 0.0000 - 1.0000 (cosine similarity)
- **Recommended Threshold**: ≥ 0.7000 for high semantic consistency
- **Weight in Final Score**: 30%

**What it measures:** Semantic alignment between image content and text prompts, character concept consistency, and high-level visual understanding.

### 2. LPIPS Perceptual Similarity
**Technical Specifications:**
- **Model**: AlexNet-based learned perceptual metric
- **Processing Mode**: CPU processing with 20-second timeout
- **Evaluation Range**: 0.0000 - 1.0000 (perceptual distance)
- **Recommended Threshold**: ≤ 0.3000 for perceptually similar images
- **Weight in Final Score**: 25%

**What it measures:** Human-aligned perceptual similarity, focusing on features that matter most to human visual perception rather than pixel-level differences.

### 3. Multi-Model Face Recognition
**Technical Specifications:**
- **Primary Models**: VGG-Face, FaceNet, OpenFace, DeepFace (via DeepFace library)
- **Consensus Method**: Weighted average with confidence scoring
- **Fallback Methods**: 
  - Multi-Scale SSIM (Structural Similarity Index)
  - LBP (Local Binary Patterns) histogram comparison
  - Template matching with normalized cross-correlation
  - ORB (Oriented FAST and Rotated BRIEF) feature matching
- **Evaluation Range**: 0.0000 - 1.0000 (identity similarity)
- **Recommended Threshold**: ≥ 0.6000 for same identity
- **Weight in Final Score**: 25%

**What it measures:** Identity consistency across different poses, lighting conditions, and expressions while providing robust fallback for challenging cases.

### 4. Traditional Computer Vision Metrics
**Included Metrics:**
- **SSIM**: Structural Similarity Index (0.0 - 1.0, threshold ≥ 0.8)
- **PSNR**: Peak Signal-to-Noise Ratio (dB, threshold ≥ 20)
- **MSE**: Mean Squared Error (lower is better)
- **Color Histogram Correlation**: (0.0 - 1.0, threshold ≥ 0.7)
- **Combined Weight in Final Score**: 20%

## 📁 Project Structure

```
Evaluation System/
├── 📄 launch_simple_amd.py              # Main application launcher (469 lines)
│   ├── 🎯 format_advanced_results()     # Professional 6-section English reporting
│   ├── 🔄 format_fallback_results()     # Fallback mode formatting
│   └── 🚀 create_gradio_interface()     # Web UI initialization
│
├── 📄 compatible_evaluation_system.py   # Core evaluation engine (536 lines)
│   ├── 🧠 _safe_clip_analysis()         # CLIP with timeout protection
│   ├── 👁️ _safe_lpips_analysis()        # LPIPS with error handling
│   ├── 🎯 evaluate_character_consistency() # Main evaluation orchestrator
│   └── 📊 _calculate_traditional_metrics() # Classical CV metrics
│
├── 📄 professional_identity_evaluator.py # Face recognition system (386 lines)
│   ├── 👤 calculate_identity_similarity() # Multi-model face analysis
│   ├── 🔄 _fallback_evaluation()        # Traditional CV fallback methods
│   └── 📈 _enhanced_fallback_analysis() # 4-method ensemble approach
│
├── 📄 requirements.txt                  # Comprehensive dependency list
├── 📖 README.md                        # This documentation
├── 🔧 .gitignore                       # Git ignore patterns
├── 📂 .venv/                           # Python virtual environment
└── 📂 .git/                            # Git version control
```

## 🚀 Installation & Setup

### Prerequisites
- **Python**: 3.8 or higher (tested with Python 3.13)
- **Operating System**: Windows 11 (tested), macOS, or Linux
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5GB available space for models and dependencies
- **Network**: Internet connection required for initial model downloads

### Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/zhangxinb/Evaluation-System.git
cd Evaluation-System

# 2. Create virtual environment (recommended)
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the application
python launch_simple_amd.py
```

### Manual Installation Steps

If you prefer manual installation or encounter issues:

```bash
# Install core dependencies
pip install tensorflow==2.20.0
pip install torch==2.7.1+cpu torchvision==0.22.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install deepface==0.0.95
pip install gradio==5.46.1
pip install opencv-python==4.12.0.88
pip install lpips==0.1.4
pip install clip-by-openai==1.0

# Install supporting packages
pip install numpy==2.1.2 pillow==11.0.0 scikit-image==0.25.2
pip install matplotlib==3.10.6 tqdm==4.67.1 requests==2.32.5
```

### Verification

After installation, verify the system works correctly:

```bash
# Check if all models load successfully
python -c "
from compatible_evaluation_system import CompatibleEvaluationSystem
system = CompatibleEvaluationSystem()
print('Available methods:', list(system.available_methods.keys()))
"
```

Expected output should show: `['professional', 'clip', 'lpips']`

## 🎯 Usage Guide

### Starting the Application

```bash
# Standard launch (recommended)
python launch_simple_amd.py

# The application will start on: http://127.0.0.1:7862
```

### Web Interface Operations

1. **Upload Images**: 
   - **Generated Image**: Primary image to evaluate (required)
   - **Reference Image**: Comparison baseline (optional, for similarity analysis)

2. **Input Parameters**:
   - **Text Prompt**: Description for CLIP semantic analysis (optional)
   - **Evaluation Type**: Character consistency analysis (default)

3. **Results Interpretation**:
   - **Final Score**: 0-100 composite score with color-coded assessment
   - **Detailed Analysis**: Six-section professional report
   - **Algorithm Breakdown**: Individual metric scores and interpretations

### API Usage Example

```python
from compatible_evaluation_system import CompatibleEvaluationSystem
import cv2

# Initialize the system
evaluator = CompatibleEvaluationSystem()

# Load images
img1 = cv2.imread('generated_character.jpg')
img2 = cv2.imread('reference_character.jpg')

# Perform evaluation
results = evaluator.evaluate_character_consistency(
    img1, img2, 
    prompt="a professional headshot of a person",
    use_advanced=True
)

# Access results
print(f"Final Score: {results['Final_Score']:.1f}/100")
print(f"CLIP Similarity: {results.get('CLIP_Similarity', 'N/A')}")
print(f"Identity Similarity: {results.get('Identity_Similarity', 'N/A')}")
print(f"LPIPS Distance: {results.get('LPIPS_Distance', 'N/A')}")
```

## ⚖️ Scoring System & Weights

### Final Score Calculation

The system uses a weighted average approach to compute the final score:

```python
# Scoring weights (customizable)
CLIP_WEIGHT = 0.30        # 30% - Semantic consistency
IDENTITY_WEIGHT = 0.25    # 25% - Face recognition accuracy  
LPIPS_WEIGHT = 0.25       # 25% - Perceptual similarity
TRADITIONAL_WEIGHT = 0.20 # 20% - Classical metrics (SSIM, PSNR, etc.)

# Final score formula
final_score = (
    clip_similarity * CLIP_WEIGHT +
    identity_similarity * IDENTITY_WEIGHT +
    (1.0 - lpips_distance) * LPIPS_WEIGHT +
    traditional_average * TRADITIONAL_WEIGHT
) * 100
```

### Score Interpretation

| Score Range | Assessment | Color Code | Interpretation |
|-------------|------------|------------|----------------|
| 90-100 | Excellent | 🟢 | Outstanding consistency, production-ready |
| 80-89 | Very Good | 🟢 | High quality with minor variations |
| 70-79 | Good | 🟡 | Acceptable consistency, some improvements possible |
| 60-69 | Fair | 🟡 | Moderate consistency, needs refinement |
| 50-59 | Poor | 🔴 | Low consistency, significant issues |
| 0-49 | Very Poor | 🔴 | Major inconsistencies, requires substantial work |

### Individual Metric Thresholds

| Metric | Excellent (≥) | Good (≥) | Fair (≥) | Poor (<) |
|--------|---------------|----------|----------|----------|
| CLIP Similarity | 0.850 | 0.700 | 0.500 | 0.500 |
| Identity Similarity | 0.800 | 0.600 | 0.400 | 0.400 |
| LPIPS Distance | ≤ 0.200 | ≤ 0.300 | ≤ 0.500 | > 0.500 |
| SSIM | 0.900 | 0.800 | 0.600 | < 0.600 |
| PSNR (dB) | 25+ | 20+ | 15+ | < 15 |

## 🔧 Configuration & Customization

### Hardware Optimization

The system automatically detects and optimizes for your hardware:

```python
# CPU optimization (automatic)
- Uses CPU-only processing for compatibility
- Implements timeout protection for model loading
- Optimized thread management for integrated graphics

# Memory management
- Automatic image resizing for memory efficiency
- Garbage collection after heavy operations
- Batch processing limits to prevent OOM errors
```

### Custom Weight Configuration

You can modify the scoring weights by editing `launch_simple_amd.py`:

```python
# Locate the scoring section and adjust weights
def calculate_final_score(results):
    # Customize these weights based on your priorities
    CLIP_WEIGHT = 0.40        # Increase for semantic focus
    IDENTITY_WEIGHT = 0.30    # Increase for face recognition priority
    LPIPS_WEIGHT = 0.20       # Perceptual similarity weight
    TRADITIONAL_WEIGHT = 0.10 # Traditional metrics weight
    
    # Weights must sum to 1.0
    assert CLIP_WEIGHT + IDENTITY_WEIGHT + LPIPS_WEIGHT + TRADITIONAL_WEIGHT == 1.0
```

## 📊 Performance Benchmarks

### Processing Times (AMD 780M System)

| Operation | Time Range | Factors |
|-----------|------------|---------|
| CLIP Analysis | 3-8 seconds | Image size, prompt complexity |
| LPIPS Calculation | 2-5 seconds | Image resolution, model warmup |
| Face Recognition | 5-12 seconds | Number of faces, detection complexity |
| Traditional Metrics | 1-2 seconds | Image size only |
| **Total Evaluation** | **8-15 seconds** | Combined processing |

### Memory Usage

| Component | RAM Usage | Notes |
|-----------|-----------|--------|
| Base System | ~2GB | Python environment, basic libraries |
| CLIP Model | ~1.5GB | ViT-B/32 model in memory |
| LPIPS Model | ~500MB | AlexNet-based network |
| DeepFace Models | ~1GB | Multiple face recognition models |
| **Peak Usage** | **~5GB** | During simultaneous model usage |

## 🛠️ Troubleshooting

### Common Issues and Solutions

**1. Model Loading Timeouts**
```bash
# Solution: Increase timeout values or check internet connection
# Models are downloaded automatically on first use
```

**2. Memory Errors**
```bash
# Solution: Reduce image resolution or restart the application
# Large images (>2048px) are automatically resized
```

**3. Face Detection Failures**
```bash
# Solution: Ensure images contain clear, frontal faces
# The system includes fallback methods for difficult cases
```

**4. CUDA/GPU Related Errors**
```bash
# Solution: The system is designed for CPU-only operation
# GPU acceleration is not required and may cause conflicts
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
# Add this to the top of launch_simple_amd.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python launch_simple_amd.py --debug
```

## 🔬 Technical Specifications

### Algorithm Implementation Details

**CLIP Integration:**
- Model: `openai/clip-vit-base-patch32`
- Image preprocessing: 224x224 resize, normalization
- Text encoding: Tokenization with 77-token limit
- Similarity: Cosine similarity between image and text embeddings

**LPIPS Implementation:**
- Network: AlexNet-based perceptual loss
- Preprocessing: Standard ImageNet normalization
- Output: Perceptual distance (0=identical, 1=completely different)
- Hardware: CPU-optimized implementation

**DeepFace Configuration:**
- Models: VGG-Face, FaceNet, OpenFace, DeepFace
- Backend: TensorFlow 2.20.0
- Detection: MTCNN for face detection
- Verification: Cosine similarity with model-specific thresholds

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Web Interface                     │
├─────────────────────────────────────────────────────────────┤
│                 launch_simple_amd.py                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐  │
│  │ Result Formatting│ │ UI Management   │ │ Error Handling│  │
│  └─────────────────┘ └─────────────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              compatible_evaluation_system.py                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐  │
│  │ CLIP Analysis   │ │ LPIPS Metrics   │ │ Traditional CV│  │
│  └─────────────────┘ └─────────────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│           professional_identity_evaluator.py               │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐  │
│  │ DeepFace Models │ │ Fallback Methods│ │ Consensus Alg.│  │
│  └─────────────────┘ └─────────────────┘ └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Research Applications

This system is particularly valuable for:

- **Academic Research**: Quantitative evaluation of generative models
- **AI Art Creation**: Quality assessment for character consistency
- **Commercial Applications**: Automated quality control for generated content
- **Dataset Validation**: Large-scale image dataset quality assessment
- **Model Development**: Benchmarking and comparison of different generation approaches

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-algorithm`)
3. **Implement** your changes with proper documentation
4. **Test** thoroughly on different hardware configurations
5. **Submit** a pull request with detailed description

### Development Setup

```bash
# Clone for development
git clone https://github.com/zhangxinb/Evaluation-System.git
cd Evaluation-System

# Install in development mode
pip install -e .

# Run tests (when available)
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP**: Revolutionary vision-language understanding
- **LPIPS**: Human-aligned perceptual similarity metrics
- **DeepFace**: Comprehensive face recognition framework
- **Gradio**: Elegant web interface framework
- **TensorFlow & PyTorch**: Deep learning framework foundations

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/zhangxinb/Evaluation-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zhangxinb/Evaluation-System/discussions)
- **Documentation**: This README and inline code documentation

---

⭐ **Star this repository** if you find it useful for your research or projects!

**Professional Image Evaluation System** - Advancing the science of AI-generated image quality assessment.