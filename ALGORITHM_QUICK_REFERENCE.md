# Algorithm Quick Reference 🚀

**Professional Image Evaluation System - Quick Algorithm Summary**

---

## Core Algorithms Overview

| Algorithm | Type | Primary Function | Weight | Threshold |
|-----------|------|------------------|---------|-----------|
| **CLIP (ViT-B/32)** | Deep Learning | Semantic similarity | 30% | ≥ 0.7000 |
| **LPIPS (AlexNet)** | Deep Learning | Perceptual similarity | 25% | ≤ 0.3000 |
| **DeepFace Multi-Model** | Deep Learning | Identity recognition | 25% | ≥ 0.6000 |
| **Traditional CV** | Classical | Image quality metrics | 20% | Varies |

---

## Algorithm Specifications

### 1. CLIP Semantic Analysis
```
Model: OpenAI ViT-B/32 (Vision Transformer)
Parameters: ~150M
Input: 224×224 RGB images
Output: 512-dim embeddings
Similarity: Cosine similarity (0.0-1.0)
Processing: CPU-optimized, 30s timeout
```

### 2. LPIPS Perceptual Similarity  
```
Backbone: AlexNet (ImageNet pre-trained)
Input: 256×256 RGB images
Output: Perceptual distance (0.0-1.0+)
Layers: 5 convolutional layers
Processing: CPU mode, 20s timeout
```

### 3. Multi-Model Face Recognition
```
Models: VGG-Face, FaceNet, OpenFace, DeepFace
Detection: MTCNN face detector
Method: Weighted consensus voting
Fallback: 4 traditional CV methods
Processing: 15s timeout per model
```

### 4. Traditional Metrics
```
SSIM: Structural similarity (0.0-1.0)
PSNR: Peak signal-to-noise ratio (dB)
MSE: Mean squared error
Color: Histogram correlation
Processing: Always available
```

---

## Fallback Algorithm Chain

```
Level 1: Advanced Deep Learning (Primary)
├── CLIP semantic analysis
├── LPIPS perceptual metrics  
└── DeepFace identity verification

Level 2: Traditional CV (Complementary)
├── Multi-scale SSIM
├── PSNR quality assessment
├── Color histogram analysis
└── Basic pixel metrics

Level 3: Robust Fallbacks (Emergency)
├── Local Binary Patterns (LBP)
├── Template matching
├── ORB feature matching
└── Enhanced SSIM variants
```

---

## Performance Benchmarks (AMD 780M)

| Algorithm | Avg Time | Memory | Success Rate |
|-----------|----------|---------|--------------|
| CLIP | 3-8s | 1.5GB | 95%+ |
| LPIPS | 2-5s | 500MB | 98%+ |
| DeepFace | 5-12s | 1GB | 90%+ |
| Traditional | 1-2s | 200MB | 100% |
| **Total** | **8-15s** | **~3GB** | **95%+** |

---

## Scoring Formula

```python
Final Score = (
    CLIP_similarity × 0.30 +
    Identity_similarity × 0.25 +
    (1.0 - LPIPS_distance) × 0.25 +
    Traditional_average × 0.20
) × 100

# Quality Levels:
# 90-100: Excellent 🟢
# 80-89:  Very Good 🟢  
# 70-79:  Good 🟡
# 60-69:  Fair 🟡
# 50-59:  Poor 🔴
# 0-49:   Very Poor 🔴
```

---

## Key Implementation Features

### ✅ Robustness
- Timeout protection on all algorithms
- Graceful degradation with fallbacks
- Memory monitoring and optimization
- Cross-platform compatibility

### ✅ Performance
- CPU-optimized for integrated graphics
- Multi-threaded processing
- Automatic image resizing
- Model caching and reuse

### ✅ Quality
- Multi-model consensus voting
- Confidence scoring for all results
- Professional 6-section reporting
- Detailed error logging

---

## Usage Examples

### Basic Evaluation
```python
from compatible_evaluation_system import CompatibleEvaluationSystem

evaluator = CompatibleEvaluationSystem()
results = evaluator.evaluate_character_consistency(img1, img2)
print(f"Final Score: {results['Final_Score']}/100")
```

### Advanced Configuration
```python
# Custom weights
weights = {'clip': 0.4, 'identity': 0.3, 'lpips': 0.2, 'traditional': 0.1}

# Custom thresholds  
thresholds = {'clip': 0.75, 'identity': 0.65, 'lpips': 0.25}
```

### Error Handling
```python
try:
    results = evaluator.evaluate_character_consistency(img1, img2)
    if 'error' in results:
        print(f"Evaluation failed: {results['error']}")
    else:
        print(f"Success: {results['Final_Score']}")
except Exception as e:
    print(f"System error: {e}")
```

---

## Troubleshooting Quick Guide

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| CLIP timeout | Large images/slow CPU | Reduce image size, increase timeout |
| DeepFace fails | No face detected | Check image quality, use fallback |
| Memory error | Insufficient RAM | Reduce image resolution |
| All algorithms fail | System overload | Restart application |

---

**Quick Reference Version**: 1.0  
**Full Documentation**: See ALGORITHM_DOCUMENTATION.md (52KB, 1498 lines)  
**Last Updated**: September 27, 2025