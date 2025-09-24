# Model Files Directory

This directory contains large pre-trained model files that are not included in the Git repository due to size limitations.

## Required Models

### CLIP ViT-B/32 Model
- **File**: `ViT-B-32.pt`
- **Size**: ~337MB
- **Purpose**: Vision-language model for image-text similarity evaluation

## How to Obtain Models

### Option 1: Automatic Download (Recommended)
The system will automatically download required models when first run:

```bash
python launch_simple_amd.py
```

The models will be downloaded to:
- Windows: `C:\Users\{username}\.cache\clip\`
- Linux/Mac: `~/.cache/clip/`

### Option 2: Manual Download
If automatic download fails, manually download from:

1. **CLIP Models**: https://github.com/openai/CLIP
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

### Option 3: Alternative Model Location
Place your model files in this directory:
```
models/
├── ViT-B-32.pt          # CLIP Vision Transformer
└── other_models/        # Additional models
```

## Model File Structure

```
models/
├── README.md            # This file
├── ViT-B-32.pt         # CLIP model (auto-downloaded)
├── .gitkeep            # Keep directory in Git
└── downloads/          # Temporary download location
```

## Troubleshooting

### Model Loading Issues
1. Check internet connection for automatic downloads
2. Ensure sufficient disk space (~1GB free)
3. Verify Python environment has required packages:
   ```bash
   pip install torch torchvision clip-by-openai
   ```

### Manual Model Placement
If you have pre-downloaded models:
1. Place `ViT-B-32.pt` in this directory
2. Ensure file permissions allow read access
3. Restart the application

## Note for Developers

These model files are excluded from Git via `.gitignore` to prevent repository bloat. When setting up the project:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application (models will auto-download)

For deployment environments, consider:
- Pre-downloading models to a shared location
- Using model caching strategies
- Implementing model version management