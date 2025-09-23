import os
import torch

# Basic configuration
class Config:
    # Device configuration optimized for AMD GPU/CPU
    @staticmethod
    def get_device():
        """Smart device detection for AMD GPU systems"""
        # Check for CUDA (NVIDIA) - not available for AMD
        if torch.cuda.is_available():
            print("⚠️ CUDA detected but you have AMD GPU - using CPU for compatibility")
            return torch.device('cpu')
        
        # Check for ROCm (AMD GPU support) - experimental
        try:
            if hasattr(torch.backends, 'cuda') and torch.backends.cuda.is_available():
                # This is actually CUDA, use CPU for AMD systems
                return torch.device('cpu')
        except:
            pass
        
        # Default to CPU for AMD systems (most stable)
        print("ℹ️ Using CPU mode - optimized for AMD 780M system")
        return torch.device('cpu')
    
    DEVICE = get_device.__func__()
    
    # Model configuration optimized for CPU/AMD systems
    CLIP_MODEL_NAME = "ViT-B/32"  # Efficient model for CPU
    FACE_DETECTION_MODEL = "hog"  # CPU-optimized face detection
    
    # Performance configuration for AMD 780M systems
    BATCH_SIZE = 1  # Conservative batch size for integrated GPU
    MAX_CONCURRENT_EVALUATIONS = 2  # Limit concurrent processing
    
    # Memory optimization settings
    ENABLE_MEMORY_OPTIMIZATION = True
    CLEAR_CACHE_INTERVAL = 5  # Clear cache every 5 evaluations
    
    # 模型配置
    CLIP_MODEL_NAME = "ViT-B/32"
    FACE_DETECTION_MODEL = "hog"  # 可选: "hog", "cnn"
    
    # 评估指标权重
    METRIC_WEIGHTS = {
        'clip_similarity': 0.3,
        'identity_similarity': 0.25,
        'lpips_similarity': 0.25,
        'ssim_similarity': 0.1,
        'psnr_similarity': 0.1
    }
    
    # 阈值设置
    THRESHOLDS = {
        'clip_threshold': 0.7,
        'identity_threshold': 0.6,
        'lpips_threshold': 0.3,
        'ssim_threshold': 0.8,
        'psnr_threshold': 20.0
    }
    
    # 图像处理配置
    IMAGE_SIZE = (224, 224)
    MAX_IMAGE_SIZE = 1024
    
    # 可视化配置
    FIGURE_SIZE = (12, 8)
    DPI = 100