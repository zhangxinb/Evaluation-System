#!/usr/bin/env python3
"""
Quick test for the AMD evaluation function
"""

import numpy as np
from PIL import Image

def test_evaluation():
    """Test the evaluation function with dummy images"""
    
    print("🧪 Testing AMD evaluation function...")
    
    # Create test images
    img1 = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    
    # Import the evaluation function from the launcher
    import sys
    sys.path.append('.')
    
    try:
        # Import required modules
        import cv2
        from amd_identity_evaluator import AMDIdentityEvaluator
        
        # Initialize evaluator
        identity_eval = AMDIdentityEvaluator()
        
        # Test evaluation
        print("Testing image conversion...")
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        print(f"Image 1 shape: {img1_np.shape}")
        print(f"Image 2 shape: {img2_np.shape}")
        
        print("Testing identity evaluation...")
        identity_results = identity_eval.calculate_identity_similarity(img1_np, img2_np)
        print(f"Identity results: {identity_results}")
        
        print("Testing traditional metrics...")
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
        
        ssim_score = ssim(gray1, gray2, data_range=255)
        psnr_score = psnr(gray1, gray2, data_range=255)
        
        print(f"SSIM: {ssim_score:.4f}")
        print(f"PSNR: {psnr_score:.4f}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_evaluation()