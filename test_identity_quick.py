#!/usr/bin/env python3
"""
Quick verification that identity evaluator is working
"""

import numpy as np
from PIL import Image

def test_identity_evaluator():
    """Test the identity evaluator directly"""
    
    print("🧪 Testing AMD Identity Evaluator...")
    
    try:
        from amd_identity_evaluator import AMDIdentityEvaluator
        
        # Create evaluator
        evaluator = AMDIdentityEvaluator()
        
        # Create test images
        img1_np = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        img2_np = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Test evaluation
        results = evaluator.calculate_identity_similarity(img1_np, img2_np)
        
        print("✅ Identity evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_identity_evaluator()