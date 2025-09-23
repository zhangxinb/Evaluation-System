#!/usr/bin/env python3
"""
Test improved identity evaluator
"""

import numpy as np
import cv2
from amd_identity_evaluator import AMDIdentityEvaluator

def test_improved_evaluator():
    """Test with different types of images"""
    
    print("🧪 Testing Improved AMD Identity Evaluator...")
    
    evaluator = AMDIdentityEvaluator()
    
    # Test 1: Very similar images (should have high similarity)
    print("\n📸 Test 1: Similar faces")
    img1 = np.random.randint(100, 150, (200, 200, 3), dtype=np.uint8)  # Similar intensity
    img2 = img1 + np.random.randint(-10, 10, (200, 200, 3), dtype=np.uint8)  # Very similar
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    result1 = evaluator.calculate_identity_similarity(img1, img2)
    print(f"Similar faces similarity: {result1['similarity']:.4f}")
    
    # Test 2: Very different images (should have low similarity)
    print("\n📸 Test 2: Different faces")
    img3 = np.random.randint(50, 100, (200, 200, 3), dtype=np.uint8)   # Dark image
    img4 = np.random.randint(150, 255, (200, 200, 3), dtype=np.uint8)  # Bright image
    
    result2 = evaluator.calculate_identity_similarity(img3, img4)
    print(f"Different faces similarity: {result2['similarity']:.4f}")
    
    # Test 3: Pattern-based test
    print("\n📸 Test 3: Pattern vs Random")
    # Create a pattern image
    img5 = np.zeros((200, 200, 3), dtype=np.uint8)
    img5[50:150, 50:150] = 255  # White square in center
    
    # Create random image
    img6 = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    result3 = evaluator.calculate_identity_similarity(img5, img6)
    print(f"Pattern vs Random similarity: {result3['similarity']:.4f}")
    
    print(f"\n✅ Tests completed!")
    print(f"Expected: Similar > Pattern/Random > Very Different")
    print(f"Actual: {result1['similarity']:.3f} > {result3['similarity']:.3f} > {result2['similarity']:.3f}")

if __name__ == "__main__":
    test_improved_evaluator()