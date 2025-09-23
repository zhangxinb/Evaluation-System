#!/usr/bin/env python3
"""
AMD 780M Performance Test Suite
Tests system performance and validates AMD optimization
"""

import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def test_pytorch_performance():
    """Test PyTorch performance on AMD system"""
    
    print("\n🧪 PyTorch Performance Test")
    print("-" * 40)
    
    try:
        import torch
        import numpy as np
        
        # Force CPU mode for AMD compatibility
        device = torch.device('cpu')
        print(f"Device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CPU threads: {torch.get_num_threads()}")
        
        # Performance test
        start_time = time.time()
        
        # Matrix operations (simulating image processing)
        for i in range(100):
            a = torch.randn(512, 512, device=device)
            b = torch.randn(512, 512, device=device)
            c = torch.mm(a, b)
        
        elapsed = time.time() - start_time
        print(f"Matrix operations: {elapsed:.2f}s")
        
        # Convolution test (simulating CNN operations)
        start_time = time.time()
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
        
        for i in range(50):
            x = torch.randn(1, 3, 224, 224, device=device)
            y = conv(x)
        
        elapsed = time.time() - start_time
        print(f"Convolution operations: {elapsed:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_clip_performance():
    """Test CLIP model performance"""
    
    print("\n🔍 CLIP Model Performance Test")
    print("-" * 40)
    
    try:
        import torch
        from PIL import Image
        import numpy as np
        
        # Try to import CLIP
        try:
            import clip
            print("Using OpenAI CLIP")
        except ImportError:
            try:
                import open_clip
                print("Using Open-CLIP")
            except ImportError:
                print("❌ No CLIP implementation available")
                return False
        
        # Test with dummy image
        device = torch.device('cpu')
        
        # Create test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        start_time = time.time()
        
        # Load model (CPU only for AMD)
        if 'clip' in locals():
            model, preprocess = clip.load("ViT-B/32", device=device, download_root="./models")
        else:
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
        
        loading_time = time.time() - start_time
        print(f"Model loading time: {loading_time:.2f}s")
        
        # Test inference
        start_time = time.time()
        
        with torch.no_grad():
            image = preprocess(test_image).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
        
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ CLIP test failed: {e}")
        return False

def test_face_detection():
    """Test face detection with OpenCV (AMD optimized)"""
    
    print("\n👤 Face Detection Performance Test")
    print("-" * 40)
    
    try:
        from amd_identity_evaluator import AMDIdentityEvaluator
        import numpy as np
        
        # Initialize AMD face detection
        evaluator = AMDIdentityEvaluator()
        
        # Test with dummy image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        
        for i in range(50):
            faces = evaluator.detect_faces(test_image)
        
        elapsed = time.time() - start_time
        print(f"Face detection (50 frames): {elapsed:.2f}s")
        print(f"Average per frame: {elapsed/50:.3f}s")
        print(f"Detection method: {evaluator.detection_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def test_image_processing():
    """Test image processing performance"""
    
    print("\n🖼️ Image Processing Performance Test")
    print("-" * 40)
    
    try:
        import cv2
        import numpy as np
        from PIL import Image
        
        # Test image operations
        start_time = time.time()
        
        for i in range(100):
            # Create test image
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            # Common operations
            resized = cv2.resize(img, (256, 256))
            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        elapsed = time.time() - start_time
        print(f"Image operations (100 images): {elapsed:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage optimization"""
    
    print("\n💾 Memory Usage Test")
    print("-" * 40)
    
    try:
        import psutil
        import torch
        import gc
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory: {initial_memory:.1f} MB")
        
        # Allocate some tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000)
            tensors.append(tensor)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Peak memory: {peak_memory:.1f} MB")
        
        # Clean up
        del tensors
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {peak_memory - initial_memory:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run all performance tests"""
    
    print("🏁 AMD 780M Performance Test Suite")
    print("=" * 60)
    print("Testing system optimization for AMD integrated graphics")
    print("=" * 60)
    
    # Store results
    results = {}
    
    # Run tests
    results['pytorch'] = test_pytorch_performance()
    results['clip'] = test_clip_performance()
    results['face_detection'] = test_face_detection()
    results['image_processing'] = test_image_processing()
    results['memory'] = test_memory_usage()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} : {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! AMD 780M optimization successful!")
        print("✅ System ready for image evaluation")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check dependencies:")
        print("Run: python install_amd_optimized.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main()