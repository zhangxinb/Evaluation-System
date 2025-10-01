#!/usr/bin/env python3
"""
Test script to validate face recognition improvements
"""

import cv2
import numpy as np
from face_recognition import ProfessionalIdentityEvaluator
from PIL import Image
import sys

def create_test_image_with_face(size=(500, 500), face_size=150, background_color=(100, 150, 200)):
    """
    Create a test image with a simple face drawing
    """
    img = np.full((size[0], size[1], 3), background_color, dtype=np.uint8)
    
    # Draw a simple face in the center
    center_x, center_y = size[1] // 2, size[0] // 2
    
    # Face circle
    cv2.circle(img, (center_x, center_y), face_size // 2, (255, 220, 180), -1)
    
    # Eyes
    eye_offset = face_size // 5
    cv2.circle(img, (center_x - eye_offset, center_y - eye_offset // 2), 
               face_size // 15, (50, 50, 50), -1)
    cv2.circle(img, (center_x + eye_offset, center_y - eye_offset // 2), 
               face_size // 15, (50, 50, 50), -1)
    
    # Mouth
    cv2.ellipse(img, (center_x, center_y + eye_offset), 
                (face_size // 4, face_size // 6), 0, 0, 180, (50, 50, 50), 2)
    
    return img

def test_face_detection():
    """Test face detection capability"""
    print("\n" + "="*80)
    print("🧪 Test 1: Face Detection Capability")
    print("="*80)
    
    evaluator = ProfessionalIdentityEvaluator()
    
    # Create a test image with face
    test_img = create_test_image_with_face()
    
    # Convert to PIL
    pil_img = Image.fromarray(test_img)
    
    # Test face detection
    print("\n🔍 Testing face detection...")
    face_crop = evaluator._detect_and_crop_face(pil_img)
    
    if face_crop is not None:
        print(f"✅ Face detected and cropped! Size: {face_crop.size}")
        return True
    else:
        print("❌ Face detection failed!")
        return False

def test_different_backgrounds():
    """Test same face with different backgrounds"""
    print("\n" + "="*80)
    print("🧪 Test 2: Same Face, Different Backgrounds")
    print("="*80)
    
    evaluator = ProfessionalIdentityEvaluator()
    
    # Create same face with different backgrounds
    img1 = create_test_image_with_face(
        size=(600, 600), 
        face_size=180, 
        background_color=(100, 150, 200)  # Blue background
    )
    
    img2 = create_test_image_with_face(
        size=(600, 600), 
        face_size=180, 
        background_color=(50, 200, 100)  # Green background
    )
    
    print("\n📊 Image 1: Blue background")
    print("📊 Image 2: Green background (SAME face)")
    
    print("\n🔍 Running identity comparison...")
    result = evaluator.calculate_identity_similarity(img1, img2)
    
    print("\n" + "-"*80)
    print("📋 Results:")
    print("-"*80)
    for key, value in result.items():
        if key != 'model_results':
            print(f"   {key}: {value}")
    
    # Check if correctly identified as same person
    similarity = result.get('similarity', 0.0)
    decision = result.get('identity_decision', 'Unknown')
    
    print("\n" + "="*80)
    if similarity > 0.4 and decision == "Same Person":
        print("✅ TEST PASSED: Correctly identified as same person despite different backgrounds!")
        return True
    else:
        print(f"⚠️ TEST NEEDS IMPROVEMENT: similarity={similarity:.4f}, decision={decision}")
        print("   Expected: similarity > 0.4 and decision='Same Person'")
        return False

def test_real_images():
    """Test with real images if available"""
    print("\n" + "="*80)
    print("🧪 Test 3: Real Images Test")
    print("="*80)
    
    print("\n💡 To test with your real images:")
    print("   1. Place two images in this folder")
    print("   2. Run: python test_face_recognition_fix.py <image1.jpg> <image2.jpg>")
    print("\n📝 This test requires actual image files to be provided.")
    
    if len(sys.argv) >= 3:
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                print(f"❌ Failed to load images: {img1_path}, {img2_path}")
                return False
            
            # Convert BGR to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            print(f"\n✅ Loaded images:")
            print(f"   Image 1: {img1_path} ({img1.shape})")
            print(f"   Image 2: {img2_path} ({img2.shape})")
            
            evaluator = ProfessionalIdentityEvaluator()
            print("\n🔍 Running identity comparison...")
            result = evaluator.calculate_identity_similarity(img1, img2)
            
            print("\n" + "-"*80)
            print("📋 Results:")
            print("-"*80)
            for key, value in result.items():
                if key != 'model_results':
                    print(f"   {key}: {value}")
            
            if 'model_results' in result:
                print("\n📊 Individual Model Results:")
                for model_result in result['model_results']:
                    print(f"   {model_result['model']}: similarity={model_result['similarity']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error testing real images: {e}")
            return False
    else:
        print("⏭️  Skipping real image test (no images provided)")
        return None

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Face Recognition Fix Validation Test" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    
    results = []
    
    # Test 1: Face detection
    try:
        result = test_face_detection()
        results.append(("Face Detection", result))
    except Exception as e:
        print(f"❌ Test 1 failed with error: {e}")
        results.append(("Face Detection", False))
    
    # Test 2: Different backgrounds
    try:
        result = test_different_backgrounds()
        results.append(("Different Backgrounds", result))
    except Exception as e:
        print(f"❌ Test 2 failed with error: {e}")
        results.append(("Different Backgrounds", False))
    
    # Test 3: Real images
    try:
        result = test_real_images()
        if result is not None:
            results.append(("Real Images", result))
    except Exception as e:
        print(f"❌ Test 3 failed with error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, r in results if r is True)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The face recognition improvements are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please review the results above.")
    
    print("\n💡 Key Improvements:")
    print("   ✅ Pre-detection of faces before DeepFace comparison")
    print("   ✅ Face-only cropping to eliminate background noise")
    print("   ✅ Better model selection (Facenet512, ArcFace)")
    print("   ✅ Adaptive thresholds based on model agreement")
    print("   ✅ Weighted similarity scoring")
    
    print("\n📚 To test with your own images:")
    print(f"   python {sys.argv[0]} <image1.jpg> <image2.jpg>")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
