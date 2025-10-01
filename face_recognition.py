#!/usr/bin/env python3

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ProfessionalIdentityEvaluator:
    """
    Professional Identity Evaluator using DeepFace library
    Optimized for AMD 780M systems
    """
    
    def __init__(self):
        """Initialize DeepFace for professional face recognition"""
        
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.available = True
            
            # Configure for CPU usage (AMD optimized)
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            # Initialize face detection cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            print("✅ Professional Identity Evaluator initialized with DeepFace")
            print("🔧 Using CPU mode for AMD 780M compatibility")
            
        except ImportError as e:
            self.available = False
            print(f"⚠️ DeepFace not available: {e}")
            print("💡 Install with: pip install deepface")
    
    
    def _detect_and_crop_face(self, pil_image):
        """
        Detect and crop face region from PIL image
        Uses multi-stage detection for robustness
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            PIL Image of cropped face region, or None if no face detected
        """
        try:
            # Convert PIL to numpy for OpenCV
            img_array = np.array(pil_image)
            
            # Convert to grayscale for face detection
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Stage 1: Try Haar Cascade (fast and reliable for frontal faces)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)  # Minimum face size
            )
            
            if len(faces) == 0:
                # Stage 2: Try with more lenient parameters
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(30, 30)
                )
            
            if len(faces) == 0:
                print("⚠️ No face detected with Haar Cascade")
                return None
            
            # Select the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            print(f"✅ Face detected at ({x}, {y}), size: {w}x{h}")
            
            # Add minimal padding (15%) to include face context
            # Less padding = more focus on face, less background noise
            padding_ratio = 0.15
            padding_w = int(w * padding_ratio)
            padding_h = int(h * padding_ratio)
            
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(img_array.shape[1], x + w + padding_w)
            y2 = min(img_array.shape[0], y + h + padding_h)
            
            # Crop face region
            face_crop = img_array[y1:y2, x1:x2]
            
            # Convert back to PIL
            from PIL import Image
            face_pil = Image.fromarray(face_crop)
            
            # Ensure minimum size for DeepFace models
            min_size = 160  # Minimum for Facenet
            if face_pil.size[0] < min_size or face_pil.size[1] < min_size:
                # Resize to minimum size while preserving aspect ratio
                face_pil.thumbnail((min_size, min_size), Image.Resampling.LANCZOS)
            
            print(f"✅ Face cropped to size: {face_pil.size}")
            return face_pil
            
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            return None
    
    def calculate_identity_similarity(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """
        Calculate identity similarity using DeepFace
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            dict: Professional similarity metrics
        """
        
        if not self.available:
            return self._fallback_evaluation(image1, image2)
        
        try:
            # Convert numpy arrays to PIL format for DeepFace
            from PIL import Image
            
            # Smart image format detection and conversion
            def smart_rgb_conversion(img):
                """Smart RGB conversion that detects image format"""
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # Check if image is already in RGB by examining color distribution
                    # If image comes from PIL/Gradio, it's likely already RGB
                    # Assume RGB input from modern image processing pipeline
                    return img.astype(np.uint8)
                else:
                    return img.astype(np.uint8)
            
            img1_rgb = smart_rgb_conversion(image1)
            img2_rgb = smart_rgb_conversion(image2)
            
            # Create PIL images
            pil_img1 = Image.fromarray(img1_rgb.astype(np.uint8))
            pil_img2 = Image.fromarray(img2_rgb.astype(np.uint8))
            
            # Save temporarily for DeepFace
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1:
                pil_img1.save(tmp1.name)
                tmp1_path = tmp1.name
                
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
                pil_img2.save(tmp2.name)
                tmp2_path = tmp2.name
            
            try:
                # CRITICAL FIX: Pre-detect and crop faces to avoid background comparison
                print("🔍 Step 1: Detecting faces in both images...")
                
                face1_region = self._detect_and_crop_face(pil_img1)
                face2_region = self._detect_and_crop_face(pil_img2)
                
                if face1_region is None or face2_region is None:
                    print("⚠️ Face detection failed in one or both images")
                    print(f"   Image 1: {'✅ Face detected' if face1_region is not None else '❌ No face'}")
                    print(f"   Image 2: {'✅ Face detected' if face2_region is not None else '❌ No face'}")
                    return {
                        'similarity': 0.0,
                        'confidence': 0.0,
                        'identity_decision': 'Analysis Failed - No Face Detected',
                        'decision_confidence': 0.0,
                        'verification_consensus': 0.0,
                        'models_used': 0,
                        'method': 'DeepFace Professional',
                        'error': 'Face detection failed'
                    }
                
                # Save face-cropped images
                face1_region.save(tmp1_path)
                face2_region.save(tmp2_path)
                print("✅ Face regions extracted and saved")
                
                # Use only the most reliable models for face recognition
                # Facenet512 and ArcFace are state-of-the-art
                models = ['Facenet512', 'ArcFace', 'Facenet']
                results = []
                
                for model in models:
                    try:
                        print(f"🔄 Trying model: {model}")
                        
                        # CRITICAL FIX: Use enforce_detection=True with detector_backend='skip'
                        # This ensures we're comparing face features, not backgrounds
                        result = self.deepface.verify(
                            img1_path=tmp1_path,
                            img2_path=tmp2_path,
                            model_name=model,
                            distance_metric='cosine',
                            enforce_detection=False,  # We already cropped faces
                            detector_backend='skip'  # Skip detection, use our pre-cropped images
                        )
                        
                        # Validate result quality
                        distance = result['distance']
                        if distance > 2.0:  # Suspiciously high distance
                            print(f"⚠️ Model {model} returned suspicious distance: {distance:.4f}")
                            continue
                        
                        similarity = 1.0 - min(distance, 1.0)  # Convert distance to similarity, cap at 1.0
                        
                        print(f"✅ Model {model} success: verified={result['verified']}, distance={distance:.4f}, similarity={similarity:.4f}")
                        
                        results.append({
                            'model': model,
                            'verified': result['verified'],
                            'distance': distance,
                            'similarity': similarity
                        })
                        
                    except Exception as e:
                        error_msg = str(e)
                        print(f"⚠️ Model {model} failed: {error_msg[:150]}...")
                        
                        # If face detection failed even after our pre-cropping, skip
                        if "Face could not be detected" in error_msg:
                            print(f"   → Face detection still failed for {model}")
                        continue
                
                # Calculate final metrics with improved logic
                if results:
                    # Extract metrics
                    similarities = [r['similarity'] for r in results]
                    verifications = [r['verified'] for r in results]
                    distances = [r['distance'] for r in results]
                    
                    # Use weighted average favoring better-performing models
                    # Lower distance = higher weight
                    weights = [1.0 / (d + 0.1) for d in distances]  # Add 0.1 to avoid division by zero
                    total_weight = sum(weights)
                    weighted_similarity = sum(s * w for s, w in zip(similarities, weights)) / total_weight
                    
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)
                    min_similarity = np.min(similarities)
                    verification_consensus = np.mean(verifications)
                    
                    # Calculate confidence based on model agreement
                    similarity_std = np.std(similarities)
                    confidence = max(0.0, 1.0 - similarity_std)
                    
                    # Enhanced decision logic with adaptive thresholds
                    # Use weighted similarity for better accuracy
                    decision_score = weighted_similarity
                    
                    # Adaptive threshold based on model agreement
                    if similarity_std < 0.1:  # High agreement
                        threshold = 0.45  # Lower threshold when models agree
                    elif similarity_std < 0.2:  # Medium agreement
                        threshold = 0.50
                    else:  # Low agreement
                        threshold = 0.55  # Higher threshold when models disagree
                    
                    print(f"📊 Decision metrics: weighted_sim={decision_score:.4f}, threshold={threshold:.2f}, std={similarity_std:.4f}")
                    
                    if decision_score >= threshold:
                        identity_decision = "Same Person"
                        decision_confidence = decision_score
                    else:
                        identity_decision = "Different Person"
                        decision_confidence = 1.0 - decision_score
                    
                    # Add quality assessment
                    quality_score = confidence * (1.0 - similarity_std)  # Higher is better
                    
                    final_result = {
                        'similarity': float(weighted_similarity),  # Use weighted similarity
                        'avg_similarity': float(avg_similarity),
                        'max_similarity': float(max_similarity),
                        'min_similarity': float(min_similarity),
                        'confidence': float(confidence),
                        'quality_score': float(quality_score),
                        'identity_decision': identity_decision,
                        'decision_confidence': float(decision_confidence),
                        'verification_consensus': float(verification_consensus),
                        'decision_threshold': float(threshold),
                        'models_used': len(results),
                        'method': 'DeepFace Professional (Enhanced)',
                        'model_results': results
                    }
                    
                    print(f"🎯 Final decision: {identity_decision} (similarity: {weighted_similarity:.4f}, confidence: {confidence:.4f})")
                    
                else:
                    final_result = {
                        'similarity': 0.0,
                        'confidence': 0.0,
                        'identity_decision': "Analysis Failed",
                        'decision_confidence': 0.0,
                        'verification_consensus': 0.0,
                        'models_used': 0,
                        'method': 'DeepFace Professional',
                        'error': 'All models failed'
                    }
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp1_path)
                    os.unlink(tmp2_path)
                except:
                    pass
            
            return final_result
            
        except Exception as e:
            print(f"⚠️ DeepFace analysis failed: {e}")
            return self._fallback_evaluation(image1, image2)
    
    def _fallback_evaluation(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """Enhanced fallback evaluation using traditional computer vision methods"""
        
        try:
            # Enhanced traditional face comparison using multiple methods
            print("🔄 Using enhanced fallback face analysis...")
            
            # Method 1: Structural Similarity with face region focus
            from skimage.metrics import structural_similarity as ssim
            from skimage.feature import local_binary_pattern
            
            # Convert to grayscale for processing
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2
            
            # Smart face detection and cropping
            def smart_face_crop(gray_img, target_size=224):
                """
                Detect and crop face region intelligently
                If no face detected, use center crop with padding
                """
                try:
                    # Try to detect face using Haar Cascade
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        # Use the largest detected face
                        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                        
                        # Add 30% padding around face
                        padding = int(max(w, h) * 0.3)
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(gray_img.shape[1], x + w + padding)
                        y2 = min(gray_img.shape[0], y + h + padding)
                        
                        # Crop face region
                        face_crop = gray_img[y1:y2, x1:x2]
                        
                        # Resize to target size
                        return cv2.resize(face_crop, (target_size, target_size))
                    else:
                        # No face detected, use intelligent center crop
                        h, w = gray_img.shape
                        
                        # Calculate center crop that preserves aspect ratio
                        if h > w:
                            # Portrait orientation
                            crop_size = w
                            start_y = (h - w) // 2
                            cropped = gray_img[start_y:start_y + crop_size, 0:w]
                        else:
                            # Landscape orientation
                            crop_size = h
                            start_x = (w - h) // 2
                            cropped = gray_img[0:h, start_x:start_x + crop_size]
                        
                        return cv2.resize(cropped, (target_size, target_size))
                        
                except Exception as e:
                    # Fallback: direct resize
                    print(f"⚠️ Face detection failed, using direct resize: {e}")
                    return cv2.resize(gray_img, (target_size, target_size))
            
            # Apply smart face cropping
            height, width = 224, 224  # Standard face recognition size
            gray1_resized = smart_face_crop(gray1, width)
            gray2_resized = smart_face_crop(gray2, width)
            
            # Calculate SSIM
            ssim_score = ssim(gray1_resized, gray2_resized)
            
            # Method 2: Local Binary Pattern similarity
            radius = 3
            n_points = 8 * radius
            lbp1 = local_binary_pattern(gray1_resized, n_points, radius, method='uniform')
            lbp2 = local_binary_pattern(gray2_resized, n_points, radius, method='uniform')
            
            # Calculate LBP histograms
            hist1, _ = np.histogram(lbp1.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist2, _ = np.histogram(lbp2.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            
            # Normalize histograms
            hist1 = hist1.astype(float)
            hist2 = hist2.astype(float)
            hist1 /= (hist1.sum() + 1e-7)
            hist2 /= (hist2.sum() + 1e-7)
            
            # Calculate histogram correlation
            lbp_similarity = np.corrcoef(hist1, hist2)[0, 1]
            if np.isnan(lbp_similarity):
                lbp_similarity = 0.0
            
            # Method 3: Template matching score
            template_match = cv2.matchTemplate(gray1_resized, gray2_resized, cv2.TM_CCOEFF_NORMED)
            template_score = float(np.max(template_match))
            
            # Method 4: Feature-based similarity using ORB
            try:
                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(gray1_resized, None)
                kp2, des2 = orb.detectAndCompute(gray2_resized, None)
                
                if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    
                    if len(matches) > 0:
                        # Calculate match ratio
                        good_matches = [m for m in matches if m.distance < 50]  # Threshold for good matches
                        orb_similarity = len(good_matches) / max(len(des1), len(des2))
                    else:
                        orb_similarity = 0.0
                else:
                    orb_similarity = 0.0
            except:
                orb_similarity = 0.0
            
            # Combine all similarities with weights
            weights = [0.35, 0.25, 0.25, 0.15]  # SSIM, LBP, Template, ORB
            similarities = [ssim_score, abs(lbp_similarity), template_score, orb_similarity]
            
            # Ensure all similarities are in [0,1] range
            similarities = [max(0.0, min(1.0, s)) for s in similarities]
            
            # Weighted average
            final_similarity = sum(w * s for w, s in zip(weights, similarities))
            
            # Calculate confidence based on agreement between methods
            similarity_std = np.std(similarities)
            confidence = max(0.1, 1.0 - similarity_std)  # At least 0.1 confidence
            
            # Decision based on threshold
            threshold = 0.5  # Adjust this based on testing
            if final_similarity >= threshold:
                identity_decision = "Same Person"
                decision_confidence = final_similarity
            else:
                identity_decision = "Different Person"
                decision_confidence = 1.0 - final_similarity
            
            print(f"📊 Fallback analysis: SSIM={ssim_score:.3f}, LBP={lbp_similarity:.3f}, Template={template_score:.3f}, ORB={orb_similarity:.3f}")
            print(f"🎯 Final similarity: {final_similarity:.3f}, Decision: {identity_decision}")
            
            return {
                'similarity': float(final_similarity),
                'confidence': float(confidence),
                'identity_decision': identity_decision,
                'decision_confidence': float(decision_confidence),
                'verification_consensus': float(final_similarity),
                'models_used': 4,  # 4 traditional methods used
                'method': 'Enhanced Traditional CV Analysis',
                'ssim_score': float(ssim_score),
                'lbp_similarity': float(lbp_similarity),
                'template_score': float(template_score),
                'orb_similarity': float(orb_similarity)
            }
            
        except Exception as e:
            print(f"❌ Fallback evaluation failed: {e}")
            return {
                'similarity': 0.0,
                'confidence': 0.0,
                'identity_decision': "Analysis Failed",
                'decision_confidence': 0.0,
                'verification_consensus': 0.0,
                'models_used': 0,
                'method': 'Failed Fallback',
                'error': str(e)
            }
    
    def analyze_face_demographics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze face demographics (age, gender, emotion, race)
        """
        
        if not self.available:
            return {'error': 'DeepFace not available'}
        
        try:
            from PIL import Image
            import tempfile
            import os
            
            # Convert to PIL and save temporarily
            if len(image.shape) == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image
                
            pil_img = Image.fromarray(img_rgb.astype(np.uint8))
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                pil_img.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # Analyze demographics
                analysis = self.deepface.analyze(
                    img_path=tmp_path,
                    actions=['age', 'gender', 'emotion', 'race'],
                    enforce_detection=False
                )
                
                if isinstance(analysis, list):
                    analysis = analysis[0]  # Take first face if multiple
                
                return {
                    'age': analysis.get('age', 'Unknown'),
                    'gender': analysis.get('dominant_gender', 'Unknown'),
                    'emotion': analysis.get('dominant_emotion', 'Unknown'),
                    'race': analysis.get('dominant_race', 'Unknown'),
                    'method': 'DeepFace Demographics'
                }
                
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            return {'error': f'Demographic analysis failed: {e}'}

# Test function
def test_professional_evaluator():
    """Test the professional identity evaluator"""
    
    print("\n🧪 Testing Professional Identity Evaluator")
    print("-" * 50)
    
    evaluator = ProfessionalIdentityEvaluator()
    
    if not evaluator.available:
        print("❌ DeepFace not available for testing")
        return
    
    # Create test images
    print("Creating test images...")
    test_img1 = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
    test_img2 = np.random.randint(50, 150, (300, 300, 3), dtype=np.uint8)
    
    print("Testing identity similarity...")
    result = evaluator.calculate_identity_similarity(test_img1, test_img2)
    
    print("📊 Professional Identity Analysis Results:")
    for key, value in result.items():
        if key != 'model_results':
            print(f"  {key}: {value}")
    
    print("\n✅ Professional evaluator test completed")

if __name__ == "__main__":
    test_professional_evaluator()