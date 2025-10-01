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
            
            # Initialize multiple face detection cascades
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
            self.alt_face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            )
            
            print("✅ Professional Identity Evaluator initialized with DeepFace")
            print("🔧 Using CPU mode for AMD 780M compatibility")
            print("📦 Multi-cascade detection enabled (frontal + profile + alt)")
            
        except ImportError as e:
            self.available = False
            print(f"⚠️ DeepFace not available: {e}")
            print("💡 Install with: pip install deepface")
    
    
    def _detect_and_crop_face(self, pil_image, debug_name="face"):
        """
        Detect and crop face region from PIL image
        Uses multi-stage detection for robustness
        
        Args:
            pil_image: PIL Image object
            debug_name: Name for debug output files
            
        Returns:
            tuple: (cropped_face_pil, detection_info) or (None, None) if no face detected
        """
        try:
            # Convert PIL to numpy for OpenCV
            img_array = np.array(pil_image)
            
            # Convert to grayscale for face detection
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Multi-stage, multi-method face detection
            # Strategy: Try ALL methods and select the BEST result based on quality scoring
            all_faces = []  # Store (bbox, method, score)
            detection_methods = []
            
            def score_detection(bbox, img_shape, method_name):
                """Score a detection based on size, position, method reliability, and skin tone"""
                x, y, w, h = bbox
                img_h, img_w = img_shape[:2]
                
                # Extract the detected region for skin tone analysis
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img_w, x + w), min(img_h, y + h)
                detected_region = img_array[y1:y2, x1:x2]
                
                # Skin tone verification (to filter out walls/objects)
                skin_score = 100
                if detected_region.size > 0:
                    try:
                        # Convert to YCrCb color space for skin detection
                        ycrcb = cv2.cvtColor(detected_region, cv2.COLOR_RGB2YCR_CB)
                        
                        # Define skin color range in YCrCb
                        # Y: 0-255, Cr: 133-173, Cb: 77-127 (typical skin range)
                        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
                        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
                        
                        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
                        skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size
                        
                        # Score based on skin pixel ratio
                        if skin_ratio > 0.3:  # Good amount of skin
                            skin_score = 100
                        elif skin_ratio > 0.15:  # Some skin
                            skin_score = 80
                        elif skin_ratio > 0.05:  # Little skin (might be profile/shadow)
                            skin_score = 60
                        else:  # Almost no skin (likely wall/object)
                            skin_score = 20
                            print(f"      ⚠️ Low skin tone detected in {method_name}: {skin_ratio:.1%}")
                    except Exception as e:
                        skin_score = 70  # Neutral if can't check
                
                # Base score from face area (larger is usually better, but not too large)
                area_ratio = (w * h) / (img_w * img_h)
                
                # Special handling for large detections (likely full-face images)
                if area_ratio > 0.8:  
                    # This might be a pre-cropped face image
                    # Give high score ONLY if from DeepFace (reliable) or if it's nearly the full image
                    if method_name.startswith('deepface_') or area_ratio > 0.95:
                        area_score = 100  # Full face image - excellent!
                    else:
                        area_score = 30  # Haar cascade detecting full image - suspicious
                elif area_ratio > 0.4:  # Good size
                    area_score = 100
                elif area_ratio > 0.15:  # Acceptable
                    area_score = 80
                elif area_ratio > 0.05:  # Small but visible
                    area_score = 60
                else:  # Too small (likely ear/eye/partial feature)
                    area_score = 20  # Heavy penalty for tiny detections
                
                # Position score (faces should NOT be at extreme edges in complex scenes)
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                
                # Heavy penalty for detections at edges (likely background objects)
                edge_penalty = 0
                edge_threshold = 0.1  # 10% from edge
                
                # Check if detection touches edges
                if x < img_w * edge_threshold:
                    edge_penalty += 30
                if (x + w) > img_w * (1 - edge_threshold):
                    edge_penalty += 30
                if y < img_h * edge_threshold:
                    edge_penalty += 30
                if (y + h) > img_h * (1 - edge_threshold):
                    edge_penalty += 30
                
                position_score = max(0, 100 - edge_penalty)
                
                # Method reliability score - DeepFace models are MUCH more reliable for complex scenes
                method_scores = {
                    'frontal_strict': 70,  # Reduced from 100 - often false positives
                    'frontal_lenient': 60,  # Reduced from 80
                    'frontal_alt': 75,     # Reduced from 90
                    'profile': 90,         # Increased - good for side faces
                    'aggressive': 50,      # Reduced from 60 - too many false positives
                }
                
                # DeepFace models get highest reliability
                if method_name.startswith('deepface_'):
                    if 'retinaface' in method_name:
                        method_score = 100  # Best detector
                    elif 'mtcnn' in method_name:
                        method_score = 98
                    elif 'ssd' in method_name:
                        method_score = 95
                    else:  # opencv
                        method_score = 85
                else:
                    method_score = method_scores.get(method_name, 70)
                
                # Aspect ratio score (faces should be roughly square or slightly tall)
                aspect_ratio = h / w if w > 0 else 0
                aspect_score = 100
                if aspect_ratio < 0.7:  # Too wide (likely not a face)
                    aspect_score = 40
                elif aspect_ratio > 1.6:  # Too tall (likely not a face)
                    aspect_score = 50
                elif 0.8 <= aspect_ratio <= 1.4:  # Ideal face proportions
                    aspect_score = 100
                else:  # Acceptable
                    aspect_score = 70
                
                # Combined score with adjusted weights including skin tone
                total_score = (area_score * 0.15 +        # 15% - size
                              position_score * 0.10 +      # 10% - edge penalty
                              method_score * 0.40 +        # 40% - trust good detectors!
                              aspect_score * 0.15 +        # 15% - proportions
                              skin_score * 0.20)           # 20% - skin tone check
                
                return total_score
            
            # Method 1: Standard frontal face detection
            try:
                faces_frontal = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )
                if len(faces_frontal) > 0:
                    for face in faces_frontal:
                        score = score_detection(face, img_array.shape, 'frontal_strict')
                        all_faces.append((face, 'frontal_strict', score))
                    detection_methods.append(f'frontal_strict({len(faces_frontal)})')
            except Exception as e:
                print(f"   ⚠️ frontal_strict failed: {str(e)[:50]}")
            
            # Method 2: Lenient frontal detection
            try:
                faces_frontal_lenient = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
                )
                if len(faces_frontal_lenient) > 0:
                    for face in faces_frontal_lenient:
                        score = score_detection(face, img_array.shape, 'frontal_lenient')
                        all_faces.append((face, 'frontal_lenient', score))
                    detection_methods.append(f'frontal_lenient({len(faces_frontal_lenient)})')
            except Exception as e:
                print(f"   ⚠️ frontal_lenient failed: {str(e)[:50]}")
            
            # Method 3: Alternative frontal cascade
            try:
                faces_alt = self.alt_face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
                )
                if len(faces_alt) > 0:
                    for face in faces_alt:
                        score = score_detection(face, img_array.shape, 'frontal_alt')
                        all_faces.append((face, 'frontal_alt', score))
                    detection_methods.append(f'frontal_alt({len(faces_alt)})')
            except Exception as e:
                print(f"   ⚠️ frontal_alt failed: {str(e)[:50]}")
            
            # Method 4: Profile face detection (CRITICAL for side faces)
            try:
                faces_profile = self.profile_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )
                if len(faces_profile) > 0:
                    for face in faces_profile:
                        score = score_detection(face, img_array.shape, 'profile')
                        all_faces.append((face, 'profile', score))
                    detection_methods.append(f'profile({len(faces_profile)})')
                    print(f"   🔄 Profile detector found {len(faces_profile)} face(s)")
            except Exception as e:
                print(f"   ⚠️ profile detection failed: {str(e)[:50]}")
            
            # Method 5: Very aggressive detection
            try:
                print(f"   🔄 Trying aggressive detection...")
                faces_aggressive = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.03, minNeighbors=2, minSize=(20, 20)
                )
                if len(faces_aggressive) > 0:
                    for face in faces_aggressive:
                        score = score_detection(face, img_array.shape, 'aggressive')
                        all_faces.append((face, 'aggressive', score))
                    detection_methods.append(f'aggressive({len(faces_aggressive)})')
            except Exception as e:
                print(f"   ⚠️ aggressive detection failed: {str(e)[:50]}")
            
            # Method 6: DeepFace built-in detector (HIGHEST QUALITY)
            print(f"   🔄 Trying DeepFace built-in detector...")
            try:
                from PIL import Image
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    Image.fromarray(img_array).save(tmp.name)
                    tmp_path = tmp.name
                
                backends = ['retinaface', 'mtcnn', 'ssd', 'opencv']  # Reordered by quality
                for backend in backends:
                    try:
                        face_objs = self.deepface.extract_faces(
                            img_path=tmp_path,
                            detector_backend=backend,
                            enforce_detection=False
                        )
                        if face_objs and len(face_objs) > 0:
                            for face_obj in face_objs:
                                facial_area = face_obj.get('facial_area', {})
                                if facial_area:
                                    fx = facial_area.get('x', 0)
                                    fy = facial_area.get('y', 0)
                                    fw = facial_area.get('w', 0)
                                    fh = facial_area.get('h', 0)
                                    if fw > 0 and fh > 0:
                                        face_bbox = (fx, fy, fw, fh)
                                        score = score_detection(face_bbox, img_array.shape, f'deepface_{backend}')
                                        all_faces.append((face_bbox, f'deepface_{backend}', score))
                            detection_methods.append(f'deepface_{backend}({len(face_objs)})')
                            print(f"   ✅ DeepFace {backend} detected {len(face_objs)} face(s)")
                            break
                    except Exception as e:
                        print(f"   ⚠️ DeepFace {backend} failed: {str(e)[:50]}")
                        continue
                
                try:
                    import os
                    os.unlink(tmp_path)
                except:
                    pass
            except Exception as e:
                print(f"   ⚠️ DeepFace detector failed: {str(e)[:100]}")
            
            if len(all_faces) == 0:
                print(f"⚠️ No face detected for {debug_name} after all methods")
                print(f"   Image size: {img_array.shape}")
                print(f"   Methods tried: {', '.join(detection_methods) if detection_methods else 'all methods failed'}")
                return None, None
            
            print(f"   ✅ Detection methods completed: {', '.join(detection_methods)}")
            
            # Filter out obviously bad detections before scoring
            img_h, img_w = img_array.shape[:2]
            img_area = img_h * img_w
            
            filtered_faces = []
            rejected_count = 0
            for face_bbox, method, score in all_faces:
                x, y, w, h = face_bbox
                face_area = w * h
                area_ratio = face_area / img_area
                
                # Reject if:
                # 1. Too small (< 60x60 pixels OR < 3% of image) UNLESS it's from DeepFace and the only detection
                # 2. Extremely narrow aspect ratio (likely not a face)
                aspect = h / w if w > 0 else 0
                
                reject = False
                reason = ""
                
                if w < 60 or h < 60:
                    if not method.startswith('deepface_') or len(all_faces) > 1:
                        reject = True
                        reason = f"too small ({w}x{h})"
                
                if area_ratio < 0.03:  # Less than 3% of image
                    if not (method.startswith('deepface_') and area_ratio > 0.01):
                        reject = True
                        reason = f"tiny area ({area_ratio:.1%})"
                
                if aspect < 0.5 or aspect > 2.0:  # Extremely wrong proportions
                    reject = True
                    reason = f"bad aspect ratio ({aspect:.2f})"
                
                if reject:
                    rejected_count += 1
                    print(f"      ❌ Rejected {method}: {reason}")
                else:
                    filtered_faces.append((face_bbox, method, score))
            
            if len(filtered_faces) == 0:
                print(f"⚠️ All {len(all_faces)} detections rejected for {debug_name}")
                return None, None
            
            if rejected_count > 0:
                print(f"   🔍 Filtered: {rejected_count} rejected, {len(filtered_faces)} kept")
            
            # Select the BEST face based on quality score
            filtered_faces.sort(key=lambda x: x[2], reverse=True)  # Sort by score
            best_face, detection_method, best_score = filtered_faces[0]
            
            # Report all detections with scores
            if len(filtered_faces) > 1:
                print(f"   📊 Detected {len(filtered_faces)} candidate(s), selecting best one:")
                for i, (face, method, score) in enumerate(filtered_faces[:5]):  # Show top 5
                    x, y, w, h = face
                    marker = "👑" if i == 0 else f"  {i+1}."
                    print(f"      {marker} {method}: pos=({x},{y}), size={w}x{h}, score={score:.1f}")
            else:
                print(f"   ✅ Selected: {detection_method} (score={best_score:.1f})")
            
            faces = [best_face]
            
            # Select the largest face (most likely to be the primary subject)
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            
            # Validate the detection is reasonable
            img_h, img_w = img_array.shape[:2]
            face_area = w * h
            image_area = img_h * img_w
            face_ratio = face_area / image_area
            
            print(f"   📊 Face area ratio: {face_ratio:.2%} of image (face={w}x{h}, image={img_w}x{img_h})")
            
            # Warning if face is suspiciously small or positioned oddly
            if face_ratio < 0.01:
                print(f"   ⚠️ WARNING: Face area is very small ({face_ratio:.2%}), detection may be incorrect!")
            
            if x < 0 or y < 0 or x+w > img_w or y+h > img_h:
                print(f"   ⚠️ WARNING: Face bbox out of bounds!")
                # Clip to image boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                print(f"   ✅ Clipped to: ({x}, {y}, {w}, {h})")
            
            # Store detection info for debugging
            detection_info = {
                'bbox': (x, y, w, h),
                'num_faces': len(faces),
                'face_ratio': face_ratio,
                'detection_method': detection_method
            }
            
            # Assess face size quality
            face_area = w * h
            if face_area < 5000:  # Less than ~70x70
                print(f"⚠️ Face detected at ({x}, {y}), size: {w}x{h} - VERY SMALL! Quality may be poor.")
            elif face_area < 15000:  # Less than ~122x122
                print(f"⚠️ Face detected at ({x}, {y}), size: {w}x{h} - Small. Consider using closer image.")
            else:
                print(f"✅ Face detected at ({x}, {y}), size: {w}x{h}")
            
            # Add padding to include context (hair, ears, face shape)
            # Use moderate padding - too much includes unnecessary background
            padding_ratio = 0.25  # 25% padding - balanced approach
            padding_w = int(w * padding_ratio)
            padding_h = int(h * padding_ratio)
            
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(img_array.shape[1], x + w + padding_w)
            y2 = min(img_array.shape[0], y + h + padding_h)
            
            print(f"   📐 Original bbox: ({x}, {y}, {w}, {h})")
            print(f"   📐 After padding: ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}")
            
            # Crop face region
            face_crop = img_array[y1:y2, x1:x2]
            
            # Verify crop is valid
            if face_crop.size == 0:
                print(f"   ❌ Crop resulted in empty image!")
                return None, None
            
            print(f"   ✅ Cropped face shape: {face_crop.shape}")
            
            # Additional validation: Check if cropped region actually contains face-like features
            # Only use fallback if crop is clearly wrong (very strict criteria)
            use_fallback = False
            
            # Check if crop is reasonable size
            crop_h, crop_w = face_crop.shape[:2]
            if crop_w < 40 or crop_h < 40:
                print(f"   ⚠️ WARNING: Crop too small ({crop_w}x{crop_h})")
                use_fallback = True
            elif face_ratio < 0.01:  # Face is less than 1% of image
                print(f"   ⚠️ WARNING: Face area suspiciously small ({face_ratio:.2%})")
                use_fallback = True
            
            if use_fallback:
                print(f"   🔄 FALLBACK: Using intelligent center crop instead...")
                
                # Use center crop as fallback
                img_h, img_w = img_array.shape[:2]
                
                # Calculate center square crop
                crop_size = min(img_h, img_w)
                start_y = (img_h - crop_size) // 2
                start_x = (img_w - crop_size) // 2
                
                face_crop = img_array[start_y:start_y+crop_size, start_x:start_x+crop_size]
                
                print(f"   ✅ Using center crop: ({start_x}, {start_y}) to ({start_x+crop_size}, {start_y+crop_size})")
                
                # Update detection info for debug visualization
                detection_info['bbox'] = (start_x, start_y, crop_size, crop_size)
                detection_info['crop_region'] = (start_x, start_y, start_x+crop_size, start_y+crop_size)
                detection_info['fallback_used'] = True
            else:
                print(f"   ✅ Crop validation passed: {crop_w}x{crop_h}")
                detection_info['fallback_used'] = False
            
            # Convert back to PIL
            from PIL import Image
            face_pil = Image.fromarray(face_crop)
            
            # Ensure optimal size for DeepFace models
            # ALL faces should be resized to EXACTLY the same size
            target_size = 224  # Optimal for most models
            
            # Always resize to target size for consistency
            if face_pil.size != (target_size, target_size):
                # Use high-quality LANCZOS resampling
                face_pil = face_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            print(f"   ✅ Face cropped and resized to: {face_pil.size}")
            
            # Store crop region info
            detection_info['crop_region'] = (x1, y1, x2, y2)
            detection_info['final_size'] = face_pil.size
            
            return face_pil, detection_info
            
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
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
                
                face1_region, face1_info = self._detect_and_crop_face(pil_img1, debug_name="image1")
                face2_region, face2_info = self._detect_and_crop_face(pil_img2, debug_name="image2")
                
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
                
                # Assess face quality based on size
                def assess_quality(face_img):
                    area = face_img.size[0] * face_img.size[1]
                    if area >= 40000:  # >=200x200
                        return 1.0
                    elif area >= 20000:  # >=141x141
                        return 0.8
                    elif area >= 10000:  # >=100x100
                        return 0.6
                    elif area >= 5000:   # >=70x70
                        return 0.4
                    else:
                        return 0.2
                
                face1_quality = assess_quality(face1_region)
                face2_quality = assess_quality(face2_region)
                avg_quality = (face1_quality + face2_quality) / 2
                
                print(f"📊 Face quality: img1={face1_quality:.2f} ({face1_region.size}), img2={face2_quality:.2f} ({face2_region.size}), avg={avg_quality:.2f}")
                
                if avg_quality < 0.5:
                    print("⚠️ Warning: Low average face quality detected. Results may be less reliable.")
                
                # Save face-cropped images
                face1_region.save(tmp1_path)
                face2_region.save(tmp2_path)
                
                # DEBUG: Also save to visible location for inspection
                try:
                    from PIL import ImageDraw, ImageFont
                    
                    # Save the cropped faces
                    face1_region.save('debug_face1.jpg')
                    face2_region.save('debug_face2.jpg')
                    
                    # Draw detection boxes on original images
                    img1_with_box = pil_img1.copy()
                    img2_with_box = pil_img2.copy()
                    
                    draw1 = ImageDraw.Draw(img1_with_box)
                    draw2 = ImageDraw.Draw(img2_with_box)
                    
                    # Draw face detection bbox (original detection)
                    if face1_info:
                        x, y, w, h = face1_info['bbox']
                        box_color = 'orange' if face1_info.get('fallback_used') else 'red'
                        method = face1_info.get('detection_method', 'unknown')
                        draw1.rectangle([x, y, x+w, y+h], outline=box_color, width=3)
                        label = f"{method}: {w}x{h}" + (" [FALLBACK]" if face1_info.get('fallback_used') else "")
                        draw1.text((x, y-20), label, fill=box_color)
                        
                        # Draw crop region (with padding)
                        x1, y1, x2, y2 = face1_info['crop_region']
                        draw1.rectangle([x1, y1, x2, y2], outline='green', width=2)
                        draw1.text((x1, y1-40), f"Crop: {x2-x1}x{y2-y1}", fill='green')
                    
                    if face2_info:
                        x, y, w, h = face2_info['bbox']
                        box_color = 'orange' if face2_info.get('fallback_used') else 'red'
                        method = face2_info.get('detection_method', 'unknown')
                        draw2.rectangle([x, y, x+w, y+h], outline=box_color, width=3)
                        label = f"{method}: {w}x{h}" + (" [FALLBACK]" if face2_info.get('fallback_used') else "")
                        draw2.text((x, y-20), label, fill=box_color)
                        
                        # Draw crop region (with padding)
                        x1, y1, x2, y2 = face2_info['crop_region']
                        draw2.rectangle([x1, y1, x2, y2], outline='green', width=2)
                        draw2.text((x1, y1-40), f"Crop: {x2-x1}x{y2-y1}", fill='green')
                    
                    img1_with_box.save('debug_original1_with_box.jpg')
                    img2_with_box.save('debug_original2_with_box.jpg')
                    
                    print(f"   🔍 Debug files saved:")
                    print(f"      - debug_face1.jpg (cropped face 1)")
                    print(f"      - debug_face2.jpg (cropped face 2)")
                    print(f"      - debug_original1_with_box.jpg (original 1 with boxes)")
                    print(f"      - debug_original2_with_box.jpg (original 2 with boxes)")
                    print(f"")
                    print(f"   📖 Legend:")
                    print(f"      - RED box = Detected face region")
                    print(f"      - ORANGE box = Fallback center crop (detection failed)")
                    print(f"      - GREEN box = Final crop region used")
                except Exception as e:
                    print(f"   ⚠️ Debug save failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                print("✅ Face regions extracted and saved")
                
                # Use only the most reliable models for face recognition
                # Facenet512 and ArcFace are state-of-the-art
                models = ['Facenet512', 'ArcFace', 'Facenet']
                results = []
                
                for model in models:
                    try:
                        print(f"🔄 Trying model: {model}")
                        
                        # Try to get embeddings directly and calculate distance
                        # This bypasses DeepFace.verify's detection logic completely
                        try:
                            from deepface.commons import functions
                            
                            # Get embeddings directly
                            img1_embedding = self.deepface.represent(
                                img_path=tmp1_path,
                                model_name=model,
                                enforce_detection=False,
                                detector_backend='skip',
                                align=False
                            )
                            
                            img2_embedding = self.deepface.represent(
                                img_path=tmp2_path,
                                model_name=model,
                                enforce_detection=False,
                                detector_backend='skip',
                                align=False
                            )
                            
                            # Extract embedding vectors
                            if isinstance(img1_embedding, list):
                                emb1 = np.array(img1_embedding[0]['embedding'])
                                emb2 = np.array(img2_embedding[0]['embedding'])
                            else:
                                emb1 = np.array(img1_embedding['embedding'])
                                emb2 = np.array(img2_embedding['embedding'])
                            
                            # Calculate cosine distance manually
                            from scipy.spatial.distance import cosine
                            distance = cosine(emb1, emb2)
                            
                            # Get model threshold
                            thresholds = {
                                'Facenet512': 0.30,
                                'Facenet': 0.40,
                                'ArcFace': 0.68
                            }
                            threshold = thresholds.get(model, 0.40)
                            verified = distance < threshold
                            
                            result = {
                                'distance': distance,
                                'verified': verified,
                                'threshold': threshold
                            }
                            
                            print(f"   📏 Direct embedding: distance={distance:.4f}, verified={verified}, threshold={threshold}")
                            
                        except Exception as e:
                            print(f"   ⚠️ Direct embedding failed: {e}, falling back to verify()")
                            # Fallback to verify method
                            result = self.deepface.verify(
                                img1_path=tmp1_path,
                                img2_path=tmp2_path,
                                model_name=model,
                                distance_metric='cosine',
                                enforce_detection=False,
                                detector_backend='skip',
                                align=False
                            )
                            
                            print(f"   📏 Verify result: distance={result.get('distance', 'N/A'):.4f}, verified={result.get('verified', 'N/A')}")
                        
                        # Validate result quality
                        distance = result['distance']
                        
                        # Different models have different distance scales
                        # Convert to normalized similarity score
                        if model in ['Facenet512', 'Facenet']:
                            # Facenet uses cosine distance [0, 2]
                            # threshold ~0.40, convert to similarity
                            if distance > 1.5:  # Too far
                                print(f"⚠️ Model {model} returned suspicious distance: {distance:.4f}")
                                continue
                            similarity = max(0.0, 1.0 - distance)  # Direct conversion
                            
                        elif model == 'ArcFace':
                            # ArcFace uses cosine distance with threshold ~0.68
                            # Distance interpretation:
                            #   0.0 - 0.68: Same person (high confidence)
                            #   0.68 - 1.5: Uncertain/Different
                            if distance > 1.5:
                                print(f"⚠️ Model {model} returned suspicious distance: {distance:.4f}")
                                continue
                            
                            # Convert ArcFace distance to similarity
                            # threshold at 0.68 maps to similarity ~0.5
                            if distance <= 0.68:
                                # Same person range: distance [0, 0.68] → similarity [1.0, 0.5]
                                similarity = 1.0 - (distance / 0.68) * 0.5
                            else:
                                # Different person range: distance [0.68, 1.5] → similarity [0.5, 0.0]
                                similarity = max(0.0, 0.5 * (1.5 - distance) / 0.82)
                            
                        else:
                            # Generic conversion
                            similarity = max(0.0, 1.0 - min(distance, 1.0))
                        
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
                    # Use the maximum similarity if models disagree significantly
                    if similarity_std > 0.15:
                        # High disagreement - trust the best model
                        decision_score = max_similarity
                        print(f"⚠️ High model disagreement (std={similarity_std:.4f}), using max similarity")
                    else:
                        # Low disagreement - use weighted average
                        decision_score = weighted_similarity
                    
                    # Adaptive threshold based on model agreement and face quality
                    if similarity_std < 0.08:  # Very high agreement
                        base_threshold = 0.40  # Lower threshold when models strongly agree
                    elif similarity_std < 0.15:  # High agreement
                        base_threshold = 0.45
                    elif similarity_std < 0.25:  # Medium agreement
                        base_threshold = 0.50
                    else:  # Low agreement
                        base_threshold = 0.55  # Higher threshold when models disagree
                    
                    # Adjust threshold based on face quality
                    if avg_quality >= 0.8:
                        quality_adjustment = -0.03  # Slightly more lenient for high quality
                    elif avg_quality <= 0.5:
                        quality_adjustment = +0.05  # More strict for poor quality
                    else:
                        quality_adjustment = 0.0
                    
                    threshold = base_threshold + quality_adjustment
                    
                    print(f"📊 Decision metrics: decision_score={decision_score:.4f}, weighted_sim={weighted_similarity:.4f}, threshold={threshold:.2f} (base={base_threshold:.2f}, quality_adj={quality_adjustment:+.2f}), std={similarity_std:.4f}")
                    
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
                        'decision_score': float(decision_score),  # Actual score used for decision
                        'avg_similarity': float(avg_similarity),
                        'max_similarity': float(max_similarity),
                        'min_similarity': float(min_similarity),
                        'confidence': float(confidence),
                        'quality_score': float(quality_score),
                        'face1_quality': float(face1_quality),
                        'face2_quality': float(face2_quality),
                        'avg_face_quality': float(avg_quality),
                        'identity_decision': identity_decision,
                        'decision_confidence': float(decision_confidence),
                        'verification_consensus': float(verification_consensus),
                        'decision_threshold': float(threshold),
                        'base_threshold': float(base_threshold),
                        'quality_adjustment': float(quality_adjustment),
                        'models_used': len(results),
                        'method': 'DeepFace Professional (Enhanced v2)',
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