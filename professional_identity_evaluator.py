#!/usr/bin/env python3
"""
Professional Identity Evaluator using DeepFace
AMD 780M Optimized Version
"""

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
            
            print("✅ Professional Identity Evaluator initialized with DeepFace")
            print("🔧 Using CPU mode for AMD 780M compatibility")
            
        except ImportError as e:
            self.available = False
            print(f"⚠️ DeepFace not available: {e}")
            print("💡 Install with: pip install deepface")
    
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
            
            # Ensure images are in RGB format
            if len(image1.shape) == 3 and image1.shape[2] == 3:
                # OpenCV uses BGR, convert to RGB
                if image1.dtype == np.uint8:
                    img1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                else:
                    img1_rgb = image1
            else:
                img1_rgb = image1
                
            if len(image2.shape) == 3 and image2.shape[2] == 3:
                if image2.dtype == np.uint8:
                    img2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                else:
                    img2_rgb = image2
            else:
                img2_rgb = image2
            
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
                # Use DeepFace for verification
                # Try multiple models for robustness
                models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
                results = []
                
                for model in models:
                    try:
                        # Verify if the faces are the same person
                        result = self.deepface.verify(
                            img1_path=tmp1_path,
                            img2_path=tmp2_path,
                            model_name=model,
                            distance_metric='cosine',
                            enforce_detection=False  # Don't fail if face detection fails
                        )
                        
                        results.append({
                            'model': model,
                            'verified': result['verified'],
                            'distance': result['distance'],
                            'similarity': 1.0 - result['distance']  # Convert distance to similarity
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Model {model} failed: {e}")
                        continue
                
                # Calculate final metrics
                if results:
                    # Average similarity across all successful models
                    similarities = [r['similarity'] for r in results]
                    verifications = [r['verified'] for r in results]
                    
                    avg_similarity = np.mean(similarities)
                    max_similarity = np.max(similarities)
                    min_similarity = np.min(similarities)
                    verification_consensus = np.mean(verifications)
                    
                    # Calculate confidence based on model agreement
                    similarity_std = np.std(similarities)
                    confidence = max(0.0, 1.0 - similarity_std)
                    
                    # Professional analysis
                    if verification_consensus >= 0.5:
                        identity_decision = "Same Person"
                        decision_confidence = verification_consensus
                    else:
                        identity_decision = "Different Person"
                        decision_confidence = 1.0 - verification_consensus
                    
                    final_result = {
                        'similarity': float(avg_similarity),
                        'max_similarity': float(max_similarity),
                        'min_similarity': float(min_similarity),
                        'confidence': float(confidence),
                        'identity_decision': identity_decision,
                        'decision_confidence': float(decision_confidence),
                        'verification_consensus': float(verification_consensus),
                        'models_used': len(results),
                        'method': 'DeepFace Professional',
                        'model_results': results
                    }
                    
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
        """Fallback evaluation when DeepFace is not available"""
        
        return {
            'similarity': 0.0,
            'confidence': 0.0,
            'identity_decision': "DeepFace Not Available",
            'decision_confidence': 0.0,
            'verification_consensus': 0.0,
            'models_used': 0,
            'method': 'Fallback',
            'error': 'DeepFace library not available'
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