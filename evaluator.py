#!/usr/bin/env python3
"""
Compatibility-Enhanced Evaluation System
Handles various dependency and compatibility issues gracefully
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class CompatibleEvaluationSystem:
    """
    Compatibility-enhanced evaluation system that gracefully handles:
    1. TensorFlow version conflicts
    2. Missing optional dependencies
    3. Model loading failures
    4. Hardware limitations
    """
    
    def __init__(self):
        """Initialize with fallback mechanisms"""
        self.available_methods = {}
        self.professional_evaluator = None
        
        print("🔧 Initializing Compatible Evaluation System...")
        
        # Initialize professional evaluator first
        self._initialize_professional_evaluator()
        
        # Try to initialize advanced methods with fallbacks
        self._initialize_clip_safe()
        self._initialize_lpips_safe()
        
        print(f"✅ Compatible system ready with methods: {list(self.available_methods.keys())}")
    
    def _initialize_professional_evaluator(self):
        """Initialize professional evaluator with compatibility checks"""
        try:
            from face_recognition import ProfessionalIdentityEvaluator
            self.professional_evaluator = ProfessionalIdentityEvaluator()
            self.available_methods['professional'] = True
            print("✅ Professional evaluator loaded")
        except Exception as e:
            print(f"⚠️ Professional evaluator failed: {e}")
            self.available_methods['professional'] = False
    
    def _initialize_clip_safe(self):
        """Safely initialize CLIP with comprehensive error handling and timeout"""
        try:
            import torch
            import threading
            import time
            
            # Try different CLIP implementations
            clip_loaded = False
            
            def load_clip_with_timeout():
                nonlocal clip_loaded
                try:
                    import clip
                    # Force CPU for AMD 780M compatibility
                    device = "cpu"
                    print(f"🔧 Loading CLIP on {device} (AMD 780M optimized)")
                    
                    # Load with explicit CPU device and timeout protection
                    self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
                    # Ensure model is in float32 and properly initialized
                    self.clip_model = self.clip_model.float().eval()
                    self.clip_device = device
                    clip_loaded = True
                    print("✅ CLIP (ViT-B/32) loaded successfully on CPU")
                    
                except Exception as e1:
                    print(f"⚠️ CLIP ViT-B/32 failed: {str(e1)[:100]}...")
                    
                    # Try smaller model as fallback
                    try:
                        import clip
                        self.clip_model, self.clip_preprocess = clip.load("RN50", device="cpu")
                        self.clip_model = self.clip_model.float().eval()
                        self.clip_device = "cpu"
                        clip_loaded = True
                        print("✅ CLIP (RN50) loaded as fallback on CPU")
                    except Exception as e2:
                        print(f"⚠️ CLIP RN50 also failed: {str(e2)[:100]}...")
                        print("🔄 Skipping CLIP - will use other algorithms")
            
            # Try loading with timeout protection
            print("⏳ Initializing CLIP with 30-second timeout...")
            thread = threading.Thread(target=load_clip_with_timeout, daemon=True)
            thread.start()
            
            # Wait for completion or timeout
            start_time = time.time()
            while thread.is_alive() and (time.time() - start_time) < 30:
                time.sleep(0.1)
            
            if thread.is_alive():
                print("⏰ CLIP loading timeout - continuing without CLIP")
                clip_loaded = False
            
            self.available_methods['clip'] = clip_loaded
            
        except ImportError:
            print("⚠️ CLIP not available - install with: pip install clip-by-openai")
            self.available_methods['clip'] = False
        except Exception as e:
            print(f"⚠️ CLIP initialization failed: {str(e)[:100]}...")
            self.available_methods['clip'] = False
    
    def _initialize_lpips_safe(self):
        """Safely initialize LPIPS with timeout protection"""
        try:
            import lpips
            import threading
            import time
            
            lpips_loaded = False
            
            def load_lpips_with_timeout():
                nonlocal lpips_loaded
                try:
                    print("🔧 Loading LPIPS (AlexNet) on CPU...")
                    self.lpips_alex = lpips.LPIPS(net='alex').cpu()
                    lpips_loaded = True
                    print("✅ LPIPS (AlexNet) loaded successfully on CPU")
                except Exception as e:
                    print(f"⚠️ LPIPS failed: {str(e)[:100]}...")
            
            # Try loading with timeout protection
            print("⏳ Initializing LPIPS with 20-second timeout...")
            thread = threading.Thread(target=load_lpips_with_timeout, daemon=True)
            thread.start()
            
            # Wait for completion or timeout
            start_time = time.time()
            while thread.is_alive() and (time.time() - start_time) < 20:
                time.sleep(0.1)
            
            if thread.is_alive():
                print("⏰ LPIPS loading timeout - continuing without LPIPS")
                lpips_loaded = False
                
            self.available_methods['lpips'] = lpips_loaded
                
        except ImportError:
            print("⚠️ LPIPS not available - install with: pip install lpips")
            self.available_methods['lpips'] = False
        except Exception as e:
            print(f"⚠️ LPIPS initialization failed: {str(e)[:100]}...")
            self.available_methods['lpips'] = False
    
    def evaluate_compatibility_enhanced(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced evaluation with compatibility handling
        """
        results = {
            'professional_identity': {},
            'advanced_metrics': {},
            'traditional_metrics': {},
            'final_assessment': {}
        }
        
        print("🔍 Running compatibility-enhanced evaluation...")
        
        # 1. Professional Identity Analysis (if available)
        if self.available_methods.get('professional', False):
            try:
                print("🎯 Running professional identity analysis...")
                identity_result = self.professional_evaluator.calculate_identity_similarity(image1, image2)
                results['professional_identity'] = identity_result
            except Exception as e:
                print(f"⚠️ Professional identity analysis failed: {e}")
                results['professional_identity'] = {'error': str(e)}
        
        # 2. CLIP Analysis (if available)
        if self.available_methods.get('clip', False):
            try:
                print("🎯 Running CLIP analysis...")
                clip_result = self._safe_clip_analysis(image1, image2)
                results['advanced_metrics']['clip'] = clip_result
            except Exception as e:
                print(f"⚠️ CLIP analysis failed: {e}")
                results['advanced_metrics']['clip'] = {'error': str(e)}
        
        # 3. LPIPS Analysis (if available)
        if self.available_methods.get('lpips', False):
            try:
                print("🎯 Running LPIPS analysis...")
                lpips_result = self._safe_lpips_analysis(image1, image2)
                results['advanced_metrics']['lpips'] = lpips_result
            except Exception as e:
                print(f"⚠️ LPIPS analysis failed: {e}")
                results['advanced_metrics']['lpips'] = {'error': str(e)}
        
        # 4. Traditional Metrics (always available)
        print("📊 Running traditional metrics...")
        results['traditional_metrics'] = self._safe_traditional_metrics(image1, image2)
        
        # 5. Final Assessment
        results['final_assessment'] = self._calculate_safe_assessment(results)
        
        return results
    
    def _safe_clip_analysis(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """Safe CLIP analysis with error handling"""
        try:
            import torch
            from PIL import Image
            
            # Convert images (handle both RGB and BGR input)
            if image1.shape[2] == 3:
                # Assume RGB input from PIL/numpy
                img1_pil = Image.fromarray(image1.astype(np.uint8))
                img2_pil = Image.fromarray(image2.astype(np.uint8))
            else:
                img1_pil = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
                img2_pil = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
            
            # Preprocess
            img1_tensor = self.clip_preprocess(img1_pil).unsqueeze(0).to(self.clip_device)
            img2_tensor = self.clip_preprocess(img2_pil).unsqueeze(0).to(self.clip_device)
            
            # Extract features
            with torch.no_grad():
                features1 = self.clip_model.encode_image(img1_tensor)
                features2 = self.clip_model.encode_image(img2_tensor)
                
                print(f"🔍 CLIP Features shape: {features1.shape}, {features2.shape}")
                
                # Normalize
                features1 = features1 / features1.norm(dim=-1, keepdim=True)
                features2 = features2 / features2.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = torch.nn.functional.cosine_similarity(features1, features2, dim=-1)
                similarity_score = float(similarity.item())
                
                print(f"🎯 CLIP Similarity calculated: {similarity_score:.4f}")
            
            return {
                'clip_image_similarity': similarity_score,
                'method': 'CLIP Compatible Analysis',
                'model_used': 'ViT-B/32 or RN50'
            }
            
        except Exception as e:
            return {'error': f'CLIP analysis failed: {e}'}
    
    def _safe_lpips_analysis(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """Safe LPIPS analysis with error handling"""
        try:
            import torch
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Prepare transform
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            # Convert and transform images (handle both RGB and BGR input)
            if image1.shape[2] == 3:
                # Assume RGB input from PIL/numpy  
                img1_pil = Image.fromarray(image1.astype(np.uint8))
                img2_pil = Image.fromarray(image2.astype(np.uint8))
            else:
                img1_pil = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
                img2_pil = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
            
            img1_tensor = transform(img1_pil).unsqueeze(0)
            img2_tensor = transform(img2_pil).unsqueeze(0)
            
            # Calculate LPIPS
            with torch.no_grad():
                lpips_distance = self.lpips_alex(img1_tensor, img2_tensor).item()
            
            # Convert to similarity
            lpips_similarity = max(0, 1.0 - lpips_distance)
            
            print(f"🔍 LPIPS Distance: {lpips_distance:.4f}, Similarity: {lpips_similarity:.4f}")
            
            return {
                'lpips_similarity': lpips_similarity,
                'lpips_distance': lpips_distance,
                'method': 'LPIPS Compatible Analysis'
            }
            
        except Exception as e:
            return {'error': f'LPIPS analysis failed: {e}'}
    
    def _safe_traditional_metrics(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """Safe traditional metrics that should always work"""
        try:
            from skimage.metrics import structural_similarity as ssim
            from skimage.metrics import peak_signal_noise_ratio as psnr
            
            # Resize to same dimensions
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            target_size = (min(h1, h2), min(w1, w2))
            
            img1_resized = cv2.resize(image1, (target_size[1], target_size[0]))
            img2_resized = cv2.resize(image2, (target_size[1], target_size[0]))
            
            # Convert to grayscale for some metrics
            if len(img1_resized.shape) == 3:
                gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
            else:
                gray1, gray2 = img1_resized, img2_resized
            
            # Calculate metrics
            ssim_score = ssim(gray1, gray2, data_range=255)
            psnr_score = psnr(gray1, gray2, data_range=255)
            mse_score = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
            
            # Color histogram correlation
            try:
                hist1 = cv2.calcHist([img1_resized], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
                hist2 = cv2.calcHist([img2_resized], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
                hist_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            except:
                hist_correlation = 0.0
            
            return {
                'ssim': float(ssim_score),
                'psnr': float(psnr_score),
                'mse': float(mse_score),
                'histogram_correlation': float(hist_correlation),
                'method': 'Safe Traditional Metrics'
            }
            
        except Exception as e:
            return {'error': f'Traditional metrics failed: {e}'}
    
    def _calculate_safe_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate assessment from available results"""
        scores = []
        methods_used = []
        
        # Professional identity score
        prof_identity = results.get('professional_identity', {})
        if 'similarity' in prof_identity:
            identity_score = prof_identity['similarity']
            if isinstance(identity_score, (int, float)) and not np.isnan(identity_score):
                scores.append(identity_score)
                methods_used.append('Professional Identity')
        
        # CLIP score
        clip_result = results.get('advanced_metrics', {}).get('clip', {})
        if 'clip_image_similarity' in clip_result:
            clip_score = clip_result['clip_image_similarity']
            if isinstance(clip_score, (int, float)) and not np.isnan(clip_score):
                scores.append(clip_score)
                methods_used.append('CLIP Analysis')
        
        # LPIPS score
        lpips_result = results.get('advanced_metrics', {}).get('lpips', {})
        if 'lpips_similarity' in lpips_result:
            lpips_score = lpips_result['lpips_similarity']
            if isinstance(lpips_score, (int, float)) and not np.isnan(lpips_score):
                scores.append(lpips_score)
                methods_used.append('LPIPS Analysis')
        
        # Traditional metrics score
        traditional = results.get('traditional_metrics', {})
        if 'ssim' in traditional and 'histogram_correlation' in traditional:
            ssim_score = traditional['ssim']
            hist_score = traditional['histogram_correlation']
            if all(isinstance(s, (int, float)) and not np.isnan(s) for s in [ssim_score, hist_score]):
                traditional_score = (ssim_score + hist_score) / 2
                scores.append(traditional_score)
                methods_used.append('Traditional Metrics')
        
        # Calculate final assessment
        if scores:
            final_score = np.mean(scores)
            confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.8
            
            # Determine consistency level
            if final_score >= 0.8:
                consistency_level = "Very High"
                color_indicator = "🟢"
                interpretation = "Excellent consistency - very likely same character/person"
            elif final_score >= 0.6:
                consistency_level = "High"
                color_indicator = "🟡"
                interpretation = "Good consistency - likely same character with variations"
            elif final_score >= 0.4:
                consistency_level = "Moderate"
                color_indicator = "🟠"
                interpretation = "Moderate consistency - some similarities detected"
            elif final_score >= 0.2:
                consistency_level = "Low"
                color_indicator = "🔴"
                interpretation = "Poor consistency - likely different characters"
            else:
                consistency_level = "Very Low"
                color_indicator = "⛔"
                interpretation = "Very poor consistency - clearly different subjects"
            
            return {
                'final_score': float(final_score),
                'consistency_level': consistency_level,
                'color_indicator': color_indicator,
                'interpretation': interpretation,
                'confidence': float(confidence),
                'methods_used': methods_used,
                'individual_scores': scores,
                'method': 'Compatible Multi-Method Assessment'
            }
        else:
            return {
                'final_score': 0.0,
                'consistency_level': "Unknown",
                'color_indicator': "⚪",
                'interpretation': "Unable to calculate - insufficient data",
                'confidence': 0.0,
                'methods_used': [],
                'error': 'No valid evaluation methods produced results'
            }

    def evaluate_character_consistency(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """
        Main character consistency evaluation method - wrapper for compatibility
        This is the method called by the launch script
        """
        try:
            print("🔬 Starting character consistency evaluation...")
            
            # Run the enhanced evaluation
            enhanced_results = self.evaluate_compatibility_enhanced(image1, image2)
            
            # Convert to format expected by the launcher
            final_results = {}
            
            # Professional Identity Results
            if 'professional_identity' in enhanced_results and not enhanced_results['professional_identity'].get('error'):
                prof_data = enhanced_results['professional_identity']
                final_results.update({
                    'Identity_Similarity': prof_data.get('similarity', 0.0),
                    'Identity_Confidence': prof_data.get('confidence', 0.0),
                    'Identity_Decision': prof_data.get('identity_decision', 'Unknown'),
                    'Decision_Confidence': prof_data.get('decision_confidence', 0.0),
                    'Models_Used': prof_data.get('models_used', 0),
                    'Detection_Method': prof_data.get('method', 'Professional Multi-Model')
                })
            
            # CLIP Results
            if 'advanced_metrics' in enhanced_results and 'clip' in enhanced_results['advanced_metrics']:
                clip_data = enhanced_results['advanced_metrics']['clip']
                if not clip_data.get('error'):
                    final_results['CLIP_Similarity'] = clip_data.get('clip_image_similarity', 0.0)
            
            # LPIPS Results
            if 'advanced_metrics' in enhanced_results and 'lpips' in enhanced_results['advanced_metrics']:
                lpips_data = enhanced_results['advanced_metrics']['lpips']
                if not lpips_data.get('error'):
                    final_results['LPIPS_Similarity'] = lpips_data.get('lpips_similarity', 0.0)
            
            # Traditional Metrics
            if 'traditional_metrics' in enhanced_results:
                trad_data = enhanced_results['traditional_metrics']
                final_results.update({
                    'SSIM': trad_data.get('ssim', 0.0),
                    'PSNR': trad_data.get('psnr', 0.0),
                    'MSE': trad_data.get('mse', 0.0),
                    'Histogram_Similarity': trad_data.get('histogram_similarity', 0.0)
                })
            
            # Final Assessment
            if 'final_assessment' in enhanced_results:
                final_data = enhanced_results['final_assessment']
                final_results.update({
                    'Final_Score': final_data.get('final_score', 0.0),
                    'Consistency_Level': final_data.get('consistency_level', 'Unknown'),
                    'Assessment_Color': final_data.get('color_indicator', '⚪'),
                    'Interpretation': final_data.get('interpretation', 'No assessment available'),
                    'Overall_Confidence': final_data.get('confidence', 0.0),
                    'Methods_Used': ', '.join(final_data.get('methods_used', ['Unknown']))
                })
            
            print(f"✅ Character consistency evaluation completed - Final Score: {final_results.get('Final_Score', 0.0):.3f}")
            return final_results
            
        except Exception as e:
            print(f"❌ Character consistency evaluation failed: {e}")
            return {
                'Error': str(e),
                'Identity_Similarity': 0.0,
                'Final_Score': 0.0,
                'Detection_Method': 'Error - Evaluation Failed'
            }

# Test function
def test_compatible_system():
    """Test the compatible evaluation system"""
    print("\n🧪 Testing Compatible Evaluation System")
    print("-" * 50)
    
    system = CompatibleEvaluationSystem()
    
    # Create test images
    print("\nCreating test images...")
    test_img1 = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
    test_img2 = np.random.randint(120, 180, (300, 300, 3), dtype=np.uint8)
    
    print("Running compatible evaluation...")
    results = system.evaluate_compatibility_enhanced(test_img1, test_img2)
    
    print("\n📊 Compatible Evaluation Results:")
    
    # Show final assessment
    final = results.get('final_assessment', {})
    if final and 'final_score' in final:
        print(f"\n{final.get('color_indicator', '⚪')} FINAL ASSESSMENT:")
        print(f"  ├─ Overall Score: {final['final_score']:.3f}")
        print(f"  ├─ Consistency Level: {final['consistency_level']}")
        print(f"  ├─ Confidence: {final['confidence']:.3f}")
        print(f"  ├─ Methods Used: {', '.join(final['methods_used'])}")
        print(f"  └─ Interpretation: {final['interpretation']}")
    
    print(f"\n📈 Method Results:")
    for category, data in results.items():
        if category != 'final_assessment' and isinstance(data, dict):
            print(f"  {category}: {'✅' if not data.get('error') else '❌'}")
    
    print("\n✅ Compatible system test completed")

if __name__ == "__main__":
    test_compatible_system()