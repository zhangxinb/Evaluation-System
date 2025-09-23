#!/usr/bin/env python3
"""
Professional Image Evaluation System Launcher
Optimized for integrated graphics and CPU processing
"""

import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_integrated_graphics_environment():
    """Configure environment for integrated graphics optimal performance"""
    
    # Force CPU mode for PyTorch (most stable for integrated graphics)
    os.environ['TORCH_USE_CUDA'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Optimize CPU performance
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    
    print("🚀 Professional Image Evaluation System Ready")
    print("✅ CPU-optimized mode active")

def start_gradio_interface():
    """Start the Gradio web interface directly"""
    
    try:
        import gradio as gr
        import torch
        from PIL import Image
        import numpy as np
        
        # Configure PyTorch for integrated graphics
        torch.set_num_threads(8)
        device = torch.device('cpu')
        
        print(f"🔧 PyTorch configured for CPU: {device}")
        
        # Initialize Professional Identity Evaluator (DeepFace)
        professional_identity_eval = None
        try:
            from professional_identity_evaluator import ProfessionalIdentityEvaluator
            professional_identity_eval = ProfessionalIdentityEvaluator()
            print("✅ Professional Identity Evaluator (DeepFace) loaded successfully")
        except ImportError as e:
            print(f"⚠️ Professional Identity evaluator not available: {e}")
        
        # Initialize Basic Identity Evaluator as fallback
        identity_eval = None
        try:
            from amd_identity_evaluator import AMDIdentityEvaluator
            identity_eval = AMDIdentityEvaluator()
            print("✅ Basic Identity Evaluator loaded as fallback")
        except ImportError as e:
            print(f"⚠️ Basic Identity evaluator not available: {e}")
        
        # Try to import other evaluators (optional)
        try:
            from core.clip_evaluator import CLIPEvaluator
            from core.lpips_evaluator import LPIPSEvaluator
            from core.traditional_evaluator import TraditionalEvaluator
            
            # Initialize evaluators
            clip_eval = CLIPEvaluator(device=device)
            lpips_eval = LPIPSEvaluator(device=device)
            traditional_eval = TraditionalEvaluator()
            
            print("✅ Additional evaluators loaded")
            
        except ImportError as e:
            print(f"⚠️ Some evaluators not available: {e}")
            print("Using basic evaluation mode")
        
        # Create simple evaluation function
        def evaluate_images(image1, image2):
            """Simple evaluation function for integrated graphics systems"""
            
            if image1 is None or image2 is None:
                return "Please upload both images"
            
            try:
                # Convert PIL to numpy
                img1_np = np.array(image1)
                img2_np = np.array(image2)
                
                results = {}
                
                # Basic similarity metrics using traditional evaluator
                try:
                    import cv2
                    from skimage.metrics import structural_similarity as ssim
                    from skimage.metrics import peak_signal_noise_ratio as psnr
                    
                    # Convert to grayscale for SSIM
                    if len(img1_np.shape) == 3:
                        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
                        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
                    else:
                        gray1, gray2 = img1_np, img2_np
                    
                    # Resize images to same size
                    h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
                    gray1_resized = cv2.resize(gray1, (w, h))
                    gray2_resized = cv2.resize(gray2, (w, h))
                    
                    # Calculate metrics
                    ssim_score = ssim(gray1_resized, gray2_resized, data_range=255)
                    psnr_score = psnr(gray1_resized, gray2_resized, data_range=255)
                    mse_score = np.mean((gray1_resized - gray2_resized) ** 2)
                    
                    results['SSIM'] = ssim_score
                    results['PSNR'] = psnr_score
                    results['MSE'] = mse_score
                    
                except Exception as e:
                    results['Traditional_Metrics_Error'] = str(e)
                
                # Professional Identity evaluation using DeepFace
                try:
                    if professional_identity_eval is not None and professional_identity_eval.available:
                        print("🔍 Using Professional DeepFace evaluation...")
                        identity_results = professional_identity_eval.calculate_identity_similarity(img1_np, img2_np)
                        
                        results['Identity_Similarity'] = identity_results.get('similarity', 0.0)
                        results['Identity_Confidence'] = identity_results.get('confidence', 0.0)
                        results['Identity_Decision'] = identity_results.get('identity_decision', 'Unknown')
                        results['Decision_Confidence'] = identity_results.get('decision_confidence', 0.0)
                        results['Models_Used'] = identity_results.get('models_used', 0)
                        results['Detection_Method'] = identity_results.get('method', 'DeepFace')
                        
                        # Add demographic analysis for first image
                        try:
                            demographics = professional_identity_eval.analyze_face_demographics(img1_np)
                            if 'error' not in demographics:
                                results['Age_Estimate'] = demographics.get('age', 'Unknown')
                                results['Gender_Estimate'] = demographics.get('gender', 'Unknown')
                                results['Emotion_Detected'] = demographics.get('emotion', 'Unknown')
                        except:
                            pass
                        
                    elif identity_eval is not None:
                        print("🔍 Using basic fallback evaluation...")
                        identity_results = identity_eval.calculate_identity_similarity(img1_np, img2_np)
                        results['Identity_Similarity'] = identity_results['similarity']
                        results['Identity_Confidence'] = identity_results['confidence']
                        results['Faces_Detected'] = identity_results['faces_detected']
                        results['Detection_Method'] = identity_results['method']
                    else:
                        results['Identity_Note'] = "No identity evaluator available"
                        
                except Exception as e:
                    results['Identity_Error'] = str(e)
                    print(f"⚠️ Identity evaluation error: {e}")
                
                # Additional basic metrics
                try:
                    # Color histogram similarity
                    hist1 = cv2.calcHist([img1_np], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist2 = cv2.calcHist([img2_np], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    results['Histogram_Similarity'] = hist_similarity
                except Exception as e:
                    results['Histogram_Error'] = str(e)
                
                # Format results with improved readability
                result_text = "📊 Professional Image Evaluation Results\n"
                result_text += "=" * 60 + "\n\n"
                
                # Image Information
                result_text += "📸 IMAGE INFORMATION\n"
                result_text += "-" * 30 + "\n"
                result_text += f"Original Image Size: {img1_np.shape[1]} × {img1_np.shape[0]} pixels\n"
                result_text += f"Generated Image Size: {img2_np.shape[1]} × {img2_np.shape[0]} pixels\n"
                result_text += f"Processing Mode: CPU Optimized\n\n"
                
                # Traditional Image Quality Metrics
                traditional_metrics = ['SSIM', 'PSNR', 'MSE']
                result_text += "🔍 TRADITIONAL IMAGE QUALITY METRICS\n"
                result_text += "-" * 40 + "\n"
                for key in traditional_metrics:
                    if key in results:
                        if key == 'SSIM':
                            result_text += f"├─ Structural Similarity (SSIM): {results[key]:.4f}\n"
                            result_text += f"   └─ Range: 0.0 (different) - 1.0 (identical)\n\n"
                        elif key == 'PSNR':
                            result_text += f"├─ Peak Signal-to-Noise Ratio (PSNR): {results[key]:.2f} dB\n"
                            result_text += f"   └─ Higher values indicate better quality\n\n"
                        elif key == 'MSE':
                            result_text += f"├─ Mean Squared Error (MSE): {results[key]:.2f}\n"
                            result_text += f"   └─ Lower values indicate better similarity\n\n"
                
                # Professional Identity Analysis
                identity_metrics = ['Identity_Similarity', 'Identity_Confidence', 'Identity_Decision', 
                                  'Decision_Confidence', 'Models_Used', 'Detection_Method']
                result_text += "🎯 PROFESSIONAL IDENTITY ANALYSIS\n"
                result_text += "-" * 40 + "\n"
                for key in identity_metrics:
                    if key in results:
                        if key == 'Identity_Similarity':
                            result_text += f"├─ Identity Similarity Score: {results[key]:.4f}\n"
                            result_text += f"   └─ Range: 0.0 (different people) - 1.0 (same person)\n\n"
                        elif key == 'Identity_Confidence':
                            result_text += f"├─ Analysis Confidence: {results[key]:.4f}\n"
                            result_text += f"   └─ Model agreement level (higher is better)\n\n"
                        elif key == 'Identity_Decision':
                            decision_icon = "✅" if "Same" in str(results[key]) else "❌"
                            result_text += f"├─ Final Decision: {decision_icon} {results[key]}\n\n"
                        elif key == 'Decision_Confidence':
                            result_text += f"├─ Decision Confidence: {results[key]:.4f}\n"
                            result_text += f"   └─ How certain the system is about the decision\n\n"
                        elif key == 'Models_Used':
                            result_text += f"├─ Deep Learning Models Used: {results[key]}\n"
                            result_text += f"   └─ Multiple models for robustness\n\n"
                        elif key == 'Detection_Method':
                            result_text += f"└─ Analysis Method: {results[key]}\n\n"
                
                # Face Demographics Analysis
                demographic_metrics = ['Age_Estimate', 'Gender_Estimate', 'Emotion_Detected']
                demographic_found = any(key in results for key in demographic_metrics)
                if demographic_found:
                    result_text += "👤 FACE DEMOGRAPHICS ANALYSIS (Original Image)\n"
                    result_text += "-" * 50 + "\n"
                    for key in demographic_metrics:
                        if key in results:
                            if key == 'Age_Estimate':
                                result_text += f"├─ Estimated Age: {results[key]} years old\n\n"
                            elif key == 'Gender_Estimate':
                                gender_icon = "👨" if "man" in str(results[key]).lower() else "👩"
                                result_text += f"├─ Gender Detection: {gender_icon} {results[key]}\n\n"
                            elif key == 'Emotion_Detected':
                                emotion_icons = {
                                    'happy': '😊', 'sad': '😢', 'angry': '😠', 
                                    'neutral': '😐', 'surprise': '😲', 'fear': '😨'
                                }
                                emotion_icon = emotion_icons.get(str(results[key]).lower(), '😐')
                                result_text += f"└─ Detected Emotion: {emotion_icon} {results[key].title()}\n\n"
                
                # Additional Color Analysis
                additional_metrics = ['Histogram_Similarity']
                result_text += "🎨 COLOR & STYLE ANALYSIS\n"
                result_text += "-" * 30 + "\n"
                for key in additional_metrics:
                    if key in results:
                        if key == 'Histogram_Similarity':
                            result_text += f"├─ Color Distribution Similarity: {results[key]:.4f}\n"
                            result_text += f"   └─ How similar the color palettes are\n\n"
                
                # Error reporting
                error_keys = [k for k in results.keys() if 'Error' in k]
                if error_keys:
                    result_text += "⚠️ ANALYSIS WARNINGS\n"
                    result_text += "-" * 25 + "\n"
                    for key in error_keys:
                        result_text += f"└─ {key}: {results[key]}\n"
                    result_text += "\n"
                
                # Technical Summary
                result_text += "💻 TECHNICAL SUMMARY\n"
                result_text += "-" * 25 + "\n"
                result_text += f"├─ Processing Platform: CPU Mode (Optimized)\n"
                result_text += f"├─ Original Image: {img1_np.shape[1]} × {img1_np.shape[0]} × {img1_np.shape[2]}\n"
                result_text += f"├─ Generated Image: {img2_np.shape[1]} × {img2_np.shape[0]} × {img2_np.shape[2]}\n"
                result_text += f"└─ Analysis Time: ~3-5 seconds per image pair\n\n"
                
                # Interpretation Guide
                result_text += "📋 INTERPRETATION GUIDE\n"
                result_text += "-" * 30 + "\n"
                result_text += "🔍 Identity Similarity Thresholds:\n"
                result_text += "  • < 0.4: Clearly different people\n"
                result_text += "  • 0.4-0.6: Possibly different people\n" 
                result_text += "  • 0.6-0.8: Possibly same person\n"
                result_text += "  • > 0.8: Very likely same person\n\n"
                result_text += "📊 Image Quality Benchmarks:\n"
                result_text += "  • SSIM > 0.8: Excellent similarity\n"
                result_text += "  • SSIM 0.6-0.8: Good similarity\n"
                result_text += "  • SSIM < 0.6: Poor similarity\n"
                result_text += "  • PSNR > 30: High quality\n"
                result_text += "  • PSNR 20-30: Acceptable quality\n"
                result_text += "  • PSNR < 20: Poor quality\n"
                
                return result_text
                
            except Exception as e:
                return f"❌ Error during evaluation: {str(e)}\\n\\nPlease check:\\n- Both images are uploaded correctly\\n- Images are in supported format (JPG, PNG)\\n- System dependencies are installed\\n\\nFor troubleshooting, run: python test_evaluation.py"
        
        # Create Gradio interface
        with gr.Blocks(title="Professional Image Evaluator", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# � Professional Image Evaluation System")
            gr.Markdown("Optimized for integrated graphics and CPU processing")
            
            with gr.Row():
                with gr.Column():
                    image1 = gr.Image(type="pil", label="Original Image")
                    image2 = gr.Image(type="pil", label="Generated Image")
                    
                with gr.Column():
                    results = gr.Textbox(
                        label="Evaluation Results", 
                        lines=15,
                        value="Upload two images to start evaluation..."
                    )
            
            with gr.Row():
                evaluate_btn = gr.Button("🔍 Evaluate Images", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            # Event handlers
            evaluate_btn.click(
                fn=evaluate_images,
                inputs=[image1, image2],
                outputs=results
            )
            
            clear_btn.click(
                fn=lambda: (None, None, "Upload two images to start evaluation..."),
                inputs=[],
                outputs=[image1, image2, results]
            )
            
            gr.Markdown("### ⚡ CPU Performance Optimizations")
            gr.Markdown("""
            - **CPU-only processing** for maximum compatibility
            - **Reduced memory usage** for integrated graphics
            - **Optimized threading** for multi-core systems
            - **Fast evaluation** with essential metrics
            """)
        
        # Launch interface
        print("\\n🌐 Starting web interface...")
        print("🔗 Access URL will be displayed below")
        print("=" * 50)
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7861,  # Use different port to avoid conflicts
            share=False,
            inbrowser=True,
            show_api=False
        )
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install gradio")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting interface: {e}")
        sys.exit(1)

def main():
    """Main launcher for integrated graphics systems"""
    
    print("🚀 Professional Image Evaluator - CPU Optimized Mode")
    print("=" * 50)
    
    # Setup integrated graphics environment
    setup_integrated_graphics_environment()
    
    # Start the interface
    start_gradio_interface()

if __name__ == "__main__":
    main()