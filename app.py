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

def format_segmented_results(results, use_advanced_algorithms, img1_np, img2_np):
    """
    Format evaluation results with clear segmentation and improved readability
    """
    
    if use_advanced_algorithms and 'Final_Score' in results:
        # Advanced algorithms results formatting
        return format_advanced_results(results, img1_np, img2_np)
    else:
        # Fallback mode results formatting
        return format_fallback_results(results, img1_np, img2_np)

def format_advanced_results(results, img1_np, img2_np):
    """Format results for advanced algorithm mode with clear segmentation - English version"""
    
    final_score = results.get('Final_Score', 0.0)
    consistency_level = results.get('Consistency_Level', 'Unknown')
    color_indicator = results.get('Assessment_Color', '⚪')
    interpretation = results.get('Interpretation', 'No assessment available')
    methods_used = results.get('Methods_Used', 'Unknown')
    
    # Start with main header
    result_text = "\n" + "="*80 + "\n"
    result_text += "🚀 Professional Character Consistency Analysis Report\n"
    result_text += "="*80 + "\n\n"
    
    # Section 1: Executive Summary
    result_text += "📋 Section 1: Executive Summary\n"
    result_text += "─"*40 + "\n"
    result_text += f"{color_indicator} Overall Assessment: {consistency_level}\n"
    result_text += f"📊 Consistency Score: {final_score:.3f} / 1.000\n"
    result_text += f"🎯 Confidence Level: {results.get('Overall_Confidence', 0.0):.3f}\n"
    result_text += f"🔬 Analysis Methods: {methods_used}\n"
    result_text += f"💬 Professional Interpretation: {interpretation}\n\n"
    
    # Section 2: Score Reference Guide
    result_text += "📖 Section 2: Score Reference Guide\n"
    result_text += "─"*40 + "\n"
    result_text += "🟢 0.8-1.0: Excellent consistency\n"
    result_text += "🟡 0.6-0.8: Good consistency\n"
    result_text += "🟠 0.4-0.6: Moderate consistency\n"
    result_text += "🔴 0.2-0.4: Low consistency\n"
    result_text += "⚫ 0.0-0.2: Very low consistency\n\n"
    
    # Section 3: Advanced Algorithm Analysis
    result_text += "🔬 Section 3: Advanced Algorithm Analysis\n"
    result_text += "─"*45 + "\n"
    
    # Identity Analysis
    if 'Identity_Similarity' in results:
        identity_score = results['Identity_Similarity']
        models_used = results.get('Models_Used', 0)
        identity_decision = results.get('Identity_Decision', 'Unknown')
        
        # Color coding for identity score
        if identity_score >= 0.6:
            identity_icon = "✅"
        elif identity_score >= 0.4:
            identity_icon = "⚠️"
        else:
            identity_icon = "❌"
            
        result_text += f"👤 Face Identity Recognition\n"
        result_text += "   " + "━"*30 + "\n"
        result_text += f"   {identity_icon} Identity Similarity: {identity_score:.4f}\n"
        result_text += f"   🧠 Models Used: {models_used} deep learning models\n"
        result_text += f"   🎯 Identity Decision: {identity_decision}\n"
        result_text += f"   📈 Algorithm: Multi-model Consensus Voting\n\n"
    
    # CLIP Analysis
    if 'CLIP_Similarity' in results:
        clip_score = results['CLIP_Similarity']
        
        # Color coding for CLIP score
        if clip_score >= 0.7:
            clip_icon = "🟢"
        elif clip_score >= 0.5:
            clip_icon = "🟡"
        else:
            clip_icon = "🔴"
            
        result_text += f"🎯 CLIP Vision-Language Analysis\n"
        result_text += "   " + "━"*35 + "\n"
        result_text += f"   {clip_icon} Semantic Similarity: {clip_score:.4f}\n"
        result_text += f"   🧠 Model Architecture: Vision Transformer (ViT-B/32)\n"
        result_text += f"   📚 Analysis Level: Semantic-level understanding\n"
        result_text += f"   💡 Feature: Cross-modal vision-language understanding\n\n"
    
    # Perceptual Analysis
    if 'LPIPS_Similarity' in results:
        lpips_score = results['LPIPS_Similarity']
        
        # Color coding for LPIPS score
        if lpips_score >= 0.6:
            lpips_icon = "👁️"
        elif lpips_score >= 0.4:
            lpips_icon = "👀"
        else:
            lpips_icon = "😵"
            
        result_text += f"👁️ LPIPS Learned Perceptual Similarity\n"
        result_text += "   " + "━"*40 + "\n"
        result_text += f"   {lpips_icon} Perceptual Similarity: {lpips_score:.4f}\n"
        result_text += f"   🧠 Neural Network: AlexNet Perceptual Model\n"
        result_text += f"   👥 Evaluation Standard: Human visual perception aligned\n"
        result_text += f"   🎯 Advantage: More accurate than traditional SSIM\n\n"
    
    # Section 4: Traditional Image Quality Metrics
    result_text += "📊 Section 4: Traditional Image Quality Metrics\n"
    result_text += "─"*50 + "\n"
    
    if 'SSIM' in results:
        ssim_score = results['SSIM']
        ssim_icon = "🟢" if ssim_score >= 0.7 else "🟡" if ssim_score >= 0.4 else "🔴"
        result_text += f"{ssim_icon} Structural Similarity Index (SSIM): {ssim_score:.4f}\n"
        result_text += f"     └─ Evaluates: Luminance, contrast, structural similarity\n"
        
    if 'PSNR' in results:
        psnr_score = results['PSNR']
        psnr_icon = "📶" if psnr_score >= 25 else "📵" if psnr_score >= 15 else "📴"
        result_text += f"{psnr_icon} Peak Signal-to-Noise Ratio (PSNR): {psnr_score:.2f} dB\n"
        result_text += f"     └─ Evaluates: Image signal quality (higher is better)\n"
        
    if 'MSE' in results:
        mse_score = results['MSE']
        mse_icon = "✅" if mse_score <= 100 else "⚠️" if mse_score <= 1000 else "❌"
        result_text += f"{mse_icon} Mean Squared Error (MSE): {mse_score:.2f}\n"
        result_text += f"     └─ Evaluates: Pixel-level differences (lower is better)\n"
        
    if 'Histogram_Similarity' in results:
        hist_score = results['Histogram_Similarity']
        hist_icon = "🎨" if hist_score >= 0.5 else "🎭" if hist_score >= 0.2 else "🖤"
        result_text += f"{hist_icon} Color Histogram Correlation: {hist_score:.4f}\n"
        result_text += f"     └─ Evaluates: Color distribution similarity\n"
    
    result_text += "\n"
    
    # Section 5: Professional Recommendations
    result_text += "💡 Section 5: Professional Recommendations\n"
    result_text += "─"*45 + "\n"
    
    if final_score >= 0.8:
        result_text += "✅ Excellent character consistency - safe to use\n"
        result_text += "✅ High algorithm consensus - very reliable results\n"
        result_text += "✅ Recommended for high-consistency requirement applications\n"
    elif final_score >= 0.6:
        result_text += "🟡 Good character consistency - minor variations acceptable\n"
        result_text += "🟡 Manual review recommended for specific requirements\n"
        result_text += "🟡 Suitable for most standard application scenarios\n"
    elif final_score >= 0.4:
        result_text += "⚠️ Moderate character consistency - some differences exist\n"
        result_text += "⚠️ Strong manual review recommended, focus on key features\n"
        result_text += "⚠️ Use with caution, consider further optimization\n"
    elif final_score >= 0.2:
        result_text += "🔴 Poor character consistency - likely different characters\n"
        result_text += "🔴 Recommend reconsidering or using alternative images\n"
        result_text += "🔴 Not recommended for strict applications\n"
    else:
        result_text += "❌ Very poor character consistency - clearly different characters\n"
        result_text += "❌ Not recommended for consistency-required applications\n"
        result_text += "❌ Strongly recommend using different image pairs\n"
    
    result_text += "\n"
    
    # Section 6: Technical Information
    result_text += "⚙️ Section 6: Technical Information\n"
    result_text += "─"*40 + "\n"
    result_text += f"📸 Image Dimensions: {img1_np.shape[1]} × {img1_np.shape[0]} pixels\n"
    result_text += f"🔧 Processing Mode: Enhanced Multi-Algorithm System\n"
    result_text += f"🧮 Detection Method: {results.get('Detection_Method', 'Unknown')}\n"
    result_text += f"💻 Hardware Optimization: CPU + Integrated Graphics Adapted\n"
    result_text += f"📊 Analysis Time: {img1_np.shape[0] * img1_np.shape[1] / 100000:.1f}s (estimated)\n"
    result_text += f"🎯 Algorithm Version: Advanced Character Consistency v2.0\n\n"
    
    result_text += "="*80 + "\n"
    result_text += "📄 Report Generation Complete\n"
    result_text += "="*80 + "\n"
    
    return result_text

def format_fallback_results(results, img1_np, img2_np):
    """Format results for fallback mode with clear segmentation - English version"""
    
    result_text = "\n" + "="*75 + "\n"
    result_text += "📊 Professional Image Evaluation Results (Basic Compatibility Mode)\n"
    result_text += "="*75 + "\n\n"
    
    # Quick Assessment
    if 'Identity_Similarity' in results:
        identity_score = results['Identity_Similarity']
        if identity_score >= 0.6:
            assessment_color = "🟢"
            assessment_text = "High Similarity"
        elif identity_score >= 0.4:
            assessment_color = "🟡"  
            assessment_text = "Moderate Similarity"
        else:
            assessment_color = "🔴"
            assessment_text = "Low Similarity"
            
        result_text += f"📋 Quick Assessment\n"
        result_text += f"{assessment_color} {assessment_text} (Score: {identity_score:.3f})\n\n"
    
    # Image Information
    result_text += "📸 Image Information\n"
    result_text += "─"*25 + "\n"
    result_text += f"📏 Original Image Size: {img1_np.shape[1]} × {img1_np.shape[0]} pixels\n"
    result_text += f"📏 Comparison Image Size: {img2_np.shape[1]} × {img2_np.shape[0]} pixels\n"
    result_text += f"⚙️ Processing Mode: Basic Fallback\n"
    result_text += f"💡 Tip: Install advanced algorithm packages for detailed analysis\n\n"
    
    # Traditional Image Quality Metrics
    result_text += "🔍 Traditional Image Quality Metrics\n"
    result_text += "─"*40 + "\n"
    
    if 'SSIM' in results:
        result_text += f"├─ Structural Similarity (SSIM): {results['SSIM']:.4f}\n"
        result_text += f"   └─ Range: 0.0 (completely different) - 1.0 (identical)\n\n"
        
    if 'PSNR' in results:
        result_text += f"├─ Peak Signal-to-Noise Ratio (PSNR): {results['PSNR']:.2f} dB\n"
        result_text += f"   └─ Higher values indicate better quality\n\n"
        
    if 'MSE' in results:
        result_text += f"├─ Mean Squared Error (MSE): {results['MSE']:.2f}\n"
        result_text += f"   └─ Lower values indicate higher similarity\n\n"
    
    # Professional Identity Analysis (fallback mode)
    identity_metrics = ['Identity_Similarity', 'Identity_Confidence', 'Identity_Decision', 
                      'Detection_Method']
    result_text += "🎯 Professional Identity Analysis\n"
    result_text += "─"*40 + "\n"
    
    for key in identity_metrics:
        if key in results:
            if key == 'Identity_Similarity':
                result_text += f"├─ Identity Similarity Score: {results[key]:.4f}\n"
                result_text += f"   └─ Range: 0.0 (different people) - 1.0 (same person)\n\n"
            elif key == 'Identity_Confidence':
                result_text += f"├─ Analysis Confidence: {results[key]:.4f}\n"
                result_text += f"   └─ Model consistency level (higher is better)\n\n"
            elif key == 'Identity_Decision':
                decision_icon = "✅" if "Same" in str(results[key]) else "❌"
                result_text += f"├─ Final Decision: {decision_icon} {results[key]}\n\n"
            elif key == 'Detection_Method':
                result_text += f"└─ Analysis Method: {results[key]}\n\n"
    
    # Basic Recommendations
    result_text += "💡 Basic Recommendations\n"
    result_text += "─"*30 + "\n"
    
    if 'Identity_Similarity' in results:
        score = results['Identity_Similarity']
        if score >= 0.6:
            result_text += "✅ High similarity - can consider using\n"
        elif score >= 0.4:
            result_text += "🟡 Moderate similarity - manual review recommended\n"
        else:
            result_text += "🔴 Low similarity - use with caution\n"
    
    result_text += "📈 Recommend upgrading to advanced algorithm mode for more accurate results\n\n"
    
    # Technical Info
    result_text += "⚙️ Technical Information\n"
    result_text += "─"*30 + "\n"
    result_text += f"🔧 System Version: Basic Evaluation System v1.0\n"
    result_text += f"💻 Compatibility Mode: CPU-optimized processing\n"
    result_text += f"📊 Processing Time: {img1_np.shape[0] * img1_np.shape[1] / 200000:.1f}s (estimated)\n\n"
    
    result_text += "="*75 + "\n"
    result_text += "📄 Basic Assessment Complete\n"
    result_text += "="*75 + "\n"
    
    return result_text

def main():
    try:
        import gradio as gr
        import numpy as np
        from PIL import Image
        import cv2
        
        print("🔧 Initializing Professional Image Evaluation System...")
        print("⚙️ CPU and Integrated Graphics Optimization Mode")
        
        # Check if advanced evaluation system is available
        try:
            from evaluator import CompatibleEvaluationSystem
            evaluator = CompatibleEvaluationSystem()
            use_advanced_algorithms = True
            print("✅ Advanced algorithms loaded successfully")
            print("🚀 Multi-model consensus analysis enabled")
        except ImportError as e:
            print(f"⚠️ Advanced algorithms not available: {e}")
            print("📱 Falling back to basic evaluation system...")
            use_advanced_algorithms = False
            
            # Basic fallback evaluator
            class BasicEvaluator:
                def evaluate_character_consistency(self, img1, img2):
                    try:
                        # Convert images to grayscale for basic analysis
                        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                        
                        # Basic SSIM calculation
                        from skimage.metrics import structural_similarity
                        ssim_score = structural_similarity(gray1, gray2)
                        
                        # Basic MSE calculation
                        mse = np.mean((gray1 - gray2) ** 2)
                        
                        # Basic PSNR calculation
                        if mse > 0:
                            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                        else:
                            psnr = float('inf')
                        
                        return {
                            'SSIM': ssim_score,
                            'MSE': mse,
                            'PSNR': psnr,
                            'Identity_Similarity': ssim_score,  # Use SSIM as basic similarity
                            'Identity_Decision': 'Same Person' if ssim_score > 0.5 else 'Different Person',
                            'Detection_Method': 'Basic SSIM Analysis'
                        }
                    except Exception as e:
                        return {'Error': str(e)}
            
            evaluator = BasicEvaluator()
        
        def evaluate_images(image1, image2):
            """Main evaluation function with enhanced formatting"""
            try:
                if image1 is None or image2 is None:
                    return "❌ 请上传两张图片进行比较 | Please upload both images for comparison"
                
                # Convert PIL images to numpy arrays
                img1_np = np.array(image1)
                img2_np = np.array(image2)
                
                # Ensure images are in RGB format
                if len(img1_np.shape) == 3 and img1_np.shape[2] == 4:  # RGBA to RGB
                    img1_np = img1_np[:, :, :3]
                if len(img2_np.shape) == 3 and img2_np.shape[2] == 4:  # RGBA to RGB
                    img2_np = img2_np[:, :, :3]
                
                print(f"📊 Processing images: {img1_np.shape} vs {img2_np.shape}")
                
                # Resize images to manageable size for CPU processing
                max_size = 512
                if max(img1_np.shape[:2]) > max_size:
                    scale = max_size / max(img1_np.shape[:2])
                    new_height = int(img1_np.shape[0] * scale)
                    new_width = int(img1_np.shape[1] * scale)
                    img1_np = cv2.resize(img1_np, (new_width, new_height))
                
                if max(img2_np.shape[:2]) > max_size:
                    scale = max_size / max(img2_np.shape[:2])
                    new_height = int(img2_np.shape[0] * scale)
                    new_width = int(img2_np.shape[1] * scale)
                    img2_np = cv2.resize(img2_np, (new_width, new_height))
                
                # Run evaluation
                print("🔬 Running character consistency analysis...")
                results = evaluator.evaluate_character_consistency(img1_np, img2_np)
                
                # Add basic histogram analysis if not present
                if 'Histogram_Similarity' not in results:
                    try:
                        hist1 = cv2.calcHist([img1_np], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                        hist2 = cv2.calcHist([img2_np], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                        results['Histogram_Similarity'] = hist_similarity
                    except Exception as e:
                        results['Histogram_Error'] = str(e)
                
                # Format results with clear segmented display
                result_text = format_segmented_results(results, use_advanced_algorithms, img1_np, img2_np)
                
                return result_text
                
            except Exception as e:
                return f"❌ Error during evaluation: {str(e)}\n\nPlease check:\n- Both images are uploaded correctly\n- Images are in supported format (JPG, PNG)\n- System dependencies are installed\n\nFor troubleshooting, run: python test_evaluation.py"
        
        # Create Gradio interface
        with gr.Blocks(title="Professional Image Evaluator", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# 🎯 Professional Image Evaluation System")
            gr.Markdown("Optimized for integrated graphics and CPU processing")
            
            with gr.Row():
                with gr.Column():
                    image1_input = gr.Image(type="pil", label="Original Image")
                    
                with gr.Column():
                    image2_input = gr.Image(type="pil", label="Generated/Comparison Image")
            
            evaluate_button = gr.Button("🔬 Analyze Character Consistency", variant="primary", size="lg")
            
            with gr.Row():
                output_text = gr.Textbox(
                    label="Analysis Results", 
                    lines=30,
                    max_lines=50,
                    show_copy_button=True
                )
            
            gr.Markdown("""
            ### 📖 Usage Instructions:
            1. Upload the **original character image** on the left
            2. Upload the **generated/comparison image** on the right  
            3. Click **"Analyze Character Consistency"** to start evaluation
            4. Review the detailed **segmented analysis report**
            
            ### 🔬 Analysis Features:
            - **Multi-model face recognition consensus** (when available)
            - **CLIP semantic similarity analysis** (Vision Transformer)
            - **LPIPS perceptual similarity** (Human-like perception)
            - **Enhanced traditional metrics** (SSIM, PSNR, MSE)
            - **Professional recommendations** with confidence levels
            - **Comprehensive technical information**
            
            ### ⚙️ System Optimization:
            - CPU-optimized processing for integrated graphics
            - Automatic fallback to compatible algorithms
            - Memory-efficient image preprocessing
            - Professional report formatting with clear segmentation
            """)
            
            evaluate_button.click(
                evaluate_images,
                inputs=[image1_input, image2_input],
                outputs=output_text
            )
        
        print("🚀 Starting web interface on http://127.0.0.1:7862")
        print("📊 Ready for professional image evaluation!")
        interface.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if all required packages are installed")
        print("2. Run: pip install gradio pillow opencv-python scikit-image numpy")
        print("3. Ensure compatible_evaluation_system.py is available")
        print("4. Try running: python test_evaluation.py")
        sys.exit(1)

if __name__ == "__main__":
    main()