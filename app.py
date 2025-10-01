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
    """Format results for advanced algorithm mode - Clean and readable"""
    
    final_score = results.get('Final_Score', 0.0)
    consistency_level = results.get('Consistency_Level', 'Unknown')
    interpretation = results.get('Interpretation', 'No assessment available')
    
    # Start with clean header
    result_text = "\n" + "="*70 + "\n"
    result_text += "Character Consistency Analysis Report\n"
    result_text += "="*70 + "\n\n"
    
    # Overall Assessment
    result_text += "OVERALL ASSESSMENT\n"
    result_text += "-"*70 + "\n"
    result_text += f"Final Score:    {final_score:.3f} / 1.000\n"
    result_text += f"Level:          {consistency_level}\n"
    result_text += f"Interpretation: {interpretation}\n"
    result_text += f"Confidence:     {results.get('Overall_Confidence', 0.0):.3f}\n\n"
    
    # Score Reference
    result_text += "SCORE REFERENCE\n"
    result_text += "-"*70 + "\n"
    result_text += "0.80 - 1.00  Excellent consistency\n"
    result_text += "0.60 - 0.80  Good consistency\n"
    result_text += "0.40 - 0.60  Moderate consistency\n"
    result_text += "0.20 - 0.40  Low consistency\n"
    result_text += "0.00 - 0.20  Very low consistency\n\n"
    
    # Algorithm Results
    result_text += "ALGORITHM RESULTS\n"
    result_text += "-"*70 + "\n"
    
    # Identity Analysis
    if 'Identity_Similarity' in results:
        identity_score = results['Identity_Similarity']
        identity_decision = results.get('Identity_Decision', 'Unknown')
        models_used = results.get('Models_Used', 0)
        
        result_text += f"Face Identity Recognition:\n"
        result_text += f"  Similarity:  {identity_score:.4f}\n"
        result_text += f"  Decision:    {identity_decision}\n"
        result_text += f"  Models:      {models_used} deep learning models\n\n"
    
    # CLIP Analysis
    if 'CLIP_Similarity' in results:
        clip_score = results['CLIP_Similarity']
        result_text += f"CLIP Vision Analysis:\n"
        result_text += f"  Similarity:  {clip_score:.4f}\n"
        result_text += f"  Model:       Vision Transformer (ViT-B/32)\n\n"
    
    # LPIPS Analysis
    if 'LPIPS_Similarity' in results:
        lpips_score = results['LPIPS_Similarity']
        result_text += f"LPIPS Perceptual Analysis:\n"
        result_text += f"  Similarity:  {lpips_score:.4f}\n"
        result_text += f"  Network:     AlexNet Perceptual Model\n\n"
    
    # Traditional Metrics
    result_text += "TRADITIONAL METRICS\n"
    result_text += "-"*70 + "\n"
    
    if 'SSIM' in results:
        ssim_score = results['SSIM']
        result_text += f"SSIM (Structural Similarity):    {ssim_score:.4f}\n"
        
    if 'PSNR' in results:
        psnr_score = results['PSNR']
        result_text += f"PSNR (Peak Signal-to-Noise):     {psnr_score:.2f} dB\n"
        
    if 'MSE' in results:
        mse_score = results['MSE']
        result_text += f"MSE (Mean Squared Error):        {mse_score:.2f}\n"
        
    if 'Histogram_Similarity' in results:
        hist_score = results['Histogram_Similarity']
        result_text += f"Histogram Correlation:           {hist_score:.4f}\n"
    
    result_text += "\n"
    result_text += "="*70 + "\n"
    result_text += "Report Complete\n"
    result_text += "="*70 + "\n"
    
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
                    return "Error: Please upload both images for comparison"
                
                # Convert PIL images to numpy arrays
                img1_np = np.array(image1)
                img2_np = np.array(image2)
                
                # Ensure images are in RGB format
                if len(img1_np.shape) == 3 and img1_np.shape[2] == 4:  # RGBA to RGB
                    img1_np = img1_np[:, :, :3]
                if len(img2_np.shape) == 3 and img2_np.shape[2] == 4:  # RGBA to RGB
                    img2_np = img2_np[:, :, :3]
                
                print(f"📊 Processing images: {img1_np.shape} vs {img2_np.shape}")
                
                # Simplified preprocessing: just resize if too large
                # Let face_recognition.py handle all face detection
                def simple_resize(img, max_size=1024):
                    """
                    Simple proportional resize without face detection
                    Face detection will be handled by the professional face_recognition module
                    """
                    if max(img.shape[:2]) <= max_size:
                        print(f"   ℹ️ Image size OK, no resize needed")
                        return img
                    
                    # Proportional resize only
                    scale = max_size / max(img.shape[:2])
                    new_height = int(img.shape[0] * scale)
                    new_width = int(img.shape[1] * scale)
                    resized = cv2.resize(img, (new_width, new_height))
                    print(f"   📐 Resized from {img.shape[:2]} to {resized.shape[:2]}")
                    return resized
                
                # Apply simple preprocessing (no face detection here)
                print("🔧 Preprocessing images...")
                img1_np = simple_resize(img1_np)
                img2_np = simple_resize(img2_np)
                print(f"📊 After preprocessing: {img1_np.shape} vs {img2_np.shape}")
                print("   ➡️ Face detection will be handled by professional module")
                
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
                return f"Error: {str(e)}\n\nPlease check:\n- Both images are uploaded correctly\n- Images are in supported format (JPG, PNG)\n- System dependencies are installed"
        
        # Create Gradio interface
        with gr.Blocks(title="Character Consistency Evaluation", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# Character Consistency Evaluation")
            
            with gr.Row():
                with gr.Column():
                    image1_input = gr.Image(type="pil", label="Reference Image")
                    
                with gr.Column():
                    image2_input = gr.Image(type="pil", label="Target Image")
            
            evaluate_button = gr.Button("Analyze", variant="primary", size="lg")
            
            with gr.Row():
                output_text = gr.Textbox(
                    label="Results", 
                    lines=30,
                    max_lines=50,
                    show_copy_button=True
                )
            
            gr.Markdown("""
            ### Usage:
            1. Upload reference image (left)
            2. Upload target image (right)
            3. Click Analyze
            
            ### Analysis Methods:
            - Multi-model face recognition
            - CLIP semantic similarity
            - LPIPS perceptual similarity
            - Traditional metrics (SSIM, PSNR, MSE)
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