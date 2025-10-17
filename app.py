#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Image Evaluation System Launcher
Optimized for integrated graphics and CPU processing
"""

import os
import sys
import warnings
from data_logger import EvaluationDataLogger

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize global data logger
DATA_LOGGER = None

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
    if 'LPIPS_Distance' in results:
        lpips_dist = results['LPIPS_Distance']
        result_text += f"LPIPS Perceptual Analysis:\n"
        result_text += f"  Distance:    {lpips_dist:.4f} (lower is better)\n"
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
    result_text += "Evaluation Results (Basic Compatibility Mode)\n"
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
            
        result_text += f"Quick Assessment\n"
        result_text += f"{assessment_color} {assessment_text} (Score: {identity_score:.3f})\n\n"
    
    # Image Information
    result_text += "Image Information\n"
    result_text += "─"*25 + "\n"
    result_text += f"Original Image Size: {img1_np.shape[1]} × {img1_np.shape[0]} pixels\n"
    result_text += f"Comparison Image Size: {img2_np.shape[1]} × {img2_np.shape[0]} pixels\n"
    result_text += f"Processing Mode: Basic Fallback\n"
    result_text += f"Tip: Install advanced algorithm packages for detailed analysis\n\n"
    
    # Traditional Image Quality Metrics
    result_text += "Traditional Image Quality Metrics\n"
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
    result_text += "Professional Identity Analysis\n"
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
    result_text += "Basic Recommendations\n"
    result_text += "─"*30 + "\n"
    
    if 'Identity_Similarity' in results:
        score = results['Identity_Similarity']
        if score >= 0.6:
            result_text += "High similarity - can consider using\n"
        elif score >= 0.4:
            result_text += "Moderate similarity - manual review recommended\n"
        else:
            result_text += "Low similarity - use with caution\n"
    
    result_text += "Recommend upgrading to advanced algorithm mode for more accurate results\n\n"
    
    # Technical Info
    result_text += "Technical Information\n"
    result_text += "─"*30 + "\n"
    result_text += f"System Version: Basic Evaluation System v1.0\n"
    result_text += f"Compatibility Mode: CPU-optimized processing\n"
    result_text += f"Processing Time: {img1_np.shape[0] * img1_np.shape[1] / 200000:.1f}s (estimated)\n\n"
    
    result_text += "="*75 + "\n"
    result_text += "Basic Assessment Complete\n"
    result_text += "="*75 + "\n"
    
    return result_text

def main():
    try:
        import gradio as gr
        import numpy as np
        from PIL import Image
        import cv2
        import pandas as pd

        print("Initializing Professional Image Evaluation System...")
        print("CPU and Integrated Graphics Optimization Mode")

        # Check if advanced evaluation system is available
        try:
            from evaluator import CompatibleEvaluationSystem
            evaluator = CompatibleEvaluationSystem()
            use_advanced_algorithms = True
            print("Advanced algorithms loaded successfully")
            print("Multi-model consensus analysis enabled")
        except ImportError as e:
            print(f"Advanced algorithms not available: {e}")
            print("📱 Falling back to basic evaluation system...")
            use_advanced_algorithms = False
            
            # Basic fallback evaluator
            class BasicEvaluator:
                def evaluate_character_consistency(self, img1, img2, prompt=None):
                    try:
                        # Convert images to grayscale for basic analysis
                        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
                        
                        # Basic SSIM calculation
                        from skimage.metrics import structural_similarity
                        ssim_score, _ = structural_similarity(gray1, gray2, full=True)
                        
                        # Basic MSE calculation
                        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
                        
                        # Basic PSNR calculation
                        if mse > 0:
                            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                        else:
                            psnr = float('inf')
                        
                        return {
                            'SSIM': ssim_score,
                            'MSE': mse,
                            'PSNR': psnr,
                            'Final_Score': ssim_score, # Use SSIM as basic similarity
                            'Identity_Decision': 'Same Person' if ssim_score > 0.5 else 'Different Person',
                            'Detection_Method': 'Basic SSIM Analysis'
                        }
                    except Exception as e:
                        return {'Error': str(e)}
            
            evaluator = BasicEvaluator()

        def evaluate_multiple_images(reference_image, target_images, sample_category="Unknown", notes=""):
            """Main evaluation function for multiple target images."""
            if reference_image is None:
                return "Error: Please upload a reference image.", None
            if not target_images:
                return "Error: Please upload at least one target image.", None

            ref_img_np = np.array(reference_image)
            if len(ref_img_np.shape) == 3 and ref_img_np.shape[2] == 4:
                ref_img_np = ref_img_np[:, :, :3]

            all_results = []

            for target_image_file in target_images:
                try:
                    # For Gradio File component, target_image_file is a tempfile object
                    target_image = Image.open(target_image_file.name)
                    target_img_np = np.array(target_image)
                    if len(target_img_np.shape) == 3 and target_img_np.shape[2] == 4:
                        target_img_np = target_img_np[:, :, :3]

                    # Run evaluation
                    results = evaluator.evaluate_character_consistency(ref_img_np, target_img_np)
                    
                    # Log data
                    if DATA_LOGGER and sample_category != "Unknown":
                        try:
                            ref_name = getattr(reference_image, 'name', 'reference.jpg')
                            target_name = os.path.basename(target_image_file.name)
                            DATA_LOGGER.log_evaluation(
                                results=results,
                                sample_category=sample_category,
                                image1_name=ref_name,
                                image2_name=target_name,
                                notes=notes
                            )
                        except Exception as e:
                            print(f"Failed to log data for {target_name}: {e}")

                    all_results.append({
                        "image_name": os.path.basename(target_image_file.name),
                        "final_score": results.get('Final_Score', 0.0),
                        "identity_similarity": results.get('Identity_Similarity', 0.0),
                        "clip_similarity": results.get('CLIP_Similarity', 0.0),
                        "lpips_distance": results.get('LPIPS_Distance', 1.0),
                    })
                except Exception as e:
                    all_results.append({
                        "image_name": os.path.basename(target_image_file.name),
                        "final_score": 0.0,
                        "error": str(e)
                    })

            # Sort results by final_score descending
            sorted_results = sorted(all_results, key=lambda x: x.get('final_score', 0.0), reverse=True)

            # Format output
            output_text = "## Batch Evaluation Results\n\n"
            output_text += f"**Reference Image:** `{getattr(reference_image, 'name', 'reference.jpg')}`\n"
            output_text += f"**Total Target Images:** `{len(target_images)}`\n\n"
            output_text += "| Rank | Target Image | Final Score | Identity Sim. | CLIP Sim. | LPIPS Dist. |\n"
            output_text += "|:----:|:-------------|:-----------:|:-------------:|:---------:|:-----------:|\n"

            for i, res in enumerate(sorted_results):
                if "error" in res:
                    output_text += f"| {i+1} | {res['image_name']} | **ERROR** | - | - | - |\n"
                else:
                    output_text += (
                        f"| {i+1} "
                        f"| `{res['image_name']}` "
                        f"| **{res.get('final_score', 0.0):.4f}** "
                        f"| {res.get('identity_similarity', 0.0):.4f} "
                        f"| {res.get('clip_similarity', 0.0):.4f} "
                        f"| {res.get('lpips_distance', 1.0):.4f} |\n"
                    )
            
            # Create a DataFrame for the plot
            df = pd.DataFrame(sorted_results)
            
            return output_text, df

        def evaluate_images(image1, image2, sample_category="Unknown", sample_id=None, notes=""):
            """Main evaluation function with enhanced formatting and data logging"""
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
                
                # Simplified preprocessing: resize to reasonable size for face detection
                def simple_resize(img, max_size=800):
                    """
                    Resize image to max_size while maintaining aspect ratio.
                    Use smaller max_size (800) to ensure faces remain detectable after resize.
                    """
                    h, w = img.shape[:2]
                    if max(h, w) <= max_size:
                        print(f"   ✅ Image size OK ({w}x{h}), no resize needed")
                        return img
                    scale = max_size / max(h, w)
                    new_height = int(h * scale)
                    new_width = int(w * scale)
                    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    print(f"   📏 Resized from {w}x{h} to {new_width}x{new_height}")
                    return resized
                
                print(f"🖼️  Original image sizes: {img1_np.shape} vs {img2_np.shape}")
                img1_np = simple_resize(img1_np)
                img2_np = simple_resize(img2_np)
                print(f"📊 Processing images: {img1_np.shape} vs {img2_np.shape}")
                
                # Run evaluation
                results = evaluator.evaluate_character_consistency(img1_np, img2_np)
                
                # Format results
                result_text = format_segmented_results(results, use_advanced_algorithms, img1_np, img2_np)
                
                # Log results
                logged_status = "\n\n" + "="*70 + "\n"
                if DATA_LOGGER:
                    if sample_category != "Unknown":
                        try:
                            img1_name = getattr(image1, 'name', 'image1.jpg')
                            img2_name = getattr(image2, 'name', 'image2.jpg')
                            logged_id = DATA_LOGGER.log_evaluation(
                                results=results,
                                sample_category=sample_category,
                                sample_id=sample_id,
                                image1_name=img1_name,
                                image2_name=img2_name,
                                notes=notes
                            )
                            logged_status += f"✅ DATA SAVED: {logged_id}\n"
                            logged_status += f"   Category: {sample_category}\n"
                            logged_status += f"   Location: evaluation_data/evaluation_results.csv\n"
                        except Exception as e:
                            logged_status += f"❌ DATA SAVE FAILED: {str(e)}\n"
                            print(f"Failed to log data: {e}")
                    else:
                        logged_status += "⚠️  DATA NOT SAVED: Please select a category (Basic/Attribute/Boundary)\n"
                else:
                    logged_status += "❌ DATA LOGGER NOT INITIALIZED\n"
                logged_status += "="*70
                
                result_text += logged_status
                
                return result_text
                
            except Exception as e:
                return f"Error: {str(e)}\n\nPlease check system dependencies."

        # Initialize data logger
        global DATA_LOGGER
        try:
            DATA_LOGGER = EvaluationDataLogger(data_dir="evaluation_data")
            print("Data logger initialized - results will be saved automatically")
        except Exception as e:
            print(f"Data logger initialization failed: {e}")
            DATA_LOGGER = None

        # Create Gradio interface
        with gr.Blocks(title="Character Consistency Evaluation", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Character Consistency Evaluation")
            gr.Markdown("*Research Data Collection: Results are automatically logged for hypothesis validation*")

            with gr.Tabs():
                with gr.TabItem("Single Comparison"):
                    with gr.Row():
                        with gr.Column():
                            image1_input = gr.Image(type="pil", label="Reference Image")
                        with gr.Column():
                            image2_input = gr.Image(type="pil", label="Target Image")
                    with gr.Row():
                        s_category_input = gr.Dropdown(
                            ["Basic", "Attribute", "Boundary", "Unknown"], 
                            value="Basic", 
                            label="Sample Category",
                            info="⚠️ Select category to save results to database"
                        )
                        s_sample_id_input = gr.Textbox(
                            label="Sample ID (Optional)", 
                            placeholder="Auto-generated if empty",
                            value=""
                        )
                        s_notes_input = gr.Textbox(
                            label="Notes (Optional)",
                            placeholder="Add notes about this evaluation"
                        )
                    s_evaluate_button = gr.Button("Analyze Single", variant="primary")
                    s_output_text = gr.Textbox(label="Results", lines=30, max_lines=50, show_copy_button=True)

                with gr.TabItem("Batch Comparison"):
                    with gr.Row():
                        with gr.Column():
                            b_image1_input = gr.Image(type="pil", label="Reference Image")
                        with gr.Column():
                            b_images2_input = gr.File(label="Target Images", file_count="multiple", file_types=["image"])
                    with gr.Row():
                        b_category_input = gr.Dropdown(
                            ["Basic", "Attribute", "Boundary", "Unknown"], 
                            value="Basic", 
                            label="Sample Category",
                            info="⚠️ Select category to save results to database"
                        )
                        b_notes_input = gr.Textbox(
                            label="Notes (Optional)",
                            placeholder="Add notes for all images in batch"
                        )
                    b_evaluate_button = gr.Button("Analyze Batch", variant="primary")
                    b_output_text = gr.Markdown(label="Batch Results")
                    b_plot_output = gr.BarPlot(x="image_name", y="final_score", title="Batch Results", y_lim=[0, 1])

            s_evaluate_button.click(
                evaluate_images,
                inputs=[image1_input, image2_input, s_category_input, s_sample_id_input, s_notes_input],
                outputs=s_output_text
            )
            
            b_evaluate_button.click(
                evaluate_multiple_images,
                inputs=[b_image1_input, b_images2_input, b_category_input, b_notes_input],
                outputs=[b_output_text, b_plot_output]
            )
        
        print("Starting web interface on http://127.0.0.1:7862")
        interface.launch(server_name="127.0.0.1", server_port=7862, show_error=True)
        
    except Exception as e:
        print(f"Failed to start system: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if all required packages are installed: pip install -r requirements.txt")
        print("2. Ensure you have a stable internet connection for model downloads.")
        sys.exit(1)

if __name__ == "__main__":
    main()