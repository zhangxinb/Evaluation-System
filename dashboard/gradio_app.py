import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
import json
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import CLIPEvaluator, IdentityEvaluator, PerceptualEvaluator, TraditionalMetrics
from utils import DataLoader, ImagePreprocessor, Visualizer
from config.settings import Config

class GradioEvaluationApp:
    """Gradio evaluation application"""
    
    def __init__(self):
        self.clip_evaluator = None
        self.identity_evaluator = None
        self.perceptual_evaluator = None
        self.traditional_metrics = None
        self.data_loader = DataLoader()
        self.preprocessor = ImagePreprocessor()
        self.visualizer = Visualizer()
        
        # Initialize evaluators
        self._initialize_evaluators()
    
    def _initialize_evaluators(self):
        """Initialize all evaluators"""
        try:
            print("Initializing evaluators...")
            self.clip_evaluator = CLIPEvaluator()
            self.identity_evaluator = IdentityEvaluator()
            self.perceptual_evaluator = PerceptualEvaluator()
            self.traditional_metrics = TraditionalMetrics()
            print("Evaluators initialization completed!")
        except Exception as e:
            print(f"Evaluator initialization failed: {e}")
            return str(e)
    
    def evaluate_single_image(self, 
                            generated_image: Image.Image,
                            reference_image: Image.Image = None,
                            prompt: str = "",
                            selected_metrics: List[str] = None) -> Tuple[Dict, str, str]:
        """Evaluate single image"""
        if generated_image is None:
            return {}, "Please upload a generated image", ""
        
        if selected_metrics is None:
            selected_metrics = ["CLIP", "Traditional Metrics"]
        
        results = {}
        error_msg = ""
        
        try:
            # CLIP evaluation
            if "CLIP" in selected_metrics and prompt.strip():
                if self.clip_evaluator:
                    clip_result = self.clip_evaluator.evaluate_prompt_consistency(generated_image, prompt)
                    results['clip'] = clip_result
            
            # Identity consistency evaluation
            if "Identity Consistency" in selected_metrics and reference_image is not None:
                if self.identity_evaluator:
                    identity_result = self.identity_evaluator.evaluate_identity_consistency(
                        reference_image, generated_image)
                    results['identity'] = identity_result
            
            # Perceptual similarity evaluation
            if "Perceptual Similarity" in selected_metrics and reference_image is not None:
                if self.perceptual_evaluator:
                    perceptual_result = self.perceptual_evaluator.evaluate_perceptual_similarity(
                        reference_image, generated_image)
                    results['perceptual'] = perceptual_result
            
            # Traditional metrics evaluation
            if "Traditional Metrics" in selected_metrics and reference_image is not None:
                if self.traditional_metrics:
                    traditional_result = self.traditional_metrics.evaluate_traditional_metrics(
                        reference_image, generated_image)
                    results['traditional'] = traditional_result
            
            # 生成报告
            report = self._generate_report(results)
            
            # 生成可视化
            viz_html = self._generate_visualization(results)
            
            return results, report, viz_html
            
        except Exception as e:
            error_msg = f"Error occurred during evaluation: {str(e)}"
            return {}, error_msg, ""
    
    def batch_evaluate(self, 
                      file_pairs: List[Tuple[str, str]], 
                      prompts_text: str = "",
                      selected_metrics: List[str] = None) -> Tuple[str, str]:
        """Batch evaluation"""
        if not file_pairs:
            return "Please upload image files", ""
        
        batch_results = []
        prompts_dict = {}
        
        # 解析提示词
        if prompts_text.strip():
            for line in prompts_text.strip().split('\n'):
                if ':' in line:
                    filename, prompt = line.split(':', 1)
                    prompts_dict[filename.strip()] = prompt.strip()
        
        try:
            for ref_path, gen_path in file_pairs:
                # 加载图像
                ref_img = self.data_loader.load_image(ref_path)
                gen_img = self.data_loader.load_image(gen_path)
                
                if ref_img is None or gen_img is None:
                    continue
                
                # 获取文件名
                filename = os.path.basename(gen_path)
                prompt = prompts_dict.get(filename, "")
                
                # 评估
                result, _, _ = self.evaluate_single_image(
                    gen_img, ref_img, prompt, selected_metrics)
                
                if result:
                    result['filename'] = filename
                    batch_results.append(result)
            
            # 生成批量报告
            report = self._generate_batch_report(batch_results)
            
            # 保存结果
            self.data_loader.save_results_to_json(batch_results, "batch_results.json")
            
            return report, "batch_results.json"
            
        except Exception as e:
            return f"Batch evaluation failed: {str(e)}", ""
    
    def _generate_report(self, results: Dict) -> str:
        """Generate evaluation report"""
        report = "# Image Consistency Evaluation Report\n\n"
        
        if 'clip' in results:
            clip_result = results['clip']
            report += f"## CLIP Semantic Consistency\n"
            report += f"- Similarity: {clip_result['similarity']:.3f}\n"
            report += f"- Score: {clip_result['score']:.1f}/100\n"
            report += f"- Evaluation: {clip_result['evaluation']}\n"
            report += f"- Passed Threshold: {'Yes' if clip_result['passed'] else 'No'}\n\n"
        
        if 'identity' in results:
            identity_result = results['identity']
            report += f"## Identity Consistency\n"
            report += f"- Final Similarity: {identity_result['final_similarity']:.3f}\n"
            report += f"- Score: {identity_result['score']:.1f}/100\n"
            report += f"- Evaluation: {identity_result['evaluation']}\n"
            report += f"- Passed Threshold: {'Yes' if identity_result['passed'] else 'No'}\n\n"
        
        if 'perceptual' in results:
            perceptual_result = results['perceptual']
            report += f"## Perceptual Similarity (LPIPS)\n"
            report += f"- Similarity: {perceptual_result['lpips_similarity']:.3f}\n"
            report += f"- Distance: {perceptual_result['lpips_distance']:.3f}\n"
            report += f"- Score: {perceptual_result['score']:.1f}/100\n"
            report += f"- Evaluation: {perceptual_result['evaluation']}\n\n"
        
        if 'traditional' in results:
            traditional_result = results['traditional']
            report += f"## Traditional Image Quality Metrics\n"
            if 'ssim' in traditional_result:
                ssim_data = traditional_result['ssim']
                report += f"- SSIM: {ssim_data['value']:.3f} (Score: {ssim_data['score']:.1f}/100)\n"
            if 'psnr' in traditional_result:
                psnr_data = traditional_result['psnr']
                report += f"- PSNR: {psnr_data['value']:.2f} dB (Score: {psnr_data['score']:.1f}/100)\n"
        
        # 计算综合评分
        all_scores = []
        if 'clip' in results:
            all_scores.append(results['clip']['score'])
        if 'identity' in results:
            all_scores.append(results['identity']['score'])
        if 'perceptual' in results:
            all_scores.append(results['perceptual']['score'])
        if 'traditional' in results:
            if 'ssim' in results['traditional']:
                all_scores.append(results['traditional']['ssim']['score'])
            if 'psnr' in results['traditional']:
                all_scores.append(results['traditional']['psnr']['score'])
        
        if all_scores:
            overall_score = np.mean(all_scores)
            report += f"\n## Overall Score\n"
            report += f"- Total Score: {overall_score:.1f}/100\n"
            
            if overall_score >= 80:
                report += "- Overall Evaluation: Excellent\n"
            elif overall_score >= 60:
                report += "- Overall Evaluation: Good\n"
            elif overall_score >= 40:
                report += "- Overall Evaluation: Fair\n"
            else:
                report += "- Overall Evaluation: Poor\n"
        
        return report
    
    def _generate_batch_report(self, batch_results: List[Dict]) -> str:
        """Generate batch evaluation report"""
        if not batch_results:
            return "No valid evaluation results"
        
        report = f"# Batch Evaluation Report\n\n"
        report += f"Total evaluated images: {len(batch_results)}\n\n"
        
        # Statistics for each metric average
        clip_scores = [r['clip']['score'] for r in batch_results if 'clip' in r]
        identity_scores = [r['identity']['score'] for r in batch_results if 'identity' in r]
        perceptual_scores = [r['perceptual']['score'] for r in batch_results if 'perceptual' in r]
        
        if clip_scores:
            report += f"## CLIP Semantic Consistency Statistics\n"
            report += f"- Average Score: {np.mean(clip_scores):.1f}\n"
            report += f"- Standard Deviation: {np.std(clip_scores):.1f}\n"
            report += f"- Highest Score: {np.max(clip_scores):.1f}\n"
            report += f"- Lowest Score: {np.min(clip_scores):.1f}\n\n"
        
        if identity_scores:
            report += f"## Identity Consistency Statistics\n"
            report += f"- Average Score: {np.mean(identity_scores):.1f}\n"
            report += f"- Standard Deviation: {np.std(identity_scores):.1f}\n"
            report += f"- Highest Score: {np.max(identity_scores):.1f}\n"
            report += f"- Lowest Score: {np.min(identity_scores):.1f}\n\n"
        
        return report
    
    def _generate_visualization(self, results: Dict) -> str:
        """Generate visualization chart HTML"""
        try:
            # Create a simple score bar chart
            scores = []
            labels = []
            
            if 'clip' in results:
                scores.append(results['clip']['score'])
                labels.append('CLIP')
            if 'identity' in results:
                scores.append(results['identity']['score'])
                labels.append('Identity Consistency')
            if 'perceptual' in results:
                scores.append(results['perceptual']['score'])
                labels.append('Perceptual Similarity')
            
            if not scores:
                return "<p>No visualization data available</p>"
            
            # Create simple HTML bar chart
            html = "<div style='margin: 20px;'>"
            html += "<h3>Evaluation Score Visualization</h3>"
            
            for label, score in zip(labels, scores):
                percentage = score
                color = "green" if score >= 70 else "orange" if score >= 50 else "red"
                
                html += f"""
                <div style='margin: 10px 0;'>
                    <div style='font-weight: bold;'>{label}: {score:.1f}/100</div>
                    <div style='background-color: #f0f0f0; height: 20px; border-radius: 10px; overflow: hidden;'>
                        <div style='background-color: {color}; height: 100%; width: {percentage}%; transition: width 0.3s;'></div>
                    </div>
                </div>
                """
            
            html += "</div>"
            return html
            
        except Exception as e:
            return f"<p>Visualization generation failed: {str(e)}</p>"
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="Image Consistency Evaluation System", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎯 Image Consistency Evaluation System")
            gr.Markdown("This system comprehensively evaluates the consistency of generated images across multiple dimensions, including semantic consistency, identity consistency, perceptual similarity, and more.")
            
            with gr.Tabs():
                # Single Image Evaluation Tab
                with gr.TabItem("Single Image Evaluation"):
                    with gr.Row():
                        with gr.Column():
                            generated_img = gr.Image(type="pil", label="Generated Image (Required)")
                            reference_img = gr.Image(type="pil", label="Reference Image (Optional)")
                            prompt_input = gr.Textbox(
                                label="Text Prompt",
                                placeholder="Enter the text prompt used to generate the image...",
                                lines=3
                            )
                            
                            metrics_choice = gr.CheckboxGroup(
                                choices=["CLIP", "Identity Consistency", "Perceptual Similarity", "Traditional Metrics"],
                                value=["CLIP", "Traditional Metrics"],
                                label="Select Evaluation Metrics"
                            )
                            
                            evaluate_btn = gr.Button("Start Evaluation", variant="primary", size="lg")
                        
                        with gr.Column():
                            results_json = gr.JSON(label="Detailed Results")
                            report_md = gr.Markdown(label="Evaluation Report")
                            viz_html = gr.HTML(label="Visualization Results")
                    
                    evaluate_btn.click(
                        fn=self.evaluate_single_image,
                        inputs=[generated_img, reference_img, prompt_input, metrics_choice],
                        outputs=[results_json, report_md, viz_html]
                    )
                
                # Batch Evaluation Tab
                with gr.TabItem("Batch Evaluation"):
                    with gr.Row():
                        with gr.Column():
                            batch_files = gr.File(
                                file_count="multiple",
                                label="Upload Image Files (Multiple Selection Supported)",
                                file_types=["image"]
                            )
                            
                            prompts_text = gr.Textbox(
                                label="Prompt List",
                                placeholder="Format: filename: prompt\nExample:\nimage1.jpg: a beautiful landscape\nimage2.jpg: a portrait of a person",
                                lines=10
                            )
                            
                            batch_metrics = gr.CheckboxGroup(
                                choices=["CLIP", "Identity Consistency", "Perceptual Similarity", "Traditional Metrics"],
                                value=["CLIP"],
                                label="Select Evaluation Metrics"
                            )
                            
                            batch_evaluate_btn = gr.Button("Batch Evaluation", variant="primary", size="lg")
                        
                        with gr.Column():
                            batch_report = gr.Markdown(label="Batch Evaluation Report")
                            results_file = gr.File(label="Download Results File")
                    
                    # Batch evaluation functionality temporarily simplified
                    batch_evaluate_btn.click(
                        fn=lambda: ("Batch evaluation feature is under development...", None),
                        outputs=[batch_report, results_file]
                    )
                
                # Settings Tab
                with gr.TabItem("System Settings"):
                    gr.Markdown("## Evaluation Threshold Settings")
                    
                    with gr.Row():
                        clip_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                            label="CLIP Similarity Threshold"
                        )
                        identity_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.6, step=0.05,
                            label="Identity Consistency Threshold"
                        )
                    
                    with gr.Row():
                        lpips_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                            label="LPIPS Distance Threshold"
                        )
                        ssim_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.8, step=0.05,
                            label="SSIM Threshold"
                        )
                    
                    update_settings_btn = gr.Button("Update Settings")
                    
                    def update_thresholds(clip_th, identity_th, lpips_th, ssim_th):
                        Config.THRESHOLDS['clip_threshold'] = clip_th
                        Config.THRESHOLDS['identity_threshold'] = identity_th
                        Config.THRESHOLDS['lpips_threshold'] = lpips_th
                        Config.THRESHOLDS['ssim_threshold'] = ssim_th
                        return "Settings updated!"
                    
                    update_settings_btn.click(
                        fn=update_thresholds,
                        inputs=[clip_threshold, identity_threshold, lpips_threshold, ssim_threshold],
                        outputs=gr.Markdown()
                    )
            
            # System Status Information
            with gr.Row():
                gr.Markdown(f"""
                ### System Information
                - Device: {Config.DEVICE}
                - CLIP Model: {Config.CLIP_MODEL_NAME}
                - Image Processing Size: {Config.IMAGE_SIZE}
                """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch application"""
        interface = self.create_interface()
        return interface.launch(**kwargs)