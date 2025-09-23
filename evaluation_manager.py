import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from PIL import Image
import json
import time
from datetime import datetime

from core import CLIPEvaluator, IdentityEvaluator, PerceptualEvaluator, TraditionalMetrics
from utils import DataLoader, ImagePreprocessor, Visualizer
from config.settings import Config

class EvaluationManager:
    """Unified evaluation manager that integrates all evaluation metrics"""
    
    def __init__(self):
        self.clip_evaluator = None
        self.identity_evaluator = None 
        self.perceptual_evaluator = None
        self.traditional_metrics = None
        
        self.data_loader = DataLoader()
        self.preprocessor = ImagePreprocessor()
        self.visualizer = Visualizer()
        
        self.evaluation_history = []
        
    def initialize_evaluators(self) -> bool:
        """Initialize all evaluators"""
        try:
            print("Initializing CLIP evaluator...")
            self.clip_evaluator = CLIPEvaluator()
            
            print("Initializing identity evaluator...")
            self.identity_evaluator = IdentityEvaluator()
            
            print("Initializing perceptual evaluator...")
            self.perceptual_evaluator = PerceptualEvaluator()
            
            print("Initializing traditional metrics evaluator...")
            self.traditional_metrics = TraditionalMetrics()
            
            print("✅ All evaluators initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Evaluator initialization failed: {e}")
            return False
    
    def evaluate_comprehensive(self, 
                             generated_image: Union[Image.Image, np.ndarray],
                             reference_image: Union[Image.Image, np.ndarray] = None,
                             prompt: str = "",
                             metrics: List[str] = None) -> Dict:
        """
        全面评估图像一致性
        
        Args:
            generated_image: 生成的图像
            reference_image: 参考图像 (可选)
            prompt: 文本提示词 (可选)
            metrics: 要评估的指标列表
            
        Returns:
            包含所有评估结果的字典
        """
        start_time = time.time()
        
        if metrics is None:
            metrics = ['clip', 'identity', 'perceptual', 'traditional']
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics_evaluated': metrics,
            'evaluation_time': 0,
            'overall_score': 0,
            'evaluation_summary': {},
            'detailed_results': {}
        }
        
        scores_for_overall = []
        
        try:
            # 预处理图像
            if not self.preprocessor.validate_image(generated_image):
                raise ValueError("Generated image is invalid")
            
            processed_gen_img = self.preprocessor.preprocess_for_evaluation(
                generated_image, normalize=False)
            
            if reference_image is not None:
                if not self.preprocessor.validate_image(reference_image):
                    print("Warning: Reference image is invalid, skipping related evaluations")
                    reference_image = None
                else:
                    processed_ref_img = self.preprocessor.preprocess_for_evaluation(
                        reference_image, normalize=False)
            
            # CLIP语义一致性评估
            if 'clip' in metrics and prompt.strip() and self.clip_evaluator:
                print("Performing CLIP semantic consistency evaluation...")
                clip_result = self.clip_evaluator.evaluate_prompt_consistency(
                    processed_gen_img, prompt)
                results['detailed_results']['clip'] = clip_result
                scores_for_overall.append(clip_result['score'])
                
                results['evaluation_summary']['clip'] = {
                    'score': clip_result['score'],
                    'evaluation': clip_result['evaluation'],
                    'passed': clip_result['passed']
                }
            
            # 身份一致性评估
            if 'identity' in metrics and reference_image is not None and self.identity_evaluator:
                print("Performing identity consistency evaluation...")
                identity_result = self.identity_evaluator.evaluate_identity_consistency(
                    processed_ref_img, processed_gen_img)
                results['detailed_results']['identity'] = identity_result
                scores_for_overall.append(identity_result['score'])
                
                results['evaluation_summary']['identity'] = {
                    'score': identity_result['score'],
                    'evaluation': identity_result['evaluation'],
                    'passed': identity_result['passed']
                }
            
            # 感知相似度评估
            if 'perceptual' in metrics and reference_image is not None and self.perceptual_evaluator:
                print("Performing perceptual similarity evaluation...")
                perceptual_result = self.perceptual_evaluator.evaluate_perceptual_similarity(
                    processed_ref_img, processed_gen_img)
                results['detailed_results']['perceptual'] = perceptual_result
                scores_for_overall.append(perceptual_result['score'])
                
                results['evaluation_summary']['perceptual'] = {
                    'score': perceptual_result['score'],
                    'evaluation': perceptual_result['evaluation'],
                    'passed': perceptual_result['passed']
                }
            
            # 传统图像质量指标评估
            if 'traditional' in metrics and reference_image is not None and self.traditional_metrics:
                print("Performing traditional metrics evaluation...")
                traditional_result = self.traditional_metrics.evaluate_traditional_metrics(
                    processed_ref_img, processed_gen_img)
                results['detailed_results']['traditional'] = traditional_result
                
                # 添加SSIM和PSNR分数
                if 'ssim' in traditional_result:
                    scores_for_overall.append(traditional_result['ssim']['score'])
                if 'psnr' in traditional_result:
                    scores_for_overall.append(traditional_result['psnr']['score'])
                
                results['evaluation_summary']['traditional'] = {
                    'ssim_score': traditional_result.get('ssim', {}).get('score', 0),
                    'psnr_score': traditional_result.get('psnr', {}).get('score', 0),
                    'ssim_evaluation': traditional_result.get('ssim', {}).get('evaluation', 'unknown'),
                    'psnr_evaluation': traditional_result.get('psnr', {}).get('evaluation', 'unknown')
                }
            
            # 计算综合评分
            if scores_for_overall:
                # 使用配置的权重计算加权平均
                weights = []
                weighted_scores = []
                
                metric_weight_map = {
                    'clip': Config.METRIC_WEIGHTS.get('clip_similarity', 0.3),
                    'identity': Config.METRIC_WEIGHTS.get('identity_similarity', 0.25),
                    'perceptual': Config.METRIC_WEIGHTS.get('lpips_similarity', 0.25),
                    'traditional': Config.METRIC_WEIGHTS.get('ssim_similarity', 0.1) + 
                                  Config.METRIC_WEIGHTS.get('psnr_similarity', 0.1)
                }
                
                for metric in results['evaluation_summary']:
                    weight = metric_weight_map.get(metric, 0.25)
                    if metric == 'traditional':
                        # 传统指标取SSIM和PSNR的平均值
                        trad_score = (results['evaluation_summary']['traditional']['ssim_score'] + 
                                    results['evaluation_summary']['traditional']['psnr_score']) / 2
                        weighted_scores.append(trad_score * weight)
                    else:
                        score = results['evaluation_summary'][metric]['score']
                        weighted_scores.append(score * weight)
                    weights.append(weight)
                
                # 归一化权重
                total_weight = sum(weights)
                if total_weight > 0:
                    results['overall_score'] = sum(weighted_scores) / total_weight
                else:
                    results['overall_score'] = np.mean(scores_for_overall)
            
            # 生成总体评价
            overall_score = results['overall_score']
            if overall_score >= 80:
                overall_evaluation = "优秀"
            elif overall_score >= 60:
                overall_evaluation = "良好"
            elif overall_score >= 40:
                overall_evaluation = "一般"
            else:
                overall_evaluation = "较差"
            
            results['overall_evaluation'] = overall_evaluation
            
            # 记录评估时间
            results['evaluation_time'] = time.time() - start_time
            
            # 添加到历史记录
            self.evaluation_history.append(results.copy())
            
            print(f"✅ Evaluation completed, overall score: {overall_score:.1f}/100 ({overall_evaluation})")
            
            return results
            
        except Exception as e:
            print(f"❌ Error occurred during evaluation: {e}")
            results['error'] = str(e)
            results['evaluation_time'] = time.time() - start_time
            return results
    
    def batch_evaluate(self, 
                      image_pairs: List[Tuple[Union[Image.Image, np.ndarray], 
                                            Union[Image.Image, np.ndarray]]],
                      prompts: List[str] = None,
                      metrics: List[str] = None) -> List[Dict]:
        """Batch evaluation for multiple image pairs"""
        if prompts is None:
            prompts = [""] * len(image_pairs)
        
        if len(prompts) < len(image_pairs):
            prompts.extend([""] * (len(image_pairs) - len(prompts)))
        
        batch_results = []
        total_pairs = len(image_pairs)
        
        print(f"Starting batch evaluation, total {total_pairs} image pairs...")
        
        for i, ((ref_img, gen_img), prompt) in enumerate(zip(image_pairs, prompts)):
            print(f"Evaluating image pair {i+1}/{total_pairs}...")
            
            result = self.evaluate_comprehensive(
                generated_image=gen_img,
                reference_image=ref_img,
                prompt=prompt,
                metrics=metrics
            )
            
            result['batch_index'] = i
            batch_results.append(result)
        
        # 生成批量统计
        batch_stats = self._generate_batch_statistics(batch_results)
        
        return batch_results, batch_stats
    
    def _generate_batch_statistics(self, batch_results: List[Dict]) -> Dict:
        """Generate batch evaluation statistics"""
        if not batch_results:
            return {}
        
        # 收集所有有效的评分
        overall_scores = [r['overall_score'] for r in batch_results if 'overall_score' in r and r['overall_score'] > 0]
        
        stats = {
            'total_evaluations': len(batch_results),
            'successful_evaluations': len(overall_scores),
            'failed_evaluations': len(batch_results) - len(overall_scores)
        }
        
        if overall_scores:
            stats.update({
                'mean_score': np.mean(overall_scores),
                'std_score': np.std(overall_scores),
                'min_score': np.min(overall_scores),
                'max_score': np.max(overall_scores),
                'median_score': np.median(overall_scores)
            })
            
            # 评价分布
            excellent_count = sum(1 for score in overall_scores if score >= 80)
            good_count = sum(1 for score in overall_scores if 60 <= score < 80)
            fair_count = sum(1 for score in overall_scores if 40 <= score < 60)
            poor_count = sum(1 for score in overall_scores if score < 40)
            
            stats['evaluation_distribution'] = {
                'excellent': excellent_count,
                'good': good_count,
                'fair': fair_count,
                'poor': poor_count
            }
        
        return stats
    
    def save_evaluation_results(self, results: Union[Dict, List[Dict]], 
                              filename: str = None) -> str:
        """Save evaluation results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        try:
            # 处理numpy数组序列化问题
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            serializable_results = convert_numpy(results)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Evaluation results saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return ""
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate readable evaluation report"""
        report = "# Image Consistency Evaluation Report\n\n"
        
        # Basic information
        report += f"**Evaluation Time**: {results.get('timestamp', 'Unknown')}\n"
        report += f"**Duration**: {results.get('evaluation_time', 0):.2f} seconds\n"
        report += f"**Overall Score**: {results.get('overall_score', 0):.1f}/100\n"
        report += f"**Overall Rating**: {results.get('overall_evaluation', 'Unknown')}\n\n"
        
        # Detailed metrics
        summary = results.get('evaluation_summary', {})
        
        if 'clip' in summary:
            clip_data = summary['clip']
            report += f"## CLIP Semantic Consistency\n"
            report += f"- Score: {clip_data['score']:.1f}/100\n"
            report += f"- Rating: {clip_data['evaluation']}\n"
            report += f"- Passed Threshold: {'Yes' if clip_data['passed'] else 'No'}\n\n"
        
        if 'identity' in summary:
            identity_data = summary['identity']
            report += f"## Identity Consistency\n"
            report += f"- Score: {identity_data['score']:.1f}/100\n"
            report += f"- Rating: {identity_data['evaluation']}\n"
            report += f"- Passed Threshold: {'Yes' if identity_data['passed'] else 'No'}\n\n"
        
        if 'perceptual' in summary:
            perceptual_data = summary['perceptual']
            report += f"## Perceptual Similarity\n"
            report += f"- Score: {perceptual_data['score']:.1f}/100\n"
            report += f"- Rating: {perceptual_data['evaluation']}\n"
            report += f"- Passed Threshold: {'Yes' if perceptual_data['passed'] else 'No'}\n\n"
        
        if 'traditional' in summary:
            traditional_data = summary['traditional']
            report += f"## Traditional Image Quality Metrics\n"
            report += f"- SSIM Score: {traditional_data['ssim_score']:.1f}/100 ({traditional_data['ssim_evaluation']})\n"
            report += f"- PSNR Score: {traditional_data['psnr_score']:.1f}/100 ({traditional_data['psnr_evaluation']})\n\n"
        
        # Recommendations and summary
        overall_score = results.get('overall_score', 0)
        report += "## Evaluation Summary\n"
        
        if overall_score >= 80:
            report += "🎉 Excellent image quality! The generated image performs excellently across all dimensions.\n"
        elif overall_score >= 60:
            report += "👍 Good image quality, most metrics show satisfactory performance.\n"
        elif overall_score >= 40:
            report += "⚠️ Average image quality, improvements recommended in some aspects.\n"
        else:
            report += "🔄 Poor image quality, regeneration or parameter adjustment recommended.\n"
        
        return report
    
    def get_evaluation_history(self, limit: int = 10) -> List[Dict]:
        """Get evaluation history"""
        return self.evaluation_history[-limit:]
    
    def clear_evaluation_history(self):
        """Clear evaluation history"""
        self.evaluation_history.clear()
        print("✅ Evaluation history cleared")