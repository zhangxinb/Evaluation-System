import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import lpips
from typing import Union, List
from config.settings import Config

class PerceptualEvaluator:
    """Perceptual similarity evaluator using LPIPS to compute perceptual differences"""
    
    def __init__(self, net: str = 'alex'):
        """
        初始化LPIPS评估器
        Args:
            net: 使用的网络类型 ('alex', 'vgg', 'squeeze')
        """
        self.device = Config.DEVICE
        self.net = net
        
        # 初始化LPIPS模型
        self.lpips_model = lpips.LPIPS(net=net).to(self.device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """预处理图像"""
        if isinstance(image, np.ndarray):
            # 确保数组是uint8类型
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 应用变换
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def compute_lpips_distance(self, image1: Union[Image.Image, np.ndarray],
                              image2: Union[Image.Image, np.ndarray]) -> float:
        """计算两张图像的LPIPS距离"""
        # 预处理图像
        tensor1 = self.preprocess_image(image1)
        tensor2 = self.preprocess_image(image2)
        
        # 计算LPIPS距离
        with torch.no_grad():
            distance = self.lpips_model(tensor1, tensor2).item()
        
        return distance
    
    def compute_lpips_similarity(self, image1: Union[Image.Image, np.ndarray],
                                image2: Union[Image.Image, np.ndarray]) -> float:
        """计算LPIPS相似度（距离的倒数形式）"""
        distance = self.compute_lpips_distance(image1, image2)
        
        # 将距离转换为相似度 (0-1范围)
        # LPIPS距离通常在0-1范围内，距离越小相似度越高
        similarity = max(0, 1 - distance)
        return similarity
    
    def evaluate_perceptual_similarity(self, reference_image: Union[Image.Image, np.ndarray],
                                     generated_image: Union[Image.Image, np.ndarray]) -> dict:
        """评估生成图像与参考图像的感知相似度"""
        # 计算LPIPS距离和相似度
        lpips_distance = self.compute_lpips_distance(reference_image, generated_image)
        lpips_similarity = self.compute_lpips_similarity(reference_image, generated_image)
        
        # 计算评分 (0-100)
        score = lpips_similarity * 100
        
        # 判断是否通过阈值
        threshold = Config.THRESHOLDS['lpips_threshold']
        passed = lpips_distance <= threshold  # 距离阈值，越小越好
        
        return {
            'lpips_distance': lpips_distance,
            'lpips_similarity': lpips_similarity,
            'score': score,
            'passed': passed,
            'threshold': threshold,
            'evaluation': 'excellent' if lpips_distance <= 0.1 else 
                         'good' if lpips_distance <= 0.3 else
                         'fair' if lpips_distance <= 0.5 else 'poor',
            'network': self.net
        }
    
    def batch_perceptual_evaluation(self, reference_images: List[Union[Image.Image, np.ndarray]],
                                  generated_images: List[Union[Image.Image, np.ndarray]]) -> List[dict]:
        """批量评估感知相似度"""
        results = []
        
        for ref_img, gen_img in zip(reference_images, generated_images):
            result = self.evaluate_perceptual_similarity(ref_img, gen_img)
            results.append(result)
        
        return results
    
    def compare_multiple_networks(self, image1: Union[Image.Image, np.ndarray],
                                 image2: Union[Image.Image, np.ndarray],
                                 networks: List[str] = ['alex', 'vgg']) -> dict:
        """使用多个网络比较感知相似度"""
        results = {}
        
        for net in networks:
            temp_model = lpips.LPIPS(net=net).to(self.device)
            
            tensor1 = self.preprocess_image(image1)
            tensor2 = self.preprocess_image(image2)
            
            with torch.no_grad():
                distance = temp_model(tensor1, tensor2).item()
            
            results[f'{net}_distance'] = distance
            results[f'{net}_similarity'] = max(0, 1 - distance)
        
        # 计算平均值
        avg_distance = np.mean([results[f'{net}_distance'] for net in networks])
        avg_similarity = np.mean([results[f'{net}_similarity'] for net in networks])
        
        results['average_distance'] = avg_distance
        results['average_similarity'] = avg_similarity
        
        return results