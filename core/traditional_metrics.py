import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Union, List
from config.settings import Config

class TraditionalMetrics:
    """Traditional image quality evaluation metrics"""
    
    def __init__(self):
        pass
    
    def preprocess_images(self, image1: Union[Image.Image, np.ndarray],
                         image2: Union[Image.Image, np.ndarray]) -> tuple:
        """Preprocess images to ensure consistent format"""
        # 转换为numpy数组
        if isinstance(image1, Image.Image):
            img1 = np.array(image1)
        else:
            img1 = image1.copy()
            
        if isinstance(image2, Image.Image):
            img2 = np.array(image2)
        else:
            img2 = image2.copy()
        
        # 确保图像尺寸一致
        if img1.shape != img2.shape:
            h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        # 确保数据类型一致
        if img1.dtype != img2.dtype:
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
        
        return img1, img2
    
    def compute_ssim(self, image1: Union[Image.Image, np.ndarray],
                     image2: Union[Image.Image, np.ndarray],
                     multichannel: bool = True) -> float:
        """计算结构相似性指数 (SSIM)"""
        img1, img2 = self.preprocess_images(image1, image2)
        
        # 如果是彩色图像
        if len(img1.shape) == 3 and multichannel:
            # 转换数据范围到[0,1]
            if img1.max() > 1:
                img1 = img1 / 255.0
                img2 = img2 / 255.0
            
            ssim_score = ssim(img1, img2, 
                             data_range=1.0,
                             channel_axis=2)
        else:
            # 灰度图像
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            if img1.max() > 1:
                img1 = img1 / 255.0
                img2 = img2 / 255.0
            
            ssim_score = ssim(img1, img2, data_range=1.0)
        
        return max(0, ssim_score)
    
    def compute_psnr(self, image1: Union[Image.Image, np.ndarray],
                     image2: Union[Image.Image, np.ndarray]) -> float:
        """计算峰值信噪比 (PSNR)"""
        img1, img2 = self.preprocess_images(image1, image2)
        
        # 转换数据范围到[0,1]
        if img1.max() > 1:
            img1 = img1 / 255.0
            img2 = img2 / 255.0
        
        try:
            psnr_score = psnr(img1, img2, data_range=1.0)
            return max(0, psnr_score)
        except ZeroDivisionError:
            # 如果图像完全相同，返回无穷大的近似值
            return 100.0
    
    def compute_mse(self, image1: Union[Image.Image, np.ndarray],
                    image2: Union[Image.Image, np.ndarray]) -> float:
        """计算均方误差 (MSE)"""
        img1, img2 = self.preprocess_images(image1, image2)
        
        # 转换数据范围到[0,1]
        if img1.max() > 1:
            img1 = img1 / 255.0
            img2 = img2 / 255.0
        
        mse = np.mean((img1 - img2) ** 2)
        return float(mse)
    
    def compute_mae(self, image1: Union[Image.Image, np.ndarray],
                    image2: Union[Image.Image, np.ndarray]) -> float:
        """计算平均绝对误差 (MAE)"""
        img1, img2 = self.preprocess_images(image1, image2)
        
        # 转换数据范围到[0,1]
        if img1.max() > 1:
            img1 = img1 / 255.0
            img2 = img2 / 255.0
        
        mae = np.mean(np.abs(img1 - img2))
        return float(mae)
    
    def evaluate_traditional_metrics(self, reference_image: Union[Image.Image, np.ndarray],
                                   generated_image: Union[Image.Image, np.ndarray]) -> dict:
        """计算所有传统图像质量指标"""
        # 计算各项指标
        ssim_score = self.compute_ssim(reference_image, generated_image)
        psnr_score = self.compute_psnr(reference_image, generated_image)
        mse_score = self.compute_mse(reference_image, generated_image)
        mae_score = self.compute_mae(reference_image, generated_image)
        
        # 判断是否通过阈值
        ssim_threshold = Config.THRESHOLDS['ssim_threshold']
        psnr_threshold = Config.THRESHOLDS['psnr_threshold']
        
        ssim_passed = ssim_score >= ssim_threshold
        psnr_passed = psnr_score >= psnr_threshold
        
        # 计算综合评分
        ssim_score_normalized = ssim_score * 100
        psnr_score_normalized = min(100, (psnr_score / 50) * 100)  # 将PSNR归一化到0-100
        
        return {
            'ssim': {
                'value': ssim_score,
                'score': ssim_score_normalized,
                'passed': ssim_passed,
                'threshold': ssim_threshold,
                'evaluation': 'excellent' if ssim_score >= 0.9 else 
                             'good' if ssim_score >= 0.7 else
                             'fair' if ssim_score >= 0.5 else 'poor'
            },
            'psnr': {
                'value': psnr_score,
                'score': psnr_score_normalized,
                'passed': psnr_passed,
                'threshold': psnr_threshold,
                'evaluation': 'excellent' if psnr_score >= 30 else 
                             'good' if psnr_score >= 20 else
                             'fair' if psnr_score >= 15 else 'poor'
            },
            'mse': {
                'value': mse_score,
                'evaluation': 'excellent' if mse_score <= 0.01 else 
                             'good' if mse_score <= 0.05 else
                             'fair' if mse_score <= 0.1 else 'poor'
            },
            'mae': {
                'value': mae_score,
                'evaluation': 'excellent' if mae_score <= 0.05 else 
                             'good' if mae_score <= 0.1 else
                             'fair' if mae_score <= 0.2 else 'poor'
            }
        }
    
    def batch_traditional_evaluation(self, reference_images: List[Union[Image.Image, np.ndarray]],
                                   generated_images: List[Union[Image.Image, np.ndarray]]) -> List[dict]:
        """批量评估传统指标"""
        results = []
        
        for ref_img, gen_img in zip(reference_images, generated_images):
            result = self.evaluate_traditional_metrics(ref_img, gen_img)
            results.append(result)
        
        return results