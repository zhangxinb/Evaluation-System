import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance
from typing import Union, Tuple, Optional
from config.settings import Config

class ImagePreprocessor:
    """图像预处理类"""
    
    def __init__(self):
        self.target_size = Config.IMAGE_SIZE
        self.max_size = Config.MAX_IMAGE_SIZE
    
    def resize_image(self, image: Union[Image.Image, np.ndarray], 
                    size: Tuple[int, int] = None,
                    keep_aspect_ratio: bool = True) -> Image.Image:
        """调整图像尺寸"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if size is None:
            size = self.target_size
        
        if keep_aspect_ratio:
            # 保持宽高比
            image.thumbnail(size, Image.Resampling.LANCZOS)
        else:
            # 直接调整到指定尺寸
            image = image.resize(size, Image.Resampling.LANCZOS)
        
        return image
    
    def pad_to_square(self, image: Image.Image, 
                     target_size: int = None,
                     fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """将图像填充为正方形"""
        if target_size is None:
            target_size = max(self.target_size)
        
        # 计算填充
        width, height = image.size
        max_dim = max(width, height)
        
        # 创建正方形画布
        new_image = Image.new('RGB', (max_dim, max_dim), fill_color)
        
        # 居中粘贴原图像
        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        # 调整到目标尺寸
        new_image = new_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return new_image
    
    def normalize_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """归一化图像到[0,1]范围"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # 转换为float类型
        image = image.astype(np.float32)
        
        # 归一化到[0,1]
        if image.max() > 1:
            image = image / 255.0
        
        return image
    
    def enhance_image_quality(self, image: Image.Image,
                            brightness: float = 1.0,
                            contrast: float = 1.0,
                            saturation: float = 1.0,
                            sharpness: float = 1.0) -> Image.Image:
        """增强图像质量"""
        enhanced = image
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness)
        
        return enhanced
    
    def auto_orient(self, image: Image.Image) -> Image.Image:
        """自动调整图像方向（基于EXIF数据）"""
        try:
            return ImageOps.exif_transpose(image)
        except:
            return image
    
    def crop_center(self, image: Image.Image, 
                   crop_size: Tuple[int, int]) -> Image.Image:
        """中心裁剪"""
        width, height = image.size
        crop_width, crop_height = crop_size
        
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        return image.crop((left, top, right, bottom))
    
    def remove_background(self, image: Image.Image, 
                         background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """简单的背景移除（基于颜色阈值）"""
        # 转换为RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        data = np.array(image)
        
        # 创建掩码（假设背景为白色）
        mask = np.all(data[:, :, :3] > [240, 240, 240], axis=2)
        
        # 设置透明度
        data[mask] = [255, 255, 255, 0]
        
        return Image.fromarray(data, 'RGBA')
    
    def preprocess_for_evaluation(self, image: Union[Image.Image, np.ndarray],
                                 target_size: Tuple[int, int] = None,
                                 normalize: bool = True,
                                 enhance: bool = False) -> Union[Image.Image, np.ndarray]:
        """为评估预处理图像"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 自动调整方向
        image = self.auto_orient(image)
        
        # 确保是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 限制最大尺寸
        if max(image.size) > self.max_size:
            ratio = self.max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 调整尺寸
        if target_size:
            image = self.resize_image(image, target_size)
        
        # 图像增强
        if enhance:
            image = self.enhance_image_quality(image)
        
        # 归一化
        if normalize:
            return self.normalize_image(image)
        
        return image
    
    def batch_preprocess(self, images: list,
                        target_size: Tuple[int, int] = None,
                        normalize: bool = True,
                        enhance: bool = False) -> list:
        """批量预处理图像"""
        processed_images = []
        
        for image in images:
            processed = self.preprocess_for_evaluation(
                image, target_size, normalize, enhance
            )
            processed_images.append(processed)
        
        return processed_images
    
    def validate_image(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """验证图像是否有效"""
        try:
            if isinstance(image, np.ndarray):
                if image.size == 0:
                    return False
                if len(image.shape) not in [2, 3]:
                    return False
                if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                    return False
            elif isinstance(image, Image.Image):
                if image.size[0] == 0 or image.size[1] == 0:
                    return False
            else:
                return False
            
            return True
        except:
            return False