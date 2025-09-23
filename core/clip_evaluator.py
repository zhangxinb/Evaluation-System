import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Union, List
from config.settings import Config

# Try different CLIP implementations
try:
    import clip
    CLIP_AVAILABLE = True
    CLIP_TYPE = "openai"
except ImportError:
    try:
        import open_clip
        CLIP_AVAILABLE = True 
        CLIP_TYPE = "open_clip"
    except ImportError:
        CLIP_AVAILABLE = False
        CLIP_TYPE = None

class CLIPEvaluator:
    """CLIP model evaluator for computing semantic consistency between images and text"""
    
    def __init__(self, model_name: str = None):
        self.device = Config.DEVICE
        self.model_name = model_name or Config.CLIP_MODEL_NAME
        
        if not CLIP_AVAILABLE:
            raise ImportError("No CLIP implementation available. Please install either:\n"
                            "- pip install git+https://github.com/openai/CLIP.git\n"
                            "- pip install open-clip-torch")
        
        if CLIP_TYPE == "openai":
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        elif CLIP_TYPE == "open_clip":
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, device=self.device
            )
        else:
            raise ImportError("No compatible CLIP implementation found")
        
    def encode_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Encode image to feature vector"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features
    
    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text to feature vector"""
        if isinstance(text, str):
            text = [text]
        
        if CLIP_TYPE == "openai":
            text_input = clip.tokenize(text).to(self.device)
        elif CLIP_TYPE == "open_clip":
            text_input = open_clip.tokenize(text).to(self.device)
        else:
            raise RuntimeError("No CLIP model available")
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def compute_similarity(self, image: Union[Image.Image, np.ndarray], 
                          text: str) -> float:
        """计算图像与文本的相似度"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        similarity = torch.cosine_similarity(image_features, text_features).item()
        return max(0, similarity)  # 确保相似度非负
    
    def batch_similarity(self, images: List[Union[Image.Image, np.ndarray]], 
                        texts: List[str]) -> List[float]:
        """批量计算图像与文本的相似度"""
        similarities = []
        
        for image, text in zip(images, texts):
            similarity = self.compute_similarity(image, text)
            similarities.append(similarity)
        
        return similarities
    
    def evaluate_prompt_consistency(self, generated_image: Union[Image.Image, np.ndarray],
                                  prompt: str) -> dict:
        """评估生成图像与提示词的一致性"""
        similarity = self.compute_similarity(generated_image, prompt)
        
        # 计算评分 (0-100)
        score = similarity * 100
        
        # 判断是否通过阈值
        threshold = Config.THRESHOLDS['clip_threshold']
        passed = similarity >= threshold
        
        return {
            'similarity': similarity,
            'score': score,
            'passed': passed,
            'threshold': threshold,
            'evaluation': 'excellent' if similarity >= 0.8 else 
                         'good' if similarity >= 0.6 else
                         'fair' if similarity >= 0.4 else 'poor'
        }