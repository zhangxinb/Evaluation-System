import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Tuple, Optional
import json

class DataLoader:
    """数据加载和管理类"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载单张图像"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            image = Image.open(image_path)
            
            # 转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            return None
    
    def load_images_from_directory(self, directory: str) -> List[Tuple[str, Image.Image]]:
        """从目录加载所有图像"""
        images = []
        
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return images
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # 检查文件扩展名
            _, ext = os.path.splitext(filename)
            if ext.lower() in self.supported_formats:
                image = self.load_image(file_path)
                if image is not None:
                    images.append((filename, image))
        
        return images
    
    def load_image_pairs(self, reference_dir: str, generated_dir: str) -> List[Tuple[str, Image.Image, Image.Image]]:
        """加载参考图像和生成图像对"""
        pairs = []
        
        ref_images = {name: img for name, img in self.load_images_from_directory(reference_dir)}
        gen_images = {name: img for name, img in self.load_images_from_directory(generated_dir)}
        
        # 匹配图像对
        for ref_name, ref_img in ref_images.items():
            if ref_name in gen_images:
                pairs.append((ref_name, ref_img, gen_images[ref_name]))
            else:
                print(f"未找到对应的生成图像: {ref_name}")
        
        return pairs
    
    def load_prompts_from_file(self, file_path: str) -> Dict[str, str]:
        """从文件加载提示词"""
        prompts = {}
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                if 'filename' in df.columns and 'prompt' in df.columns:
                    prompts = dict(zip(df['filename'], df['prompt']))
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if ':' in line:
                            filename, prompt = line.strip().split(':', 1)
                            prompts[filename.strip()] = prompt.strip()
        except Exception as e:
            print(f"加载提示词文件失败 {file_path}: {e}")
        
        return prompts
    
    def save_results_to_csv(self, results: List[Dict], output_path: str):
        """保存评估结果到CSV文件"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def save_results_to_json(self, results: List[Dict], output_path: str):
        """保存评估结果到JSON文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def create_evaluation_dataset(self, 
                                reference_dir: str,
                                generated_dir: str,
                                prompts_file: str = None) -> List[Dict]:
        """创建评估数据集"""
        dataset = []
        
        # 加载图像对
        image_pairs = self.load_image_pairs(reference_dir, generated_dir)
        
        # 加载提示词（如果提供）
        prompts = {}
        if prompts_file and os.path.exists(prompts_file):
            prompts = self.load_prompts_from_file(prompts_file)
        
        # 创建数据集
        for filename, ref_img, gen_img in image_pairs:
            item = {
                'filename': filename,
                'reference_image': ref_img,
                'generated_image': gen_img,
                'prompt': prompts.get(filename, "")
            }
            dataset.append(item)
        
        return dataset
    
    def validate_dataset(self, dataset: List[Dict]) -> bool:
        """验证数据集完整性"""
        if not dataset:
            print("数据集为空")
            return False
        
        required_keys = ['filename', 'reference_image', 'generated_image']
        
        for i, item in enumerate(dataset):
            for key in required_keys:
                if key not in item:
                    print(f"数据集第{i}项缺少必需字段: {key}")
                    return False
                
            if item['reference_image'] is None or item['generated_image'] is None:
                print(f"数据集第{i}项包含无效图像")
                return False
        
        print(f"数据集验证通过，共{len(dataset)}项")
        return True