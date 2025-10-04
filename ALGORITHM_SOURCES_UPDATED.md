# 评估算法来源说明文档 (更新版)

> **重要更新 (2025-10-03)**: 移除了自建的LPIPS相似度转换公式,现在只使用LPIPS库原始输出的距离值

---

## 🎯 核心问题

在论文写作中,您需要明确说明:每个评估指标的计算是**基于现有库方法**还是**自建公式**?

---

## 📊 评估指标总览

本系统使用以下核心评估指标:

1. **DeepFace Similarity (↑)** - 人脸身份相似度
2. **CLIP-I Similarity (↑)** - 语义图像相似度  
3. **LPIPS Distance (↓)** - 感知距离
4. 传统指标: SSIM, PSNR, MSE, Histogram Correlation

---

## 详细拆解

### 1️⃣ **DeepFace Similarity** (身份相似度)

#### 🔧 实现来源
- **基于已有方法库**: ✅ 使用 **[DeepFace](https://github.com/serengil/deepface)** 开源库
- **版本**: deepface >= 0.0.79
- **核心模型**: Facenet512, ArcFace, Facenet

#### 📐 计算方法
```python
# DeepFace 库提供的部分:
from deepface import DeepFace
result = DeepFace.verify(img1, img2, model_name='Facenet512')
distance = result['distance']  # 余弦距离
similarity = 1.0 - distance    # 简单转换

# 自建的多模型共识机制:
similarities = []
for model in ['Facenet512', 'ArcFace', 'Facenet']:
    similarity = calculate_single_model(model)
    similarities.append(similarity)
final_similarity = weighted_average(similarities)  # 自建加权平均
```

#### ✏️ 自定义部分
1. **多模型共识机制** - 使用3个模型的加权平均
2. **质量自适应阈值** - 根据人脸质量调整判断阈值

#### 📚 论文引用
> Serengil, S. I., & Ozpinar, A. (2020). LightFace: A hybrid deep face recognition framework. *Innovations in Intelligent Systems and Applications Conference (ASYU)*, 1-5.

#### ✅ 结论
**混合实现**: 基础模型来自库,共识机制和质量调整是自建的

---

### 2️⃣ **CLIP-I Similarity** (图像语义相似度)

#### 🔧 实现来源
- **100% 基于已有库**: ✅ 使用 **[OpenAI CLIP](https://github.com/openai/CLIP)** 官方库
- **模型**: ViT-B/32
- **无任何自建公式**

#### 📐 计算方法
```python
# 完全使用 CLIP 官方库:
import clip
model, preprocess = clip.load("ViT-B/32")

# 图像编码 (CLIP库实现)
img1_features = model.encode_image(preprocess(img1).unsqueeze(0))
img2_features = model.encode_image(preprocess(img2).unsqueeze(0))

# 余弦相似度 (PyTorch库实现)
import torch.nn.functional as F
similarity = F.cosine_similarity(img1_features, img2_features).item()
```

#### ✏️ 自定义部分
**无** - 完全使用库提供的方法

#### 📚 论文引用
> Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *International Conference on Machine Learning*, 8748-8763.

#### ✅ 结论
**纯库实现**: 100%使用OpenAI官方CLIP库,无自建修改

---

### 3️⃣ **LPIPS Distance** (感知距离)

#### 🔧 实现来源
- **基于已有方法库**: ✅ 使用 **[LPIPS](https://github.com/richzhang/PerceptualSimilarity)** 官方库
- **网络**: AlexNet backbone
- **输出**: 直接使用库的距离值 (无转换)

#### 📐 计算方法
```python
import lpips
import torchvision.transforms as transforms

# 1. LPIPS 库初始化 (库实现)
loss_fn = lpips.LPIPS(net='alex')

# 2. 图像预处理 (自建优化 - 提升效果)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 自建:提高分辨率
    transforms.ToTensor(),
    transforms.Normalize(           # 自建:ImageNet标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img1_tensor = transform(img1).unsqueeze(0)
img2_tensor = transform(img2).unsqueeze(0)

# 3. 计算距离 (LPIPS库实现)
lpips_distance = loss_fn(img1_tensor, img2_tensor).item()

# 4. 直接使用距离值,无转换!
# 距离范围: 0.0 (相同) 到 1.5+ (非常不同)
# - 相同人物,相似条件: 0.05-0.20
# - 相同人物,不同条件: 0.20-0.50  
# - 不同人物: 0.50-1.50+
```

#### ✏️ 自定义部分
1. **分辨率提升** - 使用512×512而非默认256×256 (保留更多细节)
2. **ImageNet标准化** - 使用ImageNet参数而非简单[-1,1] (更适合预训练模型)
3. **~~相似度转换公式~~** - ❌ **已移除!** 不再使用 `exp(-2d)` 转换

#### 📚 论文引用
> Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The unreasonable effectiveness of deep features as a perceptual metric. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 586-595.

#### ✅ 结论
**混合实现**: 核心距离计算来自库,预处理有自建优化 (但无自建转换公式)

---

## 📋 总结表格

| 指标 | 实现类型 | 库来源 | 自建部分 | 论文引用 |
|------|---------|--------|---------|---------|
| **DeepFace Similarity** | 混合 | DeepFace 库 | 多模型共识、质量调整 | ✅ 必须 |
| **CLIP-I Similarity** | 纯库 | OpenAI CLIP | 无 | ✅ 必须 |
| **LPIPS Distance** | 混合 | LPIPS 库 | 分辨率、标准化优化 | ✅ 必须 |

---

## 🎓 论文撰写建议

### 方法论 (Methodology) 章节应该这样写:

#### DeepFace Similarity
```
本研究采用 DeepFace 库(Serengil & Ozpinar, 2020)进行人脸身份识别。
为提高准确性,我们设计了多模型共识机制,整合 Facenet512、ArcFace 和 
Facenet 三个深度学习模型的预测结果,通过加权平均获得最终相似度分数。
此外,实现了质量自适应阈值机制,根据人脸图像质量动态调整判断阈值。
```

#### CLIP-I Similarity
```
图像语义相似度采用 OpenAI 的 CLIP 模型(Radford et al., 2021)。
我们使用预训练的 ViT-B/32 模型提取图像特征向量,并通过余弦相似度
计算两张图像的语义匹配程度。该指标完全基于 CLIP 官方实现,未进行
自定义修改。
```

#### LPIPS Distance
```
感知距离采用 LPIPS(Learned Perceptual Image Patch Similarity)方法
(Zhang et al., 2018),使用 AlexNet 作为骨干网络提取深度感知特征。
为更好地保留面部细节,我们将输入图像分辨率提高到 512×512 像素,
并采用 ImageNet 标准化参数进行预处理。LPIPS 距离直接使用库输出值,
其中较低的距离表示图像在感知上更加相似。
```

---

## 🔍 关键变更说明

### ❌ 已移除的内容

之前的版本包含自建的 LPIPS 相似度转换公式:
```python
lpips_similarity = exp(-2.0 * lpips_distance)  # 已移除!
```

**移除原因**:
1. 这是自建公式,不是学术标准方法
2. 可能引起论文审稿中的质疑
3. 直接使用 LPIPS 距离更符合学术规范

### ✅ 当前实现

现在系统直接使用 LPIPS 库输出的**距离值**:
- 无需转换公式
- 完全基于已发表的学术方法
- 更易于引用和复现

**解释方向**:
- 论文中说明: "LPIPS距离越低,表示图像在感知上越相似"
- 可视化时注意标注: "LPIPS Distance (↓ lower is better)"

---

## 📁 代码位置参考

- **DeepFace**: `face_recognition.py` 第 500-750 行
- **CLIP**: `evaluator.py` 第 220-250 行  
- **LPIPS**: `evaluator.py` 第 252-340 行
- **可视化**: `visualizer.py` (已更新为使用距离而非相似度)

---

## ✅ 学术诚信检查清单

在提交论文前,确保:

- [ ] 已引用 DeepFace 论文 (Serengil & Ozpinar, 2020)
- [ ] 已引用 CLIP 论文 (Radford et al., 2021)
- [ ] 已引用 LPIPS 论文 (Zhang et al., 2018)
- [ ] 明确说明了自建的多模型共识机制
- [ ] 明确说明了质量自适应阈值算法
- [ ] 说明 LPIPS 使用的是**距离值** (非相似度)
- [ ] 说明预处理优化 (512×512, ImageNet标准化)
- [ ] 所有可视化图表正确标注了指标方向 (↑/↓)

---

**文档版本**: v2.0 - 更新于 2025-10-03
**作者**: Evaluation System Development Team
**变更**: 移除 LPIPS 相似度转换公式,改为直接使用距离值
