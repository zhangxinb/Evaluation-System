# 图像一致性评估系统 🎯

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-orange)](https://gradio.app/)

这是一个全面的AI图像生成质量评估系统，专注于评估生成图像在多个维度上的一致性。系统集成了最新的深度学习技术和传统图像处理方法，提供客观、全面的图像质量评估。

![System Demo](docs/demo.gif)

## ✨ 核心特点

### 🔬 多维度评估指标
- **CLIP语义一致性**: 评估生成图像与文本提示词的语义匹配程度
- **身份一致性**: 使用人脸识别技术评估人物身份的保持程度
- **感知相似度**: 使用LPIPS评估基于人类视觉感知的图像相似度
- **传统图像质量指标**: SSIM、PSNR、MSE、MAE等经典指标

### 🏗️ 系统架构
- **模块化设计**: 便于扩展新的评估指标和功能
- **统一评分框架**: 标准化的分数归一化和聚合机制
- **智能权重分配**: 可配置的指标权重和阈值设置
- **批量处理支持**: 高效的大规模图像评估能力

### 🎨 用户界面
- **交互式Web界面**: 基于Gradio的现代化操作界面
- **实时可视化**: 动态图表和评估结果展示
- **直观报告生成**: 自动生成详细的评估报告
- **多格式导出**: 支持JSON、CSV、Markdown等格式

## 🚀 快速开始

### 自动安装（推荐）

```bash
# 1. 克隆项目
git clone <repository-url>
cd evaluation-system

# 2. 运行自动安装脚本
python install.py

# 3. 测试系统
python test_system.py

# 4. 启动应用
python app.py
```

### 手动安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. （可选）安装GPU支持
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 启动应用
python app.py
```

### 快速演示

```bash
# 查看功能演示和系统状态
python demo.py
```

## 📖 详细使用指南

### 启动系统

```bash
# 基本启动
python app.py

# 自定义配置启动
python app.py --host 0.0.0.0 --port 8080 --share

# 启用调试模式
python app.py --debug
```

### Web界面使用

1. **单图像评估**
   - 上传生成图像（必需）
   - 上传参考图像（可选，用于相似度评估）
   - 输入文本提示词（可选，用于CLIP评估）
   - 选择评估指标
   - 点击"开始评估"查看结果

2. **批量评估**
   - 上传多个图像文件
   - 提供对应的提示词列表
   - 选择评估指标
   - 下载评估报告和结果文件

3. **系统设置**
   - 调整评估阈值
   - 配置模型参数
   - 自定义权重设置

### API使用示例

```python
from evaluation_manager import EvaluationManager
from PIL import Image

# 创建评估管理器
manager = EvaluationManager()
manager.initialize_evaluators()

# 加载图像
generated_img = Image.open("generated_image.jpg")
reference_img = Image.open("reference_image.jpg")

# 全面评估
results = manager.evaluate_comprehensive(
    generated_image=generated_img,
    reference_image=reference_img,
    prompt="a beautiful landscape painting",
    metrics=['clip', 'identity', 'perceptual', 'traditional']
)

# 查看结果
print(f"综合评分: {results['overall_score']:.1f}/100")
print(f"总体评价: {results['overall_evaluation']}")

# 生成报告
report = manager.generate_evaluation_report(results)
print(report)

# 保存结果
manager.save_evaluation_results(results, "evaluation_results.json")
```

## 📁 项目结构

```
evaluation_system/
├── 📂 core/                    # 核心评估模块
│   ├── 🧠 clip_evaluator.py      # CLIP语义一致性评估
│   ├── 👤 identity_evaluator.py   # 身份一致性评估
│   ├── 👁️ perceptual_evaluator.py # 感知相似度评估
│   └── 📊 traditional_metrics.py  # 传统图像质量指标
├── 📂 utils/                   # 工具函数模块
│   ├── 📥 data_loader.py          # 数据加载和管理
│   ├── 🔧 preprocessor.py         # 图像预处理
│   └── 📈 visualizer.py           # 可视化工具
├── 📂 dashboard/               # 用户界面模块
│   └── 🌐 gradio_app.py          # Gradio Web界面
├── 📂 config/                  # 配置文件
│   └── ⚙️ settings.py            # 系统设置
├── 📂 docs/                    # 文档资源
├── 🎯 evaluation_manager.py    # 统一评估管理器
├── 🚀 app.py                  # 主应用入口
├── 🔧 install.py              # 自动安装脚本
├── 🎪 demo.py                 # 功能演示脚本
├── 📋 requirements.txt        # 依赖包列表
├── ⚙️ config.yaml             # 配置文件
├── 📚 USER_MANUAL.md          # 用户手册
└── 📖 README.md               # 项目说明
```

## 🔬 评估指标详解

### CLIP语义一致性
- **评分范围**: 0.0 - 1.0
- **推荐阈值**: ≥ 0.7
- **权重**: 30%
- **用途**: 评估图像与文本描述的语义匹配程度

### 身份一致性  
- **评分范围**: 0.0 - 1.0
- **推荐阈值**: ≥ 0.6
- **权重**: 25%
- **技术**: FaceNet + MTCNN
- **用途**: 评估人物身份的保持程度

### 感知相似度
- **评分范围**: 0.0 - 1.0 (LPIPS距离: 0.0 - 1.0)
- **推荐阈值**: 距离 ≤ 0.3
- **权重**: 25%
- **技术**: LPIPS (AlexNet/VGG)
- **用途**: 基于人类视觉感知的相似度评估

### 传统图像质量指标
- **SSIM**: 结构相似性指数 (0.0 - 1.0, 阈值 ≥ 0.8)
- **PSNR**: 峰值信噪比 (dB, 阈值 ≥ 20)
- **MSE**: 均方误差
- **MAE**: 平均绝对误差

## 🎯 适用场景

- 🎨 **AI艺术生成**: 评估Stable Diffusion、DALL-E等模型生成质量
- 👤 **人像生成**: 检验人脸生成的身份一致性和质量
- 🖼️ **图像编辑**: 评估图像修复、风格迁移等编辑效果
- 🔬 **学术研究**: 为图像生成研究提供客观评估工具
- 🏭 **工业应用**: 产品图像生成的质量控制
- 📚 **数据集评估**: 大规模图像数据集的质量筛选

## ⚙️ 系统要求

- **Python**: 3.7+
- **内存**: 8GB+ (推荐16GB)
- **显卡**: 可选，NVIDIA GPU with CUDA支持
- **存储**: 5GB+ 可用空间
- **网络**: 首次运行需要下载模型

## 🛠️ 高级配置

### 自定义配置

编辑 `config.yaml` 文件自定义系统行为：

```yaml
# 调整评估权重
metric_weights:
  clip_similarity: 0.30
  identity_similarity: 0.25
  lpips_similarity: 0.25
  ssim_similarity: 0.10
  psnr_similarity: 0.10

# 设置评估阈值
thresholds:
  clip_threshold: 0.70
  identity_threshold: 0.60
  lpips_threshold: 0.30
```

### GPU加速

```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 安装CUDA版本PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 性能基准

| 配置 | 单图评估时间 | 批量处理能力 | 内存使用 |
|------|-------------|-------------|----------|
| CPU (Intel i7) | ~15秒 | 10图像/分钟 | ~4GB |
| GPU (RTX 3080) | ~5秒 | 30图像/分钟 | ~6GB |
| GPU (RTX 4090) | ~3秒 | 50图像/分钟 | ~8GB |

## 🔧 故障排除

### 常见问题

**Q: CUDA内存不足**
```bash
A: 降低图像分辨率或使用CPU模式
export CUDA_VISIBLE_DEVICES=""
```

**Q: 模型下载失败**
```bash
A: 检查网络连接，或使用镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name
```

**Q: 人脸检测失败**
```bash
A: 确保图像包含清晰的人脸，调整检测阈值
```

详细故障排除指南请查看 [USER_MANUAL.md](USER_MANUAL.md)

## 🤝 贡献指南

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 开发环境设置

```bash
# 克隆仓库
git clone <repository-url>
cd evaluation-system

# 创建开发环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [OpenAI CLIP](https://github.com/openai/CLIP) - 语义一致性评估
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity) - 感知相似度评估  
- [FaceNet PyTorch](https://github.com/timesler/facenet-pytorch) - 人脸识别
- [Gradio](https://gradio.app/) - Web界面框架
- [Scikit-Image](https://scikit-image.org/) - 传统图像指标

## 📞 支持与联系

- 📧 **邮箱**: [contact@example.com](mailto:contact@example.com)
- 💬 **讨论**: GitHub Discussions
- 🐛 **问题报告**: GitHub Issues
- 📚 **文档**: [完整文档](https://docs.example.com)

---

如果这个项目对您有帮助，请给我们一个 ⭐ star！