# 📁 项目文件结构（清理后）

## 🗂️ 核心文件结构

```
Evaluation System/
├── 🚀 核心应用
│   ├── app.py                          # 主应用程序
│   ├── evaluation_manager.py           # 评估管理器
│   └── launch_simple_amd.py           # AMD优化启动器 ⭐
│
├── 🔧 AMD优化模块
│   ├── amd_identity_evaluator.py      # AMD身份评估器
│   ├── install_amd_optimized.py       # AMD依赖安装脚本
│   └── test_amd_performance.py        # AMD性能测试
│
├── 📦 核心模块
│   ├── core/                          # 评估算法
│   │   ├── __init__.py
│   │   ├── clip_evaluator.py         # CLIP语义评估
│   │   ├── perceptual_evaluator.py   # 感知相似度评估
│   │   └── traditional_metrics.py    # 传统图像指标
│   │
│   ├── utils/                         # 工具函数
│   │   ├── __init__.py
│   │   ├── data_loader.py            # 数据加载
│   │   ├── preprocessor.py           # 图像预处理
│   │   └── visualizer.py             # 结果可视化
│   │
│   ├── config/                        # 配置文件
│   └── dashboard/                     # Web界面组件
│
├── 🤖 模型存储
│   └── models/                        # 预训练模型存储
│
├── 📚 文档
│   ├── README.md                      # 主说明文档
│   ├── README_AMD.md                  # AMD专用说明 ⭐
│   └── AMD_OPTIMIZATION_COMPLETE.md   # AMD优化完成报告 ⭐
│
├── ⚙️ 配置
│   ├── config.yaml                    # 系统配置
│   └── requirements.txt               # Python依赖
│
└── 🔧 环境
    └── .venv/                         # Python虚拟环境
```

## 🎯 主要文件说明

### 启动文件 ⭐
- **launch_simple_amd.py** - AMD 780M优化启动器（推荐使用）

### AMD专用文件 ⭐
- **amd_identity_evaluator.py** - 针对AMD优化的身份评估
- **install_amd_optimized.py** - 一键安装AMD兼容依赖
- **test_amd_performance.py** - AMD系统性能测试

### 核心功能
- **app.py** - 主应用程序
- **evaluation_manager.py** - 评估逻辑管理

### 文档
- **README_AMD.md** - AMD系统使用指南
- **AMD_OPTIMIZATION_COMPLETE.md** - 完整优化报告

## 🚀 快速启动

```bash
# 进入项目目录
cd "c:\Users\ZX\OneDrive - LUT University\桌面\Thesis\Evaluation System"

# 启动AMD优化版本
python launch_simple_amd.py
```

## 📊 文件统计

- **总文件数**: 约16个核心文件
- **代码文件**: 8个Python文件
- **文档文件**: 3个Markdown文件
- **配置文件**: 2个配置文件
- **AMD专用**: 3个优化文件

所有临时文件、重复文件和缓存文件已清理完毕！