# 数据结构与存储说明

**更新日期**: 2025-10-03  
**状态**: ✅ 已标准化

---

## 📁 目录结构

```
Evaluation System/
├── evaluation_data/              ← 主数据目录 (所有评估数据)
│   ├── evaluation_results.csv    ← CSV格式结果
│   ├── evaluation_results.json   ← JSON格式结果(含完整数据)
│   └── visualizations/           ← 可视化图表
│       ├── metric_comparison.png
│       ├── hypothesis_validation.png
│       ├── metric_heatmap.png
│       ├── score_distribution.png
│       └── metric_directions.png
│
├── app.py                        ← Web评估界面
├── data_logger.py                ← 数据记录模块
├── visualizer.py                 ← 可视化模块
├── analyze_data.py               ← 数据分析脚本
└── generate_plots.py             ← 图表生成脚本
```

**重要**: ~~`test_data/`~~ 已删除,所有数据统一使用 `evaluation_data/`

---

## 📊 CSV 数据结构

### 列定义 (共20列)

| # | 列名 | 类型 | 说明 |
|---|------|------|------|
| 1 | `timestamp` | ISO datetime | 记录时间戳 |
| 2 | `sample_id` | string | 唯一样本ID (格式: Category_YYYYMMDD_HHMMSS_mmm_rnd) |
| 3 | `sample_category` | enum | 类别: Basic/Attribute/Boundary |
| 4 | `image1_name` | string | 第一张图片名称 |
| 5 | `image2_name` | string | 第二张图片名称 |
| 6 | `deepface_similarity` | float | DeepFace身份相似度 (0-1, ↑) |
| 7 | `deepface_confidence` | float | DeepFace置信度 (0-1) |
| 8 | `deepface_decision` | string | 判断结果: Same Person/Different Person |
| 9 | `models_used` | int | 使用的模型数量 |
| 10 | `clip_image_similarity` | float | CLIP语义相似度 (0-1, ↑) |
| 11 | `lpips_distance` | float | LPIPS感知距离 (0-1.5+, ↓) |
| 12 | `ssim_whole_image` | float | SSIM结构相似度 (0-1, ↑) |
| 13 | `ssim_face_region` | float | 面部区域SSIM (预留) |
| 14 | `psnr` | float | 峰值信噪比 (dB, ↑) |
| 15 | `mse` | float | 均方误差 (↓) |
| 16 | `histogram_correlation` | float | 直方图相关性 (0-1, ↑) |
| 17 | `final_score` | float | 最终综合分数 (0-1, ↑) |
| 18 | `consistency_level` | string | 一致性等级 |
| 19 | `overall_confidence` | float | 整体置信度 (0-1) |
| 20 | `notes` | string | 备注信息 |

### 核心评估指标 (3个)

| 指标 | 列名 | 方向 | 范围 | 说明 |
|------|------|------|------|------|
| **DeepFace** | `deepface_similarity` | ↑ | 0.6-1.0 | 人脸身份相似度 |
| **CLIP-I** | `clip_image_similarity` | ↑ | 0.7-1.0 | 语义图像相似度 |
| **LPIPS** | `lpips_distance` | ↓ | 0.0-1.5 | 感知距离 |

**LPIPS 距离解释**:
- `0.05-0.20`: 相同人物,相似条件 ✅
- `0.20-0.50`: 相同人物,不同条件 ⚠️
- `0.50+`: 不同人物 ❌

---

## 🔧 程序配置

### 所有程序默认使用 `evaluation_data`

| 程序文件 | 数据目录配置 | 状态 |
|---------|-------------|------|
| `app.py` | `data_dir="evaluation_data"` | ✅ |
| `data_logger.py` | `data_dir="evaluation_data"` | ✅ |
| `visualizer.py` | `data_dir="evaluation_data"` | ✅ |
| `analyze_data.py` | `data_dir="evaluation_data"` | ✅ |
| `generate_plots.py` | `data_path="evaluation_data/..."` | ✅ |

### Sample ID 生成规则

**格式**: `{Category}_{YYYYMMDD}_{HHMMSS}_{mmm}_{rnd}`

**示例**: `Basic_20251003_133929_147_640`

**组成部分**:
- `Category`: Basic/Attribute/Boundary
- `YYYYMMDD`: 日期
- `HHMMSS`: 时间
- `mmm`: 毫秒(前3位)
- `rnd`: 随机数(100-999)

**唯一性保证**: 时间戳 + 微秒 + 随机数 = 确保即使同一毫秒内也不会重复

---

## 📝 使用工作流

### 1. 数据采集

```bash
# 启动Web界面进行评估
python app.py

# 访问: http://localhost:7860
# 上传图片对,系统自动记录到 evaluation_data/
```

**数据自动保存到**:
- `evaluation_data/evaluation_results.csv`
- `evaluation_data/evaluation_results.json`

---

### 2. 数据分析

```bash
# 查看数据摘要和统计
python analyze_data.py

# 功能:
# - 显示样本分布
# - 各类别详细统计
# - 假设验证结果
# - 生成可视化图表
```

**输出**:
- 控制台统计信息
- `evaluation_data/visualizations/*.png` (可选)

---

### 3. 生成图表

```bash
# 手动生成特定图表
python generate_plots.py

# 菜单选项:
# 1. Metric Comparison (指标对比箱线图)
# 2. Hypothesis Validation (假设验证柱状图)
# 3. Metric Heatmap (指标热力图)
# 4. Score Distribution (分数分布图)
# 5. Metric Directions (指标方向参考)
# 6. All plots (生成所有图表)
```

**输出位置**: `evaluation_data/visualizations/`

---

### 4. 测试功能

```bash
# 测试可视化功能(生成模拟数据)
python visualizer.py

# 测试数据记录功能
python data_logger.py
```

**注意**: 测试也会写入 `evaluation_data/`,与真实数据混合存储

---

## 🎯 数据类别说明

### Basic Consistency (基础一致性)

**预期**: 所有指标都应该高
- DeepFace: > 0.7
- CLIP: > 0.7
- LPIPS Distance: < 0.3
- SSIM: > 0.7

**测试场景**: 相同角色,相似姿势/表情/光照

---

### Attribute Consistency (属性一致性)

**预期**: 身份高,结构低
- DeepFace: > 0.7 ✅ (身份识别准确)
- CLIP: 0.6-0.8 ⚠️ (语义有变化)
- LPIPS Distance: 0.3-0.5 ⚠️ (感知有差异)
- SSIM: < 0.5 ❌ (结构差异大)

**测试场景**: 相同角色,不同服装/发型/配饰

---

### Boundary Cases (边界情况)

**预期**: 所有指标都应该低
- DeepFace: < 0.5
- CLIP: < 0.5
- LPIPS Distance: > 0.5
- SSIM: < 0.4

**测试场景**: 不同角色,或极端变化

---

## 📈 数据质量检查

### 检查CSV结构

```powershell
# PowerShell命令
$csv = Import-Csv "evaluation_data\evaluation_results.csv"

# 检查记录数
Write-Host "总记录数: $($csv.Count)"

# 检查列结构
$csv[0].PSObject.Properties.Name

# 检查唯一性
$unique = ($csv.sample_id | Select-Object -Unique).Count
if($unique -eq $csv.Count) {
    Write-Host "✅ 所有ID唯一!"
} else {
    Write-Host "❌ 存在重复ID!"
}

# 检查分类分布
$csv | Group-Object sample_category | Format-Table Count, Name
```

### 检查LPIPS数据

```powershell
# 按类别统计LPIPS距离
$csv | Group-Object sample_category | ForEach-Object {
    $lpips = $_.Group.lpips_distance | ForEach-Object {[double]$_}
    $min = ($lpips | Measure-Object -Minimum).Minimum
    $max = ($lpips | Measure-Object -Maximum).Maximum
    $avg = ($lpips | Measure-Object -Average).Average
    Write-Host "$($_.Name): Min=$([math]::Round($min,3)) Max=$([math]::Round($max,3)) Avg=$([math]::Round($avg,3))"
}
```

**预期范围**:
- Basic: 0.05-0.25 (平均 ~0.15)
- Attribute: 0.25-0.50 (平均 ~0.35)
- Boundary: 0.50-0.90 (平均 ~0.65)

---

## 🔄 数据清理

### 删除所有数据

```bash
# 删除CSV和JSON
Remove-Item "evaluation_data\evaluation_results.*" -Force

# 删除可视化图表
Remove-Item "evaluation_data\visualizations\*.png" -Force
```

### 重新初始化

```bash
# 运行任何程序都会自动重新创建CSV
python app.py
# 或
python visualizer.py
```

---

## 📋 数据导出

### 导出到Excel (可选)

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('evaluation_data/evaluation_results.csv')

# 导出为Excel
df.to_excel('evaluation_data/results.xlsx', index=False)
```

### 导出统计摘要

```python
from data_logger import EvaluationDataLogger

logger = EvaluationDataLogger(data_dir="evaluation_data")
stats = logger.get_summary_statistics()

# 保存为JSON
import json
with open('evaluation_data/statistics.json', 'w') as f:
    json.dump(stats, indent=2)
```

---

## ✅ 检查清单

在提交论文数据前,确保:

- [ ] 所有数据在 `evaluation_data/` 目录
- [ ] CSV文件包含20列,无缺失
- [ ] 所有 `sample_id` 唯一
- [ ] 三个类别都有足够样本 (建议每类 ≥15)
- [ ] LPIPS距离范围合理 (Basic<0.3, Boundary>0.5)
- [ ] 生成了所有5个可视化图表
- [ ] 统计结果支持研究假设

---

## 🆘 常见问题

### Q: CSV文件损坏怎么办?

**A**: 删除CSV,保留JSON,从JSON重建:

```python
import json
import pandas as pd

# 读取JSON
with open('evaluation_data/evaluation_results.json') as f:
    data = json.load(f)

# 提取基础字段(不含full_results)
records = [{k: v for k, v in item.items() if k != 'full_results'} 
           for item in data]

# 保存为CSV
df = pd.DataFrame(records)
df.to_csv('evaluation_data/evaluation_results.csv', index=False)
```

### Q: 如何合并多个CSV文件?

**A**: 使用pandas合并:

```python
import pandas as pd

df1 = pd.read_csv('evaluation_data/evaluation_results.csv')
df2 = pd.read_csv('backup/old_results.csv')

# 合并并去重
df_merged = pd.concat([df1, df2]).drop_duplicates(subset=['sample_id'])
df_merged.to_csv('evaluation_data/evaluation_results.csv', index=False)
```

### Q: 如何筛选特定类别的数据?

**A**: 使用pandas筛选:

```python
import pandas as pd

df = pd.read_csv('evaluation_data/evaluation_results.csv')

# 只保留Basic类别
df_basic = df[df['sample_category'] == 'Basic']
df_basic.to_csv('evaluation_data/basic_only.csv', index=False)
```

---

**文档版本**: 1.0  
**最后更新**: 2025-10-03  
**维护者**: Evaluation System Team
