#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整数据流:评估 -> 记录 -> 验证
"""

import numpy as np
import cv2
import sys
import io

# 设置输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from evaluator import CompatibleEvaluationSystem
from data_logger import EvaluationDataLogger
import pandas as pd

print("\n" + "="*70)
print("TEST DATA FLOW")
print("="*70 + "\n")

# 1. 初始化系统
print("1. Initialize evaluation system...")
evaluator = CompatibleEvaluationSystem()

# 2. 创建测试图片 (不同的灰度)
print("2. Create test images...")
img1 = np.ones((256, 256, 3), dtype=np.uint8) * 100  # 灰色
img2 = np.ones((256, 256, 3), dtype=np.uint8) * 150  # 不同的灰色

# 3. 运行评估
print("3. Run evaluation...")
results = evaluator.evaluate_character_consistency(img1, img2)

# 4. 检查结果
print("\n4. Check evaluation results:")
print(f"   Identity_Similarity: {results.get('Identity_Similarity', 'NOT FOUND')}")
print(f"   CLIP_Similarity: {results.get('CLIP_Similarity', 'NOT FOUND')}")
print(f"   LPIPS_Distance: {results.get('LPIPS_Distance', 'NOT FOUND')}")
print(f"   SSIM: {results.get('SSIM', 'NOT FOUND')}")
print(f"   Final_Score: {results.get('Final_Score', 'NOT FOUND')}")

# 5. 记录数据
print("\n5. Log data...")
logger = EvaluationDataLogger(data_dir="evaluation_data")
sample_id = logger.log_evaluation(
    results=results,
    sample_category='Basic',
    image1_name='test_img1.jpg',
    image2_name='test_img2.jpg',
    notes='Data flow test'
)
print(f"   Logged ID: {sample_id}")

# 6. 读取并验证
print("\n6. Read CSV and verify...")
df = pd.read_csv('evaluation_data/evaluation_results.csv')
print(f"   Total rows in CSV: {len(df)}")
print(f"   Looking for sample_id: {sample_id}")

matching_rows = df[df['sample_id'] == sample_id]
if len(matching_rows) == 0:
    print(f"   ERROR: Sample ID not found in CSV!")
    print(f"   Last 3 sample_ids in CSV: {df['sample_id'].tail(3).tolist()}")
    sys.exit(1)

latest_row = matching_rows.iloc[0]

print(f"   CSV lpips_distance: {latest_row['lpips_distance']}")
print(f"   Result LPIPS_Distance: {results.get('LPIPS_Distance', 'NOT FOUND')}")

# 7. 比较
print("\n7. Data consistency check:")
csv_lpips = float(latest_row['lpips_distance'])
result_lpips = float(results.get('LPIPS_Distance', 0.0))

if abs(csv_lpips - result_lpips) < 0.0001:
    print(f"   OK! Consistent: {csv_lpips:.4f} == {result_lpips:.4f}")
else:
    print(f"   ERROR! Inconsistent: CSV: {csv_lpips:.4f} != Result: {result_lpips:.4f}")
    print(f"   Difference: {abs(csv_lpips - result_lpips):.6f}")

print("\n" + "="*70)
print("Test complete!")
print("="*70 + "\n")
