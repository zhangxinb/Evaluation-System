import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

# --- 1. 配置图片路径和标题 ---
# 请将这里的 'path/to/...' 替换为您的实际图片文件名
image_paths = {
    'a': r'C:\Users\ZX\OneDrive - LUT University\桌面\Thesis\Dataset\ZhangXin\zhangxin-ref.jpg',
    'b': r'C:\Users\ZX\OneDrive - LUT University\桌面\Thesis\Dataset\ZhangXin\zhangxin-Basic01.jpg',
    'c': r'C:\Users\ZX\OneDrive - LUT University\桌面\Thesis\Dataset\ZhangXin\zhangxin-Attribute03.jpg',
    'd': r'C:\Users\ZX\OneDrive - LUT University\桌面\Thesis\Dataset\ZhangXin\zhangxin-Boundary03.jpg'
}

# 子图的标题
titles = {
    'a': '(a) Reference Set',
    'b': '(b) Basic Consistency',
    'c': '(c) Attribute Consistency',
    'd': '(d) Boundary Case'
}

# --- 2. 创建一个 2x2 的复合图 ---
# figsize 控制整个图的大小，可以按需调整
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 将 2x2 的子图数组展开为一维，方便遍历
axs = axs.flatten()

# --- 3. 遍历并填充每个子图 ---
for i, key in enumerate(titles.keys()):
    ax = axs[i]
    path = image_paths[key]
    title = titles[key]

    try:
        # 读取图片
        img = Image.open(path)
        
        # 在子图中显示图片
        ax.imshow(img)
        
        # 设置子图标题
        ax.set_title(title, fontsize=14, pad=10) # pad 增加标题和图片的间距
        
        # 关闭坐标轴刻度
        ax.axis('off')

    except FileNotFoundError:
        print(f"错误：找不到图片 '{path}'。请检查路径和文件名。")
        ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis('off')


# --- 4. 调整整体布局并保存 ---
# 调整子图之间的间距，防止标题重叠
plt.tight_layout(pad=3.0)

# 保存最终的复合图
output_filename = 'cced_dataset_visualization.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"复合图已成功保存为 '{output_filename}'")