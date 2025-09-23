import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Union, Tuple, Optional
from config.settings import Config

class Visualizer:
    """可视化工具类"""
    
    def __init__(self):
        self.figure_size = Config.FIGURE_SIZE
        self.dpi = Config.DPI
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_evaluation_scores(self, results: Dict, title: str = "评估结果") -> plt.Figure:
        """绘制评估分数条形图"""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        metrics = []
        scores = []
        colors = []
        
        # 提取分数数据
        if 'clip_similarity' in results:
            metrics.append('CLIP相似度')
            scores.append(results['clip_similarity'] * 100)
            colors.append('#ff7f0e')
        
        if 'identity_similarity' in results:
            metrics.append('身份一致性')
            scores.append(results['identity_similarity'] * 100)
            colors.append('#2ca02c')
        
        if 'lpips_similarity' in results:
            metrics.append('感知相似度')
            scores.append(results['lpips_similarity'] * 100)
            colors.append('#d62728')
        
        if 'ssim' in results:
            metrics.append('SSIM')
            scores.append(results['ssim']['score'])
            colors.append('#9467bd')
        
        if 'psnr' in results:
            metrics.append('PSNR')
            scores.append(results['psnr']['score'])
            colors.append('#8c564b')
        
        # 绘制条形图
        bars = ax.bar(metrics, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('分数')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_radar_chart(self, results: Dict, title: str = "多维度评估雷达图") -> go.Figure:
        """绘制雷达图"""
        metrics = []
        values = []
        
        # 提取数据
        if 'clip_similarity' in results:
            metrics.append('CLIP相似度')
            values.append(results['clip_similarity'])
        
        if 'identity_similarity' in results:
            metrics.append('身份一致性')
            values.append(results['identity_similarity'])
        
        if 'lpips_similarity' in results:
            metrics.append('感知相似度')
            values.append(results['lpips_similarity'])
        
        if 'ssim' in results:
            metrics.append('SSIM')
            values.append(results['ssim']['value'])
        
        if 'psnr' in results:
            metrics.append('PSNR')
            values.append(min(1.0, results['psnr']['value'] / 50))  # 归一化PSNR
        
        # 创建雷达图
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='评估结果',
            line_color='rgb(1,90,200)',
            fillcolor='rgba(1,90,200,0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title,
            font=dict(size=12)
        )
        
        return fig
    
    def plot_comparison_grid(self, images: List[Tuple[str, Image.Image]], 
                           grid_size: Tuple[int, int] = None,
                           title: str = "图像对比") -> plt.Figure:
        """绘制图像对比网格"""
        n_images = len(images)
        
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=self.dpi)
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (label, image) in enumerate(images):
            if i < len(axes):
                axes[i].imshow(image)
                axes[i].set_title(label, fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_batch_results(self, batch_results: List[Dict], 
                          metric_name: str = 'clip_similarity',
                          title: str = None) -> plt.Figure:
        """绘制批量结果分布"""
        if title is None:
            title = f"{metric_name} 分布"
        
        # 提取分数
        scores = []
        for result in batch_results:
            if metric_name in result:
                if isinstance(result[metric_name], dict):
                    scores.append(result[metric_name].get('value', 0))
                else:
                    scores.append(result[metric_name])
        
        if not scores:
            print(f"未找到 {metric_name} 数据")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        # 直方图
        ax1.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('分数')
        ax1.set_ylabel('频次')
        ax1.set_title(f'{metric_name} 分布直方图')
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        box_plot = ax2.boxplot(scores, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
        ax2.set_ylabel('分数')
        ax2.set_title(f'{metric_name} 箱线图')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ax2.text(0.02, 0.98, f'均值: {mean_score:.3f}\n标准差: {std_score:.3f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_matrix(self, batch_results: List[Dict]) -> plt.Figure:
        """绘制指标相关性矩阵"""
        # 构建数据框
        data = []
        for result in batch_results:
            row = {}
            if 'clip_similarity' in result:
                row['CLIP'] = result['clip_similarity']
            if 'identity_similarity' in result:
                row['Identity'] = result['identity_similarity']
            if 'lpips_similarity' in result:
                row['LPIPS'] = result['lpips_similarity']
            if 'ssim' in result:
                row['SSIM'] = result['ssim']['value'] if isinstance(result['ssim'], dict) else result['ssim']
            if 'psnr' in result:
                row['PSNR'] = result['psnr']['value'] if isinstance(result['psnr'], dict) else result['psnr']
            
            if row:  # 只添加非空行
                data.append(row)
        
        if not data:
            print("没有足够的数据绘制相关性矩阵")
            return None
        
        df = pd.DataFrame(data)
        correlation_matrix = df.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # 绘制热力图
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title('评估指标相关性矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_interactive_dashboard(self, results: Dict) -> go.Figure:
        """创建交互式仪表盘"""
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('分数条形图', '时间序列', '分布图', '综合评分'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "indicator"}]]
        )
        
        # 提取数据
        metrics = []
        scores = []
        
        if 'clip_similarity' in results:
            metrics.append('CLIP')
            scores.append(results['clip_similarity'] * 100)
        if 'identity_similarity' in results:
            metrics.append('Identity')
            scores.append(results['identity_similarity'] * 100)
        if 'lpips_similarity' in results:
            metrics.append('LPIPS')
            scores.append(results['lpips_similarity'] * 100)
        
        # 条形图
        fig.add_trace(
            go.Bar(x=metrics, y=scores, name="评估分数"),
            row=1, col=1
        )
        
        # 综合评分指示器
        overall_score = np.mean(scores) if scores else 0
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "综合评分"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 80], 'color': "yellow"},
                           {'range': [80, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="评估仪表盘")
        
        return fig
    
    def save_plot(self, fig: plt.Figure, filename: str, dpi: int = None):
        """保存图表"""
        if dpi is None:
            dpi = self.dpi
        
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"图表已保存: {filename}")