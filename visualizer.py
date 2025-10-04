#!/usr/bin/env python3
"""
Data Visualization for Hypothesis Validation
Creates charts to compare metrics across sample categories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import os

class HypothesisVisualizer:
    """
    Visualizes evaluation results to validate hypotheses:
    - Hypothesis A: Basic Consistency (high scores across all metrics)
    - Hypothesis B: Attribute Consistency (identity high, SSIM low)
    - Hypothesis C: Boundary Cases (sharp drops in identity metrics)
    """
    
    def __init__(self, data_path: str = "evaluation_data/evaluation_results.csv"):
        """Initialize visualizer with data path"""
        self.data_path = data_path
        self.df = None
        # Use the same directory as the data file
        data_dir = os.path.dirname(data_path)
        self.output_dir = os.path.join(data_dir, "visualizations")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load evaluation data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} samples from {self.data_path}")
            
            # Print category distribution
            if 'sample_category' in self.df.columns:
                print("\n📊 Sample Distribution:")
                for category in ['Basic', 'Attribute', 'Boundary']:
                    count = len(self.df[self.df['sample_category'] == category])
                    print(f"  {category}: {count} samples")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def plot_metric_comparison(self, save_path: Optional[str] = None):
        """
        Plot key metrics across sample categories
        Validates all three hypotheses
        
        Note: Metrics have different directions
        - Higher is better: DeepFace, CLIP, SSIM
        - Lower is better: LPIPS Distance
        """
        if self.df.empty:
            print("⚠️ No data to visualize")
            return
        
        # Key metrics for hypothesis validation
        metrics = {
            'DeepFace\nSimilarity\n(↑ better)': 'deepface_similarity',
            'CLIP\nSimilarity\n(↑ better)': 'clip_image_similarity',
            'LPIPS\nDistance\n(↓ better)': 'lpips_distance',
            'SSIM\n(↑ better)': 'ssim_whole_image'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Metric Comparison Across Categories (Mixed Directions: ↑ and ↓)', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        categories = ['Basic', 'Attribute', 'Boundary']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        
        for idx, (metric_name, metric_col) in enumerate(metrics.items()):
            ax = axes[idx // 2, idx % 2]
            
            # Prepare data
            data = []
            labels = []
            for category in categories:
                cat_data = self.df[self.df['sample_category'] == category][metric_col].dropna()
                if len(cat_data) > 0:
                    data.append(cat_data)
                    labels.append(category)
            
            if not data:
                continue
            
            # Create box plot
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Color boxes
            for patch, color in zip(bp['boxes'], colors[:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Add scatter points
            for i, d in enumerate(data):
                x = np.random.normal(i+1, 0.04, size=len(d))
                ax.scatter(x, d, alpha=0.5, s=30, color=colors[i])
            
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add mean value annotations
            for i, d in enumerate(data):
                mean_val = d.mean()
                ax.text(i+1, ax.get_ylim()[1]*0.95, f'μ={mean_val:.3f}',
                       ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.3))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "metric_comparison.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_hypothesis_validation(self, save_path: Optional[str] = None):
        """
        Create comprehensive hypothesis validation plot
        Shows expected vs actual patterns
        """
        if self.df.empty:
            print("⚠️ No data to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Hypothesis Validation: Expected Patterns', 
                     fontsize=16, fontweight='bold')
        
        categories = ['Basic', 'Attribute', 'Boundary']
        
        # Hypothesis A: Basic Consistency
        ax = axes[0]
        metrics = ['deepface_similarity', 'clip_image_similarity', 'lpips_distance', 'ssim_whole_image']
        metric_names = ['DeepFace', 'CLIP-I', 'LPIPS (dist)', 'SSIM']
        
        basic_df = self.df[self.df['sample_category'] == 'Basic']
        if len(basic_df) > 0:
            values = [basic_df[m].mean() for m in metrics]
            colors = ['#2ecc71' if v > 0.7 else '#e74c3c' for v in values]
            
            bars = ax.bar(metric_names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.axhline(y=0.7, color='blue', linestyle='--', linewidth=2, label='Expected Threshold')
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_title('Hypothesis A: Basic Consistency\n(All metrics should be HIGH)', 
                        fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Hypothesis B: Attribute Consistency
        ax = axes[1]
        attr_df = self.df[self.df['sample_category'] == 'Attribute']
        if len(attr_df) > 0:
            identity_metrics = ['deepface_similarity', 'clip_image_similarity']
            attribute_metrics = ['ssim_whole_image']
            
            identity_vals = [attr_df[m].mean() for m in identity_metrics]
            attribute_vals = [attr_df[m].mean() for m in attribute_metrics]
            
            x = np.arange(3)
            all_vals = identity_vals + attribute_vals
            labels = ['DeepFace', 'CLIP-I', 'SSIM']
            colors = ['#2ecc71', '#2ecc71', '#f39c12']  # Identity high, SSIM lower
            
            bars = ax.bar(labels, all_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.axhline(y=0.7, color='blue', linestyle='--', linewidth=2, label='Identity Threshold')
            ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='SSIM Threshold')
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_title('Hypothesis B: Attribute Consistency\n(Identity HIGH, SSIM LOWER)', 
                        fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, all_vals):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Hypothesis C: Boundary Cases
        ax = axes[2]
        boundary_df = self.df[self.df['sample_category'] == 'Boundary']
        if len(boundary_df) > 0:
            metrics = ['deepface_similarity', 'clip_image_similarity', 'lpips_distance']
            metric_names = ['DeepFace', 'CLIP-I', 'LPIPS (dist)']
            
            values = [boundary_df[m].mean() for m in metrics]
            colors = ['#e74c3c' if v < 0.5 else '#2ecc71' for v in values]
            
            bars = ax.bar(metric_names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.axhline(y=0.4, color='red', linestyle='--', linewidth=2, label='Failure Threshold')
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax.set_title('Hypothesis C: Boundary Cases\n(All metrics should be LOW)', 
                        fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "hypothesis_validation.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_heatmap(self, save_path: Optional[str] = None):
        """
        Create heatmap of metrics across categories
        All metrics normalized to "higher is better"
        """
        if self.df.empty:
            print("⚠️ No data to visualize")
            return
        
        # Calculate mean values for each category (mixed directions)
        metrics = {
            'DeepFace (↑)': 'deepface_similarity',
            'CLIP-I (↑)': 'clip_image_similarity',
            'LPIPS (↓)': 'lpips_distance',
            'SSIM (↑)': 'ssim_whole_image',
            'Final Score (↑)': 'final_score'
        }
        
        categories = ['Basic', 'Attribute', 'Boundary']
        
        # Create matrix
        data = []
        for category in categories:
            cat_df = self.df[self.df['sample_category'] == category]
            if len(cat_df) > 0:
                row = [cat_df[col].mean() for col in metrics.values()]
                data.append(row)
            else:
                data.append([0] * len(metrics))
        
        data = np.array(data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(list(metrics.keys()))
        ax.set_yticklabels(categories)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title("Metric Heatmap: Mean Scores by Category\n(All metrics: Higher = Better)", 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "metric_heatmap.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_score_distribution(self, save_path: Optional[str] = None):
        """
        Plot distribution of final scores by category
        """
        if self.df.empty:
            print("⚠️ No data to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ['Basic', 'Attribute', 'Boundary']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        for category, color in zip(categories, colors):
            cat_df = self.df[self.df['sample_category'] == category]
            if len(cat_df) > 0:
                scores = cat_df['final_score'].dropna()
                ax.hist(scores, bins=20, alpha=0.6, label=category, color=color, edgecolor='black')
        
        ax.set_xlabel('Final Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Final Scores by Category', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "score_distribution.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_metric_directions(self, save_path: Optional[str] = None):
        """
        Create a reference chart showing metric directions and interpretations
        Helpful for understanding which metrics are "higher is better" vs "lower is better"
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Metric information: (name, direction, optimal_range, color)
        metric_info = [
            ('DeepFace Similarity', '↑', '0.6-1.0 = Same Person', '#2ecc71', True),
            ('CLIP-I Similarity', '↑', '0.7-1.0 = High Semantic Match', '#2ecc71', True),
            ('LPIPS Distance', '↓', '0.0-0.4 = Very Similar', '#e74c3c', False),
            ('SSIM', '↑', '0.5-1.0 = Structurally Similar', '#2ecc71', True),
            ('PSNR (dB)', '↑', '20-40+ = Good Quality', '#2ecc71', True),
            ('MSE', '↓', '0-1000 = Lower is Better', '#e74c3c', False),
            ('Histogram Correlation', '↑', '0.6-1.0 = Similar Colors', '#2ecc71', True),
        ]
        
        # Get actual data if available
        if not self.df.empty:
            metrics_with_data = []
            for name, direction, optimal, color, higher_better in metric_info:
                # Map display name to column name
                col_map = {
                    'DeepFace Similarity': 'deepface_similarity',
                    'CLIP-I Similarity': 'clip_image_similarity',
                    'LPIPS Distance': 'lpips_distance',
                    'SSIM': 'ssim_whole_image',
                    'PSNR (dB)': 'psnr',
                    'MSE': 'mse',
                    'Histogram Correlation': 'histogram_correlation'
                }
                
                col_name = col_map.get(name)
                if col_name and col_name in self.df.columns:
                    mean_val = self.df[col_name].mean()
                    std_val = self.df[col_name].std()
                    metrics_with_data.append((name, direction, optimal, color, higher_better, mean_val, std_val))
            
            if metrics_with_data:
                y_pos = np.arange(len(metrics_with_data))
                
                # Create horizontal bar chart showing current data
                for i, (name, direction, optimal, color, higher_better, mean_val, std_val) in enumerate(metrics_with_data):
                    # Normalize value for display (0-1 scale for most metrics)
                    if 'PSNR' in name:
                        display_val = min(1.0, mean_val / 40.0)  # Normalize PSNR to 0-1
                    elif 'MSE' in name:
                        display_val = max(0, 1.0 - min(1.0, mean_val / 10000.0))  # Invert MSE
                    else:
                        display_val = mean_val
                    
                    # Draw bar
                    bar = ax.barh(i, display_val, color=color, alpha=0.6, edgecolor='black', linewidth=2)
                    
                    # Add direction indicator
                    direction_color = '#2ecc71' if higher_better else '#e74c3c'
                    ax.text(-0.08, i, direction, fontsize=20, fontweight='bold', 
                           color=direction_color, ha='center', va='center')
                    
                    # Add metric name
                    ax.text(-0.15, i, name, fontsize=11, fontweight='bold', ha='right', va='center')
                    
                    # Add current value
                    ax.text(display_val + 0.02, i, f'{mean_val:.3f}±{std_val:.3f}', 
                           fontsize=9, va='center', fontweight='bold')
                    
                    # Add optimal range
                    ax.text(1.02, i, optimal, fontsize=9, va='center', style='italic', color='gray')
                
                ax.set_yticks([])
                ax.set_xlim(-0.2, 1.3)
                ax.set_xlabel('Normalized Score (for display)', fontsize=12, fontweight='bold')
                ax.set_title('Metric Directions & Current Data Summary\n🟢 ↑ = Higher is Better  |  🔴 ↓ = Lower is Better', 
                           fontsize=14, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#2ecc71', alpha=0.6, edgecolor='black', label='↑ Higher is Better'),
                    Patch(facecolor='#e74c3c', alpha=0.6, edgecolor='black', label='↓ Lower is Better')
                ]
                ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, "metric_directions.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n📊 Generating all visualizations...")
        print("-" * 50)
        
        self.plot_metric_comparison()
        self.plot_hypothesis_validation()
        self.plot_heatmap()
        self.plot_score_distribution()
        self.plot_metric_directions()
        
        print("\n✅ All visualizations generated!")
        print(f"📁 Output directory: {self.output_dir}")


def test_visualizer():
    """Test the visualizer with sample data"""
    print("\n🧪 Testing Visualizer")
    print("-" * 50)
    
    # First, create some test data
    from data_logger import EvaluationDataLogger
    
    logger = EvaluationDataLogger(data_dir="evaluation_data")
    
    # Generate test samples
    # Basic Consistency (high scores)
    for i in range(5):
        results = {
            'Identity_Similarity': np.random.uniform(0.75, 0.95),
            'Identity_Confidence': np.random.uniform(0.8, 0.95),
            'Identity_Decision': 'Same Person',
            'Models_Used': 3,
            'CLIP_Similarity': np.random.uniform(0.7, 0.9),
            'LPIPS_Distance': np.random.uniform(0.05, 0.25),  # Low distance = similar
            'SSIM': np.random.uniform(0.7, 0.9),
            'PSNR': np.random.uniform(25, 35),
            'Final_Score': np.random.uniform(0.75, 0.9)
        }
        logger.log_evaluation(results, 'Basic', image1_name=f'basic_{i}_1.jpg', image2_name=f'basic_{i}_2.jpg')
    
    # Attribute Consistency (identity high, SSIM low)
    for i in range(5):
        results = {
            'Identity_Similarity': np.random.uniform(0.7, 0.9),
            'Identity_Confidence': np.random.uniform(0.75, 0.9),
            'Identity_Decision': 'Same Person',
            'Models_Used': 3,
            'CLIP_Similarity': np.random.uniform(0.65, 0.85),
            'LPIPS_Distance': np.random.uniform(0.25, 0.5),  # Medium distance
            'SSIM': np.random.uniform(0.3, 0.5),  # Lower SSIM
            'PSNR': np.random.uniform(15, 25),
            'Final_Score': np.random.uniform(0.55, 0.7)
        }
        logger.log_evaluation(results, 'Attribute', image1_name=f'attr_{i}_1.jpg', image2_name=f'attr_{i}_2.jpg')
    
    # Boundary Cases (all low scores)
    for i in range(5):
        results = {
            'Identity_Similarity': np.random.uniform(0.2, 0.4),
            'Identity_Confidence': np.random.uniform(0.3, 0.5),
            'Identity_Decision': 'Different Person',
            'Models_Used': 3,
            'CLIP_Similarity': np.random.uniform(0.3, 0.5),
            'LPIPS_Distance': np.random.uniform(0.5, 0.9),  # High distance = different
            'SSIM': np.random.uniform(0.2, 0.4),
            'PSNR': np.random.uniform(10, 20),
            'Final_Score': np.random.uniform(0.2, 0.4)
        }
        logger.log_evaluation(results, 'Boundary', image1_name=f'boundary_{i}_1.jpg', image2_name=f'boundary_{i}_2.jpg')
    
    # Now test visualizer
    viz = HypothesisVisualizer(data_path="evaluation_data/evaluation_results.csv")
    viz.generate_all_plots()
    
    print("\n✅ Visualizer test completed")
    print(f"📁 Check 'evaluation_data/visualizations/' for generated plots")


if __name__ == "__main__":
    test_visualizer()
