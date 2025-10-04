#!/usr/bin/env python3
"""
Quick Analysis Script
View logged data and generate visualizations
"""

import sys
import os
from data_logger import EvaluationDataLogger
from visualizer import HypothesisVisualizer

def main():
    """Main analysis function"""
    
    print("\n" + "="*70)
    print("📊 EVALUATION DATA ANALYSIS")
    print("="*70 + "\n")
    
    # Check if data exists
    data_path = "evaluation_data/evaluation_results.csv"
    if not os.path.exists(data_path):
        print("❌ No evaluation data found!")
        print(f"   Expected: {data_path}")
        print("\n💡 Run evaluations first using the web interface (app.py)")
        return
    
    # Initialize logger
    logger = EvaluationDataLogger(data_dir="evaluation_data")
    
    # Print summary
    logger.print_summary()
    
    # Get statistics
    print("\n📈 DETAILED STATISTICS BY CATEGORY")
    print("="*70)
    stats = logger.get_summary_statistics()
    
    for category, data in stats.items():
        print(f"\n{category} Consistency ({data['count']} samples):")
        print("-" * 50)
        for metric_name, metric_data in data['metrics'].items():
            print(f"  {metric_name}:")
            print(f"    Mean: {metric_data['mean']:.4f}")
            print(f"    Std:  {metric_data['std']:.4f}")
            print(f"    Range: [{metric_data['min']:.4f}, {metric_data['max']:.4f}]")
    
    # Validate hypotheses
    print("\n" + "="*70)
    print("🔬 HYPOTHESIS VALIDATION")
    print("="*70)
    
    validation = logger.validate_hypothesis()
    
    for hypothesis_name, hypothesis_data in validation.items():
        if hypothesis_name == 'hypothesis_a':
            print("\n✅ Hypothesis A: Basic Consistency Samples")
            print("   All key metrics should yield high scores")
        elif hypothesis_name == 'hypothesis_b':
            print("\n✅ Hypothesis B: Attribute Consistency Samples")
            print("   Identity metrics high, SSIM shows decline")
        elif hypothesis_name == 'hypothesis_c':
            print("\n✅ Hypothesis C: Boundary Case Samples")
            print("   Identity & perceptual metrics show sharp drops")
        
        print(f"   Samples: {hypothesis_data['sample_count']}")
        print(f"   Status: {hypothesis_data['validation']}")
        
        if 'results' in hypothesis_data:
            print("   Results:")
            for key, value in hypothesis_data['results'].items():
                if isinstance(value, float):
                    print(f"     {key}: {value:.2%}")
                else:
                    print(f"     {key}: {value}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    try:
        viz = HypothesisVisualizer(data_path=data_path)
        viz.generate_all_plots()
        
        print("\n✅ Analysis complete!")
        print(f"📁 Visualizations saved to: evaluation_data/visualizations/")
        print("\n📈 Generated plots:")
        print("   - metric_comparison.png     (Box plots by category)")
        print("   - hypothesis_validation.png (Hypothesis testing)")
        print("   - metric_heatmap.png        (Heatmap of all metrics)")
        print("   - score_distribution.png    (Score distributions)")
        
    except Exception as e:
        print(f"\n⚠️ Visualization failed: {e}")
        print("   Install required packages: pip install matplotlib seaborn")
    
    # Export for external analysis
    print("\n" + "="*70)
    print("📤 EXPORT OPTIONS")
    print("="*70)
    
    export_path = logger.export_for_visualization()
    print(f"\n✅ Data exported for external analysis:")
    print(f"   {export_path}")
    print("\n💡 You can import this CSV into:")
    print("   - Excel/Google Sheets for manual analysis")
    print("   - R/Python for statistical tests")
    print("   - Tableau/PowerBI for interactive visualizations")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
