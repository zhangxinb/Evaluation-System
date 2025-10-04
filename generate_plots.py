#!/usr/bin/env python3
"""
Manual Plot Generation Script
Generate specific visualizations on demand
"""

import sys
import os
from visualizer import HypothesisVisualizer

def main():
    """Generate plots from collected data"""
    
    print("\n" + "="*70)
    print("📊 PLOT GENERATION")
    print("="*70 + "\n")
    
    # Check if data exists
    data_path = "evaluation_data/evaluation_results.csv"
    
    if not os.path.exists(data_path):
        print("❌ No data found!")
        print(f"   Expected: {data_path}")
        print("\n💡 Please run evaluations first using:")
        print("   python app.py")
        return
    
    # Initialize visualizer
    print(f"📂 Loading data from: {data_path}")
    viz = HypothesisVisualizer(data_path=data_path)
    
    if viz.df.empty:
        print("❌ Data file is empty!")
        return
    
    print(f"✅ Loaded {len(viz.df)} samples\n")
    
    # Menu for plot selection
    print("Select plots to generate:")
    print("  1. Metric Comparison (Box plots)")
    print("  2. Hypothesis Validation (Bar charts)")
    print("  3. Metric Heatmap")
    print("  4. Score Distribution")
    print("  5. Metric Directions Reference")
    print("  6. All plots")
    print("  0. Exit")
    
    while True:
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == '0':
            print("👋 Exiting...")
            break
        
        elif choice == '1':
            print("\n📊 Generating Metric Comparison...")
            viz.plot_metric_comparison()
            print("✅ Done!")
        
        elif choice == '2':
            print("\n📊 Generating Hypothesis Validation...")
            viz.plot_hypothesis_validation()
            print("✅ Done!")
        
        elif choice == '3':
            print("\n📊 Generating Metric Heatmap...")
            viz.plot_heatmap()
            print("✅ Done!")
        
        elif choice == '4':
            print("\n📊 Generating Score Distribution...")
            viz.plot_score_distribution()
            print("✅ Done!")
        
        elif choice == '5':
            print("\n📊 Generating Metric Directions Reference...")
            viz.plot_metric_directions()
            print("✅ Done!")
        
        elif choice == '6':
            print("\n📊 Generating All Plots...")
            viz.generate_all_plots()
            print("✅ All plots generated!")
            break
        
        else:
            print("⚠️ Invalid choice! Please enter 0-6")
    
    print("\n📁 Plots saved to:")
    print(f"   {viz.output_dir}/")
    print("\n✅ Plot generation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
