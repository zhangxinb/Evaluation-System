#!/usr/bin/env python3
"""
Data Logger for Evaluation Results
Records all metrics for hypothesis validation and visualization
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class EvaluationDataLogger:
    """
    Logs evaluation results for research analysis
    Supports hypothesis validation for:
    - Basic Consistency Samples
    - Attribute Consistency Samples  
    - Boundary Case Samples
    """
    
    # CSV field names (MUST be consistent between init and append!)
    FIELDNAMES = [
        # Metadata
        'timestamp',
        'sample_id',
        'sample_category',  # Basic/Attribute/Boundary
        'image1_name',
        'image2_name',
        
        # Identity-Centric Metrics (Hypothesis a, b, c)
        'deepface_similarity',
        'deepface_confidence',
        'deepface_decision',
        'models_used',
        
        # Semantic Metrics (Hypothesis a, b, c)
        'clip_image_similarity',
        
        # Perceptual Metrics (Hypothesis a, c) - Distance only
        'lpips_distance',
        
        # Structural Metrics (Hypothesis b)
        'ssim_whole_image',
        'ssim_face_region',  # For face-specific analysis
        'psnr',
        'mse',
        'histogram_correlation',
        
        # Overall Assessment
        'final_score',
        'consistency_level',
        'overall_confidence',
        
        # Notes
        'notes'
    ]
    
    def __init__(self, data_dir: str = "evaluation_data"):
        """Initialize data logger with storage directory"""
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "evaluation_results.csv")
        self.json_path = os.path.join(data_dir, "evaluation_results.json")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.csv_path):
            self._initialize_csv()
        
        # Load existing JSON data
        self.json_data = self._load_json_data()
        
        print(f"✅ Data Logger initialized: {data_dir}")
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(self.FIELDNAMES)
        
        print(f"📝 CSV initialized: {self.csv_path}")
    
    def _load_json_data(self) -> list:
        """Load existing JSON data"""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def log_evaluation(self, 
                       results: Dict[str, Any],
                       sample_category: str,
                       sample_id: Optional[str] = None,
                       image1_name: str = "image1",
                       image2_name: str = "image2",
                       notes: str = "") -> str:
        """
        Log evaluation results
        
        Args:
            results: Evaluation results dictionary
            sample_category: 'Basic', 'Attribute', or 'Boundary'
            sample_id: Optional unique identifier
            image1_name: Name of first image
            image2_name: Name of second image
            notes: Optional notes
            
        Returns:
            sample_id: The assigned sample ID
        """
        
        # Validate category
        valid_categories = ['Basic', 'Attribute', 'Boundary']
        if sample_category not in valid_categories:
            print(f"⚠️ Warning: '{sample_category}' not in {valid_categories}, using 'Unknown'")
            sample_category = 'Unknown'
        
        # Generate sample ID if not provided or empty
        if not sample_id or sample_id.strip() == "":
            import random
            timestamp = datetime.now()
            microsec = timestamp.strftime('%f')[:3]  # 取前3位微秒
            rand_suffix = random.randint(100, 999)
            sample_id = f"{sample_category}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{microsec}_{rand_suffix}"
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id,
            'sample_category': sample_category,
            'image1_name': image1_name,
            'image2_name': image2_name,
            
            # Identity-Centric Metrics
            'deepface_similarity': results.get('Identity_Similarity', 0.0),
            'deepface_confidence': results.get('Identity_Confidence', 0.0),
            'deepface_decision': results.get('Identity_Decision', 'Unknown'),
            'models_used': results.get('Models_Used', 0),
            
            # Semantic Metrics
            'clip_image_similarity': results.get('CLIP_Similarity', 0.0),
            
            # Perceptual Metrics (Distance only - lower is better)
            'lpips_distance': results.get('LPIPS_Distance', 0.0),
            
            # Structural Metrics
            'ssim_whole_image': results.get('SSIM', 0.0),
            'ssim_face_region': results.get('SSIM_Face', 0.0),  # May not exist yet
            'psnr': results.get('PSNR', 0.0),
            'mse': results.get('MSE', 0.0),
            'histogram_correlation': results.get('Histogram_Similarity', 0.0),
            
            # Overall Assessment
            'final_score': results.get('Final_Score', 0.0),
            'consistency_level': results.get('Consistency_Level', 'Unknown'),
            'overall_confidence': results.get('Overall_Confidence', 0.0),
            
            # Notes
            'notes': notes
        }
        
        # Append to CSV (use class FIELDNAMES to ensure column order matches header)
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(log_entry)
        
        # Append to JSON (with full results for detailed analysis)
        json_entry = {
            **log_entry,
            'full_results': results  # Store complete results
        }
        self.json_data.append(json_entry)
        
        # Save JSON
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Logged: {sample_id} ({sample_category})")
        return sample_id
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get all logged data as pandas DataFrame"""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"📊 Loaded {len(df)} records")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics by category"""
        df = self.get_dataframe()
        
        if df.empty:
            return {}
        
        summary = {}
        
        for category in ['Basic', 'Attribute', 'Boundary']:
            cat_df = df[df['sample_category'] == category]
            
            if len(cat_df) == 0:
                continue
            
            summary[category] = {
                'count': len(cat_df),
                'metrics': {
                    'deepface_similarity': {
                        'mean': float(cat_df['deepface_similarity'].mean()),
                        'std': float(cat_df['deepface_similarity'].std()),
                        'min': float(cat_df['deepface_similarity'].min()),
                        'max': float(cat_df['deepface_similarity'].max())
                    },
                    'clip_image_similarity': {
                        'mean': float(cat_df['clip_image_similarity'].mean()),
                        'std': float(cat_df['clip_image_similarity'].std()),
                        'min': float(cat_df['clip_image_similarity'].min()),
                        'max': float(cat_df['clip_image_similarity'].max())
                    },
                    'lpips_distance': {
                        'mean': float(cat_df['lpips_distance'].mean()),
                        'std': float(cat_df['lpips_distance'].std()),
                        'min': float(cat_df['lpips_distance'].min()),
                        'max': float(cat_df['lpips_distance'].max())
                    },
                    'ssim_whole_image': {
                        'mean': float(cat_df['ssim_whole_image'].mean()),
                        'std': float(cat_df['ssim_whole_image'].std()),
                        'min': float(cat_df['ssim_whole_image'].min()),
                        'max': float(cat_df['ssim_whole_image'].max())
                    }
                }
            }
        
        return summary
    
    def validate_hypothesis(self) -> Dict[str, Any]:
        """
        Validate research hypotheses based on collected data
        
        Returns validation results for each hypothesis
        """
        df = self.get_dataframe()
        
        if df.empty:
            return {'error': 'No data available for validation'}
        
        validation = {}
        
        # Hypothesis A: Basic Consistency Samples
        basic_df = df[df['sample_category'] == 'Basic']
        if len(basic_df) > 0:
            validation['hypothesis_a'] = {
                'sample_count': len(basic_df),
                'description': 'All key metrics should yield high scores',
                'results': {
                    'deepface_high': float((basic_df['deepface_similarity'] > 0.6).mean()),
                    'clip_high': float((basic_df['clip_image_similarity'] > 0.7).mean()),
                    'lpips_low_distance': float((basic_df['lpips_distance'] < 0.5).mean()),
                },
                'validation': 'PASS' if (
                    (basic_df['deepface_similarity'] > 0.6).mean() > 0.8 and
                    (basic_df['clip_image_similarity'] > 0.7).mean() > 0.8 and
                    (basic_df['lpips_distance'] < 0.5).mean() > 0.8
                ) else 'FAIL'
            }
        
        # Hypothesis B: Attribute Consistency Samples
        attr_df = df[df['sample_category'] == 'Attribute']
        if len(attr_df) > 0:
            validation['hypothesis_b'] = {
                'sample_count': len(attr_df),
                'description': 'Identity metrics high, attribute metrics (SSIM) show decline',
                'results': {
                    'deepface_high': float((attr_df['deepface_similarity'] > 0.6).mean()),
                    'clip_high': float((attr_df['clip_image_similarity'] > 0.7).mean()),
                    'ssim_lower': float(attr_df['ssim_whole_image'].mean())
                },
                'validation': 'PASS' if (
                    (attr_df['deepface_similarity'] > 0.6).mean() > 0.8 and
                    attr_df['ssim_whole_image'].mean() < 0.5  # SSIM should be lower
                ) else 'FAIL'
            }
        
        # Hypothesis C: Boundary Case Samples
        boundary_df = df[df['sample_category'] == 'Boundary']
        if len(boundary_df) > 0:
            validation['hypothesis_c'] = {
                'sample_count': len(boundary_df),
                'description': 'Identity and perceptual metrics show sharp drops',
                'results': {
                    'deepface_low': float((boundary_df['deepface_similarity'] < 0.4).mean()),
                    'clip_low': float((boundary_df['clip_image_similarity'] < 0.5).mean()),
                    'lpips_high_distance': float((boundary_df['lpips_distance'] > 0.6).mean()),
                },
                'validation': 'PASS' if (
                    (boundary_df['deepface_similarity'] < 0.4).mean() > 0.7 and
                    (boundary_df['clip_image_similarity'] < 0.5).mean() > 0.7
                ) else 'FAIL'
            }
        
        return validation
    
    def export_for_visualization(self, output_path: str = None) -> str:
        """
        Export data in format optimized for visualization
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = os.path.join(self.data_dir, "visualization_data.csv")
        
        df = self.get_dataframe()
        
        if df.empty:
            print("⚠️ No data to export")
            return None
        
        # Select key columns for visualization
        viz_columns = [
            'sample_id',
            'sample_category',
            'deepface_similarity',
            'clip_image_similarity',
            'lpips_distance',
            'ssim_whole_image',
            'final_score'
        ]
        
        viz_df = df[viz_columns]
        viz_df.to_csv(output_path, index=False)
        
        print(f"✅ Exported visualization data: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a summary of logged data"""
        df = self.get_dataframe()
        
        if df.empty:
            print("📊 No data logged yet")
            return
        
        print("\n" + "="*70)
        print("📊 EVALUATION DATA SUMMARY")
        print("="*70)
        
        print(f"\nTotal Samples: {len(df)}")
        
        for category in ['Basic', 'Attribute', 'Boundary']:
            cat_df = df[df['sample_category'] == category]
            if len(cat_df) > 0:
                print(f"\n{category} Consistency: {len(cat_df)} samples")
                print(f"  ├─ DeepFace Similarity: {cat_df['deepface_similarity'].mean():.3f} ± {cat_df['deepface_similarity'].std():.3f}")
                print(f"  ├─ CLIP Similarity: {cat_df['clip_image_similarity'].mean():.3f} ± {cat_df['clip_image_similarity'].std():.3f}")
                print(f"  ├─ LPIPS Distance: {cat_df['lpips_distance'].mean():.3f} ± {cat_df['lpips_distance'].std():.3f}")
                print(f"  └─ SSIM: {cat_df['ssim_whole_image'].mean():.3f} ± {cat_df['ssim_whole_image'].std():.3f}")
        
        print("\n" + "="*70 + "\n")


def test_data_logger():
    """Test the data logger"""
    print("\n🧪 Testing Data Logger")
    print("-" * 50)
    
    logger = EvaluationDataLogger(data_dir="evaluation_data")
    
    # Test logging different categories
    test_results = {
        'Identity_Similarity': 0.85,
        'Identity_Confidence': 0.92,
        'Identity_Decision': 'Same Person',
        'Models_Used': 3,
        'CLIP_Similarity': 0.78,
        'LPIPS_Similarity': 0.75,
        'SSIM': 0.82,
        'PSNR': 28.5,
        'MSE': 150.0,
        'Histogram_Similarity': 0.65,
        'Final_Score': 0.80,
        'Consistency_Level': 'High',
        'Overall_Confidence': 0.88
    }
    
    # Log samples
    logger.log_evaluation(test_results, 'Basic', image1_name='test1.jpg', image2_name='test2.jpg')
    logger.log_evaluation(test_results, 'Attribute', image1_name='test3.jpg', image2_name='test4.jpg')
    logger.log_evaluation(test_results, 'Boundary', image1_name='test5.jpg', image2_name='test6.jpg')
    
    # Print summary
    logger.print_summary()
    
    # Get statistics
    stats = logger.get_summary_statistics()
    print("\n📈 Summary Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Validate hypotheses
    validation = logger.validate_hypothesis()
    print("\n✅ Hypothesis Validation:")
    print(json.dumps(validation, indent=2))
    
    print("\n✅ Data logger test completed")


if __name__ == "__main__":
    test_data_logger()
