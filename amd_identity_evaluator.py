#!/usr/bin/env python3
"""
AMD-optimized Identity Evaluator using OpenCV
Designed for AMD 780M integrated graphics
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os

class AMDIdentityEvaluator:
    """
    AMD-optimized Identity Evaluator using OpenCV
    Designed for AMD 780M integrated graphics
    """
    
    def __init__(self):
        """Initialize OpenCV face detection for AMD systems"""
        
        # Initialize OpenCV face detection
        # Use Haar cascades for lightweight face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.detection_method = 'haar'
            print("✅ AMD Identity Evaluator initialized with Haar Cascades")
        else:
            # Fallback to basic template matching
            self.face_cascade = None
            self.detection_method = 'template'
            print("⚠️ Using template matching fallback for face detection")
        
        # Configure for AMD performance
        self.scale_factor = 1.1  # Faster detection
        self.min_neighbors = 3   # Reduced for speed
        self.min_size = (30, 30)  # Minimum face size
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in image using OpenCV
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            list: List of face bounding boxes [(x, y, w, h), ...]
        """
        
        if self.detection_method == 'haar' and self.face_cascade is not None:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces.tolist() if len(faces) > 0 else []
        
        else:
            # Fallback: assume single face covering most of the image
            h, w = image.shape[:2]
            return [(w//4, h//4, w//2, h//2)]
    
    def extract_face_features(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract improved features from detected face region
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            np.ndarray: Feature vector
        """
        
        x, y, w, h = face_box
        
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return np.zeros(256)  # Return zero vector if no face
        
        # Resize to standard size for consistent feature extraction
        face_resized = cv2.resize(face_region, (128, 128))
        
        # Convert to grayscale
        if len(face_resized.shape) == 3:
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_resized
        
        # Normalize lighting
        face_gray = cv2.equalizeHist(face_gray)
        
        # Extract multiple types of features
        features = []
        
        # 1. Gradient-based features (improved edge information)
        sobel_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Extract gradient statistics from different regions
        h, w = face_gray.shape
        regions = [
            face_gray[0:h//3, 0:w//3],        # Top-left (forehead)
            face_gray[0:h//3, w//3:2*w//3],   # Top-center (forehead)
            face_gray[0:h//3, 2*w//3:w],      # Top-right (forehead)
            face_gray[h//3:2*h//3, 0:w//3],   # Mid-left (left eye area)
            face_gray[h//3:2*h//3, w//3:2*w//3], # Mid-center (nose area)
            face_gray[h//3:2*h//3, 2*w//3:w], # Mid-right (right eye area)
            face_gray[2*h//3:h, 0:w//3],      # Bottom-left (left mouth)
            face_gray[2*h//3:h, w//3:2*w//3], # Bottom-center (mouth)
            face_gray[2*h//3:h, 2*w//3:w],    # Bottom-right (right mouth)
        ]
        
        for region in regions:
            if region.size > 0:
                # Statistical features for each region
                features.extend([
                    np.mean(region),
                    np.std(region),
                    np.median(region),
                ])
            else:
                features.extend([0, 0, 0])
        
        # 2. Texture features using LBP (improved)
        lbp_features = self._calculate_improved_lbp_features(face_gray)
        features.extend(lbp_features[:64])  # Limit to 64 features
        
        # 3. Geometric features (face proportions)
        geometric_features = self._calculate_geometric_features(face_gray)
        features.extend(geometric_features)
        
        # 4. Intensity distribution features
        hist = cv2.calcHist([face_gray], [0], None, [32], [0, 256])
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-7)  # Normalize
        features.extend(hist[:32])
        
        # Ensure fixed size feature vector
        features = np.array(features[:256])
        if len(features) < 256:
            features = np.pad(features, (0, 256 - len(features)), 'constant')
        
        return features
    
    def _calculate_improved_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Calculate improved Local Binary Pattern features"""
        
        h, w = image.shape
        features = []
        
        # Calculate LBP for different radii
        for radius in [1, 2, 3]:
            for n_points in [8, 16]:
                if radius * 2 + 1 <= min(h, w):
                    try:
                        # Simple LBP approximation
                        lbp_image = self._simple_lbp(image, radius, n_points)
                        
                        # Calculate histogram
                        hist, _ = np.histogram(lbp_image.ravel(), bins=min(n_points + 2, 32), 
                                             range=(0, n_points + 2))
                        hist = hist / (np.sum(hist) + 1e-7)
                        features.extend(hist)
                    except:
                        features.extend([0] * min(n_points + 2, 32))
        
        return np.array(features)
    
    def _simple_lbp(self, image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """Simple LBP implementation"""
        
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                binary_string = []
                
                # Sample points around the center
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if 0 <= x < h and 0 <= y < w:
                        binary_string.append(1 if image[x, y] >= center else 0)
                    else:
                        binary_string.append(0)
                
                # Convert binary string to decimal
                lbp[i, j] = sum([bit * (2 ** idx) for idx, bit in enumerate(binary_string)])
        
        return lbp
    
    def _calculate_geometric_features(self, image: np.ndarray) -> list:
        """Calculate geometric features of the face"""
        
        h, w = image.shape
        features = []
        
        # Face aspect ratio
        features.append(w / h)
        
        # Symmetry features (compare left and right halves)
        left_half = image[:, :w//2]
        right_half = np.fliplr(image[:, w//2:])
        
        # Resize right half to match left half
        if right_half.shape[1] != left_half.shape[1]:
            right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
        
        # Calculate symmetry score
        if left_half.shape == right_half.shape:
            symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            if np.isnan(symmetry):
                symmetry = 0.0
            features.append(symmetry)
        else:
            features.append(0.0)
        
        # Vertical thirds analysis (forehead, eyes-nose, mouth-chin)
        third_h = h // 3
        upper_third = image[:third_h, :]
        middle_third = image[third_h:2*third_h, :]
        lower_third = image[2*third_h:, :]
        
        # Calculate variance for each third
        features.extend([
            np.var(upper_third),
            np.var(middle_third),
            np.var(lower_third)
        ])
        
        # Intensity distribution across face regions
        center_region = image[h//4:3*h//4, w//4:3*w//4]
        edge_region_mask = np.ones_like(image, dtype=bool)
        edge_region_mask[h//4:3*h//4, w//4:3*w//4] = False
        edge_region = image[edge_region_mask]
        
        if center_region.size > 0 and edge_region.size > 0:
            center_mean = np.mean(center_region)
            edge_mean = np.mean(edge_region)
            features.append(center_mean - edge_mean)
        else:
            features.append(0.0)
        
        return features
    
    def calculate_identity_similarity(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, Any]:
        """
        Calculate identity similarity between two images
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            dict: Similarity metrics
        """
        
        # Detect faces in both images
        faces1 = self.detect_faces(image1)
        faces2 = self.detect_faces(image2)
        
        if not faces1 or not faces2:
            return {
                'similarity': 0.0,
                'confidence': 0.0,
                'faces_detected': [len(faces1), len(faces2)],
                'method': self.detection_method
            }
        
        # Use largest face from each image
        face1 = max(faces1, key=lambda f: f[2] * f[3])  # Largest by area
        face2 = max(faces2, key=lambda f: f[2] * f[3])
        
        # Extract features
        features1 = self.extract_face_features(image1, face1)
        features2 = self.extract_face_features(image2, face2)
        
        # Calculate similarity
        similarity = self._calculate_feature_similarity(features1, features2)
        
        # Calculate confidence based on face size and detection quality
        face1_area = face1[2] * face1[3]
        face2_area = face2[2] * face2[3]
        img1_area = image1.shape[0] * image1.shape[1]
        img2_area = image2.shape[0] * image2.shape[1]
        
        face1_ratio = face1_area / img1_area
        face2_ratio = face2_area / img2_area
        
        confidence = min(face1_ratio, face2_ratio) * 2  # Scale to reasonable range
        confidence = min(confidence, 1.0)
        
        return {
            'similarity': float(similarity),
            'confidence': float(confidence),
            'faces_detected': [len(faces1), len(faces2)],
            'face_areas': [face1_area, face2_area],
            'method': self.detection_method
        }
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate improved similarity between feature vectors"""
        
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Ensure same length
        min_len = min(len(features1), len(features2))
        features1 = features1[:min_len]
        features2 = features2[:min_len]
        
        # Normalize features
        features1 = features1 / (np.linalg.norm(features1) + 1e-7)
        features2 = features2 / (np.linalg.norm(features2) + 1e-7)
        
        # Calculate multiple similarity metrics
        similarities = []
        
        # 1. Cosine similarity
        cosine_sim = np.dot(features1, features2)
        similarities.append(max(0.0, cosine_sim))
        
        # 2. Euclidean distance (converted to similarity)
        euclidean_dist = np.linalg.norm(features1 - features2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        similarities.append(euclidean_sim)
        
        # 3. Correlation coefficient
        if np.std(features1) > 1e-7 and np.std(features2) > 1e-7:
            correlation = np.corrcoef(features1, features2)[0, 1]
            if not np.isnan(correlation):
                correlation_sim = max(0.0, correlation)
                similarities.append(correlation_sim)
        
        # 4. Manhattan distance (converted to similarity)
        manhattan_dist = np.sum(np.abs(features1 - features2))
        manhattan_sim = 1.0 / (1.0 + manhattan_dist)
        similarities.append(manhattan_sim)
        
        # Calculate weighted average (cosine similarity gets higher weight)
        if len(similarities) >= 4:
            final_similarity = (
                0.4 * similarities[0] +  # Cosine similarity
                0.25 * similarities[1] + # Euclidean similarity
                0.2 * similarities[2] +  # Correlation similarity
                0.15 * similarities[3]   # Manhattan similarity
            )
        else:
            final_similarity = np.mean(similarities)
        
        # Apply threshold to make discrimination more strict
        # If similarity is below certain threshold, reduce it significantly
        if final_similarity < 0.7:
            final_similarity = final_similarity * 0.5  # Make low similarities even lower
        
        # Apply non-linear transformation to enhance differences
        # This makes high similarities higher and low similarities lower
        final_similarity = final_similarity ** 1.5
        
        # Ensure in [0, 1] range
        final_similarity = max(0.0, min(1.0, final_similarity))
        
        return final_similarity
    
    def batch_evaluate(self, image_pairs: list) -> list:
        """
        Evaluate multiple image pairs
        
        Args:
            image_pairs: List of (image1, image2) tuples
            
        Returns:
            list: List of similarity results
        """
        
        results = []
        
        for i, (img1, img2) in enumerate(image_pairs):
            try:
                result = self.calculate_identity_similarity(img1, img2)
                result['pair_index'] = i
                results.append(result)
            except Exception as e:
                print(f"⚠️ Error processing pair {i}: {e}")
                results.append({
                    'similarity': 0.0,
                    'confidence': 0.0,
                    'faces_detected': [0, 0],
                    'method': self.detection_method,
                    'pair_index': i,
                    'error': str(e)
                })
        
        return results

# Test function
def test_amd_identity_evaluator():
    """Test the AMD identity evaluator"""
    
    print("\n🧪 Testing AMD Identity Evaluator")
    print("-" * 40)
    
    # Create evaluator
    evaluator = AMDIdentityEvaluator()
    
    # Create test images
    test_img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test face detection
    print("Testing face detection...")
    faces1 = evaluator.detect_faces(test_img1)
    faces2 = evaluator.detect_faces(test_img2)
    
    print(f"Faces detected in image 1: {len(faces1)}")
    print(f"Faces detected in image 2: {len(faces2)}")
    
    # Test similarity calculation
    print("Testing similarity calculation...")
    result = evaluator.calculate_identity_similarity(test_img1, test_img2)
    
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Method: {result['method']}")
    
    print("✅ AMD Identity Evaluator test completed")

if __name__ == "__main__":
    test_amd_identity_evaluator()