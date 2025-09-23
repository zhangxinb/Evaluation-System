# API Reference Documentation

## Table of Contents
1. [Core Classes](#core-classes)
2. [Main Functions](#main-functions)
3. [Utility Functions](#utility-functions)
4. [Configuration](#configuration)
5. [Data Structures](#data-structures)
6. [Error Codes](#error-codes)

## Core Classes

### ProfessionalIdentityEvaluator

Main class for professional face recognition and identity verification.

```python
class ProfessionalIdentityEvaluator:
    """
    Professional identity evaluation using multiple deep learning models
    
    Attributes:
        models (list): List of available DeepFace models
        confidence_threshold (float): Minimum confidence for reliable results
        similarity_threshold (float): Threshold for same/different person decision
    """
    
    def __init__(self, confidence_threshold=0.7, similarity_threshold=0.6):
        """
        Initialize the evaluator
        
        Args:
            confidence_threshold (float): Minimum confidence score (0.0-1.0)
            similarity_threshold (float): Identity similarity threshold (0.0-1.0)
        """
    
    def evaluate_identity(self, image1_path, image2_path):
        """
        Comprehensive identity evaluation
        
        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
            
        Returns:
            dict: Evaluation results containing:
                - Identity_Similarity (float): Similarity score (0.0-1.0)
                - Identity_Confidence (float): Analysis confidence (0.0-1.0)
                - Identity_Decision (str): "Same Person" or "Different Person"
                - Decision_Confidence (float): Decision confidence (0.0-1.0)
                - Models_Used (int): Number of models successfully used
                - Detection_Method (str): "DeepFace Professional"
                - Age_Estimate (int): Estimated age in years
                - Gender_Estimate (str): "Man" or "Woman"
                - Emotion_Detected (str): Detected emotion
        
        Raises:
            FileNotFoundError: If image files don't exist
            ValueError: If images cannot be processed
        """
    
    def verify_faces(self, image1_path, image2_path):
        """
        Basic face verification
        
        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
            
        Returns:
            dict: Verification results
        """
    
    def analyze_demographics(self, image_path):
        """
        Demographic analysis of a single image
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Demographic analysis results
        """
```

### ImageQualityAssessor

Class for traditional image quality metrics calculation.

```python
class ImageQualityAssessor:
    """
    Traditional image quality assessment using multiple metrics
    """
    
    @staticmethod
    def calculate_ssim(image1, image2):
        """
        Calculate Structural Similarity Index Measure
        
        Args:
            image1 (numpy.ndarray): First image array
            image2 (numpy.ndarray): Second image array
            
        Returns:
            float: SSIM score (0.0-1.0)
        """
    
    @staticmethod
    def calculate_psnr(image1, image2):
        """
        Calculate Peak Signal-to-Noise Ratio
        
        Args:
            image1 (numpy.ndarray): First image array
            image2 (numpy.ndarray): Second image array
            
        Returns:
            float: PSNR score in decibels
        """
    
    @staticmethod
    def calculate_mse(image1, image2):
        """
        Calculate Mean Squared Error
        
        Args:
            image1 (numpy.ndarray): First image array
            image2 (numpy.ndarray): Second image array
            
        Returns:
            float: MSE score
        """
    
    @staticmethod
    def calculate_histogram_similarity(image1, image2):
        """
        Calculate color histogram correlation
        
        Args:
            image1 (numpy.ndarray): First image array
            image2 (numpy.ndarray): Second image array
            
        Returns:
            float: Histogram correlation (0.0-1.0)
        """
```

## Main Functions

### evaluate_images_professional()

Main evaluation function optimized for integrated graphics hardware.

```python
def evaluate_images_professional(image1_path, image2_path):
    """
    Comprehensive image evaluation optimized for integrated graphics
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        
    Returns:
        dict: Complete evaluation results containing:
            - Traditional metrics (SSIM, PSNR, MSE)
            - Professional identity analysis
            - Demographic information
            - Color analysis
            - Processing metadata
            
    Raises:
        FileNotFoundError: If image files don't exist
        ValueError: If images cannot be processed
        RuntimeError: If evaluation fails
        
    Example:
        >>> results = evaluate_images_professional("person1.jpg", "person2.jpg")
        >>> print(f"Identity similarity: {results['Identity_Similarity']:.4f}")
        >>> print(f"Decision: {results['Identity_Decision']}")
    """
```

### format_results()

Format evaluation results for display.

```python
def format_results(results, img1_shape, img2_shape):
    """
    Format evaluation results for user-friendly display
    
    Args:
        results (dict): Raw evaluation results
        img1_shape (tuple): Shape of first image (height, width, channels)
        img2_shape (tuple): Shape of second image (height, width, channels)
        
    Returns:
        str: Formatted result string with proper spacing and icons
        
    Example:
        >>> formatted = format_results(results, (480, 640, 3), (720, 1080, 3))
        >>> print(formatted)
    """
```

## Utility Functions

### Image Processing

```python
def load_and_preprocess_image(image_path, target_size=None):
    """
    Load and preprocess image for analysis
    
    Args:
        image_path (str): Path to image file
        target_size (tuple, optional): Target size (width, height)
        
    Returns:
        numpy.ndarray: Preprocessed image array
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """

def resize_image_for_comparison(image1, image2):
    """
    Resize images to same dimensions for comparison
    
    Args:
        image1 (numpy.ndarray): First image array
        image2 (numpy.ndarray): Second image array
        
    Returns:
        tuple: (resized_image1, resized_image2)
    """

def validate_image_format(image_path):
    """
    Validate image format and quality
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Validation results with 'valid', 'warnings', 'errors' keys
    """
```

### Hardware Optimization

```python
def configure_integrated_graphics_optimization():
    """
    Configure system for integrated graphics optimal performance
    
    Sets up:
        - CPU threading configuration
        - Memory management
        - TensorFlow/PyTorch backend settings
        
    Returns:
        dict: Configuration status
    """

def optimize_memory_usage():
    """
    Optimize memory usage for integrated graphics
    
    Performs:
        - Garbage collection optimization
        - Memory growth configuration
        - Cache management
    """

def get_system_performance_info():
    """
    Get current system performance metrics
    
    Returns:
        dict: Performance information including:
            - Memory usage
            - CPU utilization
            - Available GPU memory
            - Processing thread count
    """
```

## Configuration

### Environment Variables

```python
# TensorFlow Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

# OpenMP Configuration
os.environ['OMP_NUM_THREADS'] = '8'  # Optimize for AMD 780M CPU cores
os.environ['MKL_NUM_THREADS'] = '8'  # Intel MKL optimization

# DeepFace Configuration
os.environ['DEEPFACE_LOG_LEVEL'] = 'ERROR'  # Reduce DeepFace logging
```

### Model Configuration

```python
# Available DeepFace models
AVAILABLE_MODELS = [
    'VGG-Face',    # VGG-16 based face recognition
    'Facenet',     # Google's FaceNet
    'OpenFace',    # CMU OpenFace
    'DeepFace'     # Facebook's DeepFace
]

# Detection backends
DETECTION_BACKENDS = [
    'opencv',      # OpenCV Haar cascades (default)
    'mtcnn',       # Multi-task CNN
    'retinaface',  # RetinaFace detector
    'dlib'         # Dlib face detector
]

# Analysis actions
ANALYSIS_ACTIONS = [
    'age',         # Age estimation
    'gender',      # Gender classification
    'emotion',     # Emotion recognition
    'race'         # Race/ethnicity detection
]
```

## Data Structures

### EvaluationResult

```python
class EvaluationResult:
    """
    Structured result from image evaluation
    
    Attributes:
        ssim (float): Structural Similarity Index
        psnr (float): Peak Signal-to-Noise Ratio
        mse (float): Mean Squared Error
        identity_similarity (float): Face similarity score
        identity_confidence (float): Analysis confidence
        identity_decision (str): Same/Different person decision
        decision_confidence (float): Decision confidence
        models_used (int): Number of models used
        detection_method (str): Detection method used
        age_estimate (int): Estimated age
        gender_estimate (str): Detected gender
        emotion_detected (str): Detected emotion
        histogram_similarity (float): Color histogram correlation
        processing_time (float): Total processing time in seconds
        warnings (list): List of warning messages
        errors (list): List of error messages
    """
```

### DemographicAnalysis

```python
class DemographicAnalysis:
    """
    Demographic analysis results
    
    Attributes:
        age (int): Estimated age in years
        age_confidence (float): Age estimation confidence
        gender (str): Detected gender ("Man" or "Woman")
        gender_confidence (float): Gender classification confidence
        emotion (str): Dominant emotion
        emotion_scores (dict): All emotion scores
        race (str): Detected race/ethnicity
        race_scores (dict): All race classification scores
    """
```

### SystemConfiguration

```python
class SystemConfiguration:
    """
    System configuration parameters
    
    Attributes:
        device (str): Processing device ("cpu" or "cuda")
        cpu_threads (int): Number of CPU threads
        memory_limit (int): Memory limit in MB
        batch_size (int): Processing batch size
        image_max_size (tuple): Maximum image dimensions
        confidence_threshold (float): Minimum confidence threshold
        similarity_threshold (float): Identity similarity threshold
    """
```

## Error Codes

### Exception Types

```python
class ImageEvaluationError(Exception):
    """Base exception for image evaluation errors"""
    pass

class ModelLoadingError(ImageEvaluationError):
    """Raised when deep learning models fail to load"""
    pass

class FaceDetectionError(ImageEvaluationError):
    """Raised when face detection fails"""
    pass

class InsufficientQualityError(ImageEvaluationError):
    """Raised when image quality is insufficient for analysis"""
    pass

class HardwareCompatibilityError(ImageEvaluationError):
    """Raised when hardware optimization fails"""
    pass
```

### Error Response Format

```python
{
    "success": False,
    "error_code": "FACE_DETECTION_FAILED",
    "error_message": "No faces detected in one or both images",
    "error_details": {
        "image1_faces": 0,
        "image2_faces": 1,
        "detector_used": "opencv",
        "suggestions": [
            "Ensure faces are clearly visible",
            "Check image lighting conditions",
            "Try different detection backend"
        ]
    },
    "timestamp": "2025-09-23T19:30:00Z"
}
```

### Common Error Codes

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `FILE_NOT_FOUND` | Image file doesn't exist | Check file path |
| `INVALID_FORMAT` | Unsupported image format | Use JPG, PNG, BMP, or TIFF |
| `FACE_DETECTION_FAILED` | No faces detected | Improve image quality/lighting |
| `MODEL_LOADING_ERROR` | Deep learning model failed to load | Check internet connection, reinstall dependencies |
| `INSUFFICIENT_MEMORY` | Not enough memory for processing | Reduce image size or restart application |
| `PROCESSING_TIMEOUT` | Analysis took too long | Reduce image size or check system performance |
| `HARDWARE_ERROR` | Integrated graphics optimization failed | Check drivers and system configuration |

## Usage Examples

### Basic Usage

```python
from professional_identity_evaluator import ProfessionalIdentityEvaluator

# Initialize evaluator
evaluator = ProfessionalIdentityEvaluator()

# Evaluate two images
results = evaluator.evaluate_identity("person1.jpg", "person2.jpg")

# Check if same person
if results['Identity_Decision'] == "Same Person":
    print(f"Same person detected (confidence: {results['Decision_Confidence']:.2f})")
else:
    print(f"Different people (similarity: {results['Identity_Similarity']:.4f})")
```

### Advanced Usage

```python
# Configure for high accuracy
evaluator = ProfessionalIdentityEvaluator(
    confidence_threshold=0.8,
    similarity_threshold=0.7
)

# Full evaluation with error handling
try:
    results = evaluator.evaluate_identity("image1.jpg", "image2.jpg")
    
    # Access detailed results
    print(f"Identity Similarity: {results['Identity_Similarity']:.4f}")
    print(f"Age Estimate: {results['Age_Estimate']} years")
    print(f"Gender: {results['Gender_Estimate']}")
    print(f"Emotion: {results['Emotion_Detected']}")
    
except FileNotFoundError as e:
    print(f"Image file not found: {e}")
except FaceDetectionError as e:
    print(f"Face detection failed: {e}")
except Exception as e:
    print(f"Evaluation error: {e}")
```

---

**API Documentation Version**: 1.0  
**Compatible With**: Professional Image Evaluation System  
**Last Updated**: September 23, 2025