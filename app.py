#!/usr/bin/env python3
"""
Image Consistency Evaluation System Main Entry
Run this file to start the Gradio Web interface
"""

import sys
import os
import argparse
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from dashboard import GradioEvaluationApp
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Image Consistency Evaluation System')
    
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Server host address (default: 127.0.0.1)')
    
    parser.add_argument('--port', type=int, default=7860,
                       help='Server port number (default: 7860)')
    
    parser.add_argument('--share', action='store_true',
                       help='Create public shareable link')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not automatically open browser')
    
    return parser.parse_args()

def check_dependencies():
    """Check dependency packages"""
    required_packages = [
        'torch', 'torchvision', 'transformers', 'clip',
        'gradio', 'opencv-python', 'numpy', 'pandas',
        'pillow', 'scikit-image', 'matplotlib', 'seaborn',
        'lpips', 'face-recognition', 'facenet-pytorch', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                from PIL import Image
            elif package == 'scikit-image':
                from skimage import metrics
            elif package == 'clip':
                import clip
            elif package == 'face-recognition':
                import face_recognition
            elif package == 'facenet-pytorch':
                from facenet_pytorch import MTCNN
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing the following dependency packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease run the following command to install:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependency packages are installed")
    return True

def print_banner():
    """Print startup banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                Image Consistency Evaluation System           ║
    ║                                                              ║
    ║  Core Features:                                              ║
    ║  • CLIP Semantic Consistency Evaluation                     ║
    ║  • Face Identity Consistency Evaluation                     ║
    ║  • LPIPS Perceptual Similarity Evaluation                   ║
    ║  • Traditional Image Quality Metrics (SSIM, PSNR)          ║
    ║  • Interactive Web Interface                                ║
    ║  • Batch Evaluation Support                                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """主函数"""
    # 打印启动横幅
    print_banner()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志
    if args.debug:
        setup_logging()
        logging.info("Debug mode enabled")
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    try:
        # Create application instance
        print("Initializing evaluation system...")
        app = GradioEvaluationApp()        # 启动参数
        launch_kwargs = {
            'server_name': args.host,
            'server_port': args.port,
            'share': args.share,
            'inbrowser': not args.no_browser,
            'show_error': args.debug
        }
        
        print(f"🚀 Starting server...")
        print(f"   Address: http://{args.host}:{args.port}")
        if args.share:
            print("   Public link will be displayed after startup")
        print("   Press Ctrl+C to stop the server")
        print()
        
        # Launch application
        app.launch(**launch_kwargs)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()