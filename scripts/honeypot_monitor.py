"""
Honeypot-based active defense monitoring system.
Monitors a directory for new image uploads and automatically scans them for steganography.

Usage:
    python scripts/honeypot_monitor.py --watch honeypot_watch/ --model checkpoints/resnet50_best.pth
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import time
import hashlib
import json
from datetime import datetime
from PIL import Image
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from torchvision import transforms

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.modified_resnet import create_model


class StegoDetector:
    """Steganography detector using trained model"""
    def __init__(self, model_path, device='cpu', img_size=224):
        self.device = device
        self.img_size = img_size
        
        # Load model
        print(f"📦 Loading model from: {model_path}")
        ckpt = torch.load(model_path, map_location=device)
        model_name = ckpt.get('model_name', 'resnet50')
        self.model = create_model(model_name=model_name, pretrained=False, freeze_until='none', device=device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(device)
        self.model.eval()
        print(f"✅ Loaded {model_name} model")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
    
    def detect(self, image_path):
        """Detect steganography in an image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                prob = torch.sigmoid(output).item()
                prediction = "STEGO" if prob > 0.5 else "COVER"
                confidence = prob if prob > 0.5 else 1 - prob
            
            return prediction, confidence, prob
        except Exception as e:
            return None, None, None


class HoneypotHandler(FileSystemEventHandler):
    """File system event handler for honeypot monitoring"""
    def __init__(self, detector, log_file='honeypot_log.json', alert_threshold=0.5):
        self.detector = detector
        self.log_file = log_file
        self.alert_threshold = alert_threshold
        self.processed_files = set()
        
        # Load existing log
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.log = json.load(f)
        else:
            self.log = []
        
        # Image extensions
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pgm'}
    
    def compute_hash(self, file_path):
        """Compute SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def is_image(self, file_path):
        """Check if file is an image"""
        return Path(file_path).suffix.lower() in self.image_extensions
    
    def on_created(self, event):
        """Handle file creation event"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's an image
        if not self.is_image(file_path):
            return
        
        # Wait a bit for file to be fully written
        time.sleep(0.5)
        
        # Check if file exists and is readable
        if not file_path.exists() or not file_path.is_file():
            return
        
        # Compute hash
        file_hash = self.compute_hash(file_path)
        
        # Skip if already processed
        if file_hash in self.processed_files:
            return
        
        self.processed_files.add(file_hash)
        
        # Detect steganography
        print(f"\n{'='*60}")
        print(f"🔍 [ALERT] New file detected: {file_path.name}")
        print(f"📁 Path: {file_path}")
        print(f"🔐 SHA-256: {file_hash[:16]}...")
        
        prediction, confidence, prob = self.detector.detect(file_path)
        
        if prediction is None:
            print(f"❌ Error processing image")
            return
        
        # Log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": file_path.name,
            "filepath": str(file_path),
            "hash": file_hash,
            "prediction": prediction,
            "confidence": confidence,
            "probability": prob,
            "alert": prediction == "STEGO" and confidence >= self.alert_threshold
        }
        
        self.log.append(log_entry)
        
        # Save log
        with open(self.log_file, 'w') as f:
            json.dump(self.log, f, indent=2)
        
        # Print results
        print(f"🎯 Prediction: {prediction}")
        print(f"📊 Confidence: {confidence:.2%}")
        print(f"📈 Probability: {prob:.4f}")
        
        if log_entry["alert"]:
            print(f"🚨 [SECURITY ALERT] Suspicious image detected!")
            print(f"   This image may contain hidden steganographic content.")
        else:
            print(f"✅ Image appears clean")
        
        print(f"{'='*60}\n")


def monitor_directory(watch_dir, model_path, log_file='honeypot_log.json', alert_threshold=0.5):
    """Start monitoring directory for new files"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    # Create detector
    detector = StegoDetector(model_path, device=device)
    
    # Create event handler
    event_handler = HoneypotHandler(detector, log_file=log_file, alert_threshold=alert_threshold)
    
    # Create observer
    observer = Observer()
    observer.schedule(event_handler, str(watch_dir), recursive=False)
    observer.start()
    
    print(f"\n{'='*60}")
    print(f"🛡️  Honeypot Active Defense System Started")
    print(f"{'='*60}")
    print(f"📁 Monitoring directory: {watch_dir}")
    print(f"🤖 Model: {model_path}")
    print(f"📊 Log file: {log_file}")
    print(f"🚨 Alert threshold: {alert_threshold:.2%}")
    print(f"\n⏳ Waiting for new image uploads...")
    print(f"   (Press Ctrl+C to stop)\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n\n🛑 Stopping honeypot monitor...")
        observer.stop()
    
    observer.join()
    print(f"✅ Honeypot monitor stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Honeypot-based active defense monitoring system")
    parser.add_argument("--watch", type=str, required=True, 
                        help="Directory to monitor for new image uploads")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--log", type=str, default="honeypot_log.json",
                        help="Path to log file for storing detection results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for security alerts (0.0-1.0)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size for model input")
    
    args = parser.parse_args()
    
    # Check if watch directory exists
    watch_path = Path(args.watch)
    if not watch_path.exists():
        print(f"📁 Creating watch directory: {watch_path}")
        watch_path.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        sys.exit(1)
    
    monitor_directory(args.watch, args.model, args.log, args.threshold)

