"""
Grad-CAM visualization for steganalysis model interpretability.
Generates attention heatmaps showing which regions the model focuses on.

Usage:
    python scripts/gradcam_visualization.py --img test_image.png --model checkpoints/resnet50_best.pth
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import sys
from pathlib import Path
from torchvision import transforms

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.modified_resnet import create_model


class GradCAM:
    """Grad-CAM implementation for model interpretability"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_layers()
    
    def hook_layers(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks on target layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        # For binary classification, we need to backpropagate from the positive class
        if output.dim() > 1:
            output = output.squeeze()
        output.backward(torch.ones_like(output), retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def load_image(image_path, img_size=224):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img


def visualize_gradcam(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    # Load model
    print(f"📦 Loading model from: {args.model}")
    ckpt = torch.load(args.model, map_location=device)
    model_name = ckpt.get('model_name', 'resnet50')
    model = create_model(model_name=model_name, pretrained=False, freeze_until='none', device=device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    print(f"✅ Loaded {model_name} model")
    
    # Get target layer (last conv layer before classifier)
    # ResNet50/101 use conv3, ResNet18 uses conv2
    if hasattr(model.layer4[-1], 'conv3'):
        target_layer = model.layer4[-1].conv3
    elif hasattr(model.layer4[-1], 'conv2'):
        target_layer = model.layer4[-1].conv2
    else:
        # Fallback to the last layer
        target_layer = model.layer4[-1]
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Load image
    print(f"📷 Loading image: {args.img}")
    img_tensor, original_img = load_image(args.img, args.img_size)
    img_tensor = img_tensor.to(device)
    
    # Generate CAM
    print("🔄 Generating Grad-CAM heatmap...")
    cam = gradcam.generate_cam(img_tensor)
    
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (original_img.size[0], original_img.size[1]))
    cam_resized = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    original_np = np.array(original_img)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "STEGO" if prob > 0.5 else "COVER"
        confidence = prob if prob > 0.5 else 1 - prob
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original Image\nPrediction: {prediction} ({confidence:.2%})')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"💾 Grad-CAM visualization saved to: {args.output}")
    plt.close()
    
    print(f"\n📊 Prediction: {prediction} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization for steganalysis model")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for model input")
    parser.add_argument("--output", type=str, default="outputs/gradcam_output.png",
                        help="Output path for Grad-CAM visualization")
    
    args = parser.parse_args()
    visualize_gradcam(args)

