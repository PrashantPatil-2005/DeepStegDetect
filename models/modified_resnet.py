"""
Modified ResNet models with SRM preprocessing frontend for steganalysis.
Supports ResNet18, ResNet50, and ResNet101 with transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
from pathlib import Path

# Add scripts directory to path to import SRMFrontend
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))
from srm_frontend import SRMFrontend


class ResNetWithSRM(nn.Module):
    """
    ResNet architecture with SRM preprocessing frontend.
    The SRM frontend extracts residual features before passing to ResNet backbone.
    """
    def __init__(self, model_name='resnet50', pretrained=True, freeze_until='layer3', device='cpu'):
        super(ResNetWithSRM, self).__init__()
        
        # SRM frontend (non-trainable residual filters)
        self.srm = SRMFrontend()
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif model_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            in_features = 2048
        elif model_name == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose from resnet18, resnet50, resnet101")
        
        # Remove the original first conv layer and maxpool
        # We'll replace it with SRM + adapter
        self.conv1_adapter = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Copy remaining layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Replace classifier head for binary classification
        self.fc = nn.Linear(in_features, 1)
        
        # Freezing strategy
        if freeze_until == 'layer1':
            for param in self.layer1.parameters():
                param.requires_grad = False
        elif freeze_until == 'layer2':
            for param in list(self.layer1.parameters()) + list(self.layer2.parameters()):
                param.requires_grad = False
        elif freeze_until == 'layer3':
            for param in list(self.layer1.parameters()) + list(self.layer2.parameters()) + list(self.layer3.parameters()):
                param.requires_grad = False
        elif freeze_until == 'none':
            # All layers trainable
            pass
        
    def forward(self, x):
        # x: [B, 3, H, W] RGB image in [0, 1]
        
        # Apply SRM frontend to get residual features
        residual = self.srm(x)  # [B, 5, H, W]
        
        # Adapter: map 5 SRM channels to 64 channels (matching ResNet's first conv)
        x = self.conv1_adapter(residual)  # [B, 64, H, W]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Binary classification head
        x = self.fc(x)
        
        return x


def create_model(model_name='resnet50', pretrained=True, freeze_until='layer3', device='cpu'):
    """
    Factory function to create a ResNet model with SRM preprocessing.
    
    Args:
        model_name: 'resnet18', 'resnet50', or 'resnet101'
        pretrained: Whether to use ImageNet pretrained weights
        freeze_until: Which layers to freeze ('none', 'layer1', 'layer2', 'layer3')
        device: 'cpu' or 'cuda'
    
    Returns:
        ResNetWithSRM model
    """
    model = ResNetWithSRM(
        model_name=model_name,
        pretrained=pretrained,
        freeze_until=freeze_until,
        device=device
    )
    return model.to(device)

