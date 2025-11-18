"""
Models package for DeepStegDetect.
Contains ResNet models with SRM preprocessing for steganalysis.
"""

from .modified_resnet import ResNetWithSRM, create_model

__all__ = ['ResNetWithSRM', 'create_model']

