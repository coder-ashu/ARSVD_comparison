# models/vgg_cifar.py
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn

def vgg16_cifar(num_classes=10, pretrained=False, batch_norm=True):
    """
    Use torchvision VGG16-BN backbone but replace the classifier to suit CIFAR-10 (32x32).
    """
    if batch_norm:
        model = vgg16_bn(pretrained=pretrained)
    else:
        from torchvision.models import vgg16
        model = vgg16(pretrained=pretrained)

    # Remove the large ImageNet classifier and replace with a smaller one.
    # After 5 poolings on 32x32 input -> feature map is 1x1, channels=512 -> flatten size 512
    model.classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model
