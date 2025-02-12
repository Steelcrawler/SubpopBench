import torch
import torchvision.models as models
import os

# Create directory for models
os.makedirs("pretrained_models", exist_ok=True)

# Download ResNet18
resnet18 = models.resnet18(pretrained=True)
torch.save(resnet18.state_dict(), "/cluster/home/t130016uhn/PNXbench/SubpopBench/subpopbench/pretrained_models/resnet18_imagenet.pth")

# Download ResNet50
resnet50 = models.resnet50(pretrained=True)
torch.save(resnet50.state_dict(), "/cluster/home/t130016uhn/PNXbench/SubpopBench/subpopbenchpretrained_models/resnet50_imagenet.pth")