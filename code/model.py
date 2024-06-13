import torch
import torch.nn as nn
from torchvision import models

class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.image_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.image_model.children())[:-1])

    def forward(self, image):
        image_features = self.feature_extractor(image)
        return image_features.squeeze()
