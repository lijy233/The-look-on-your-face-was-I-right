import torch
import torch.nn as nn
from torchvision import models

# 定义多模态模型
class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()
        self.image_model = models.resnet18(weights=models.resnet.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(self.image_model.children())[:-1])  # 去掉最后一层全连接层

    def forward(self, image):
        image_features = self.feature_extractor(image)
        return image_features.squeeze()
