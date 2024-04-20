from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class NNModel(nn.Module):
    """Представляет собой абстрактный класс нейронной сети\n
    Наследует nn.Module из PyTorch"""
    
    def __init__(self):
        super(NNModel, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass
    
    def save_model(self, path: Path | str):
        torch.save(self, path)
    
    def save_model_state_dict(self, path: Path | str):
        torch.save(self.state_dict(), path)


class ResNet18(NNModel):
    """Представляет собой нейронную сеть ResNet18"""
    
    def __init__(self,
                 num_classes: int,
                 pretrained: bool = False):
        super(ResNet18, self).__init__()
        self.cnn = models.resnet18(pretrained=pretrained)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return F.log_softmax(self.cnn(x), dim=1)


class SimpleCNN(NNModel):
    """Простая CNN"""
    ...
