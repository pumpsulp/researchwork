import torch
from torch import nn

class ClassicCNN(nn.Module):
    def __init__(self, num_classes):
        super(ClassicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


class CNNWithEncoder(nn.Module):
    def __init__(self, num_classes):
        super(CNNWithEncoder, self).__init__()
        
        # Верхний путь (features_up) - сверточные слои
        self.features_up = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),  # (batch_size, 16, 256)
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (batch_size, 16, 128)
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),  # (batch_size, 32, 128)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (batch_size, 32, 64)
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),  # (batch_size, 64, 64)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (batch_size, 64, 32)
        )
        
        # Полносвязный линейный слой после features_up
        self.conv_out = nn.Linear(64 * 32, 64)  # Преобразуем из (64 * 32) в 64
        
        # Нижний путь (features_down) - полносвязные слои
        self.features_down = nn.Sequential(
            nn.Linear(256, 128),  # Преобразуем из (256) в 128
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 64),  # Преобразуем из (128) в 64
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 64)
        )
        
        # Финальный классификатор
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),  # Преобразуем из (64) в 128
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),  # Преобразуем из (128) в 64
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)  # Преобразуем из (64) в количество классов
        )
    
    def forward(self, x):
        # Верхний путь (features_up)
        out_up = self.features_up(x)
        out_up = out_up.view(out_up.size(0), -1)  # Flatten to (batch_size, 64 * 32)
        out_up = self.conv_out(out_up)  # (batch_size, 64)
        
        # Нижний путь (features_down)
        out_down = x.view(x.size(0), -1)  # Flatten to (batch_size, 256)
        out_down = self.features_down(out_down)  # (batch_size, 64)
        
        # Суммирование двух путей
        out = out_up + out_down
        
        # Финальная классификация
        out = self.classifier(out)  # (batch_size, num_classes)
        
        return out