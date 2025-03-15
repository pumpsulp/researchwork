import os

import numpy as np
import torch
import wfdb
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class MITBIHDataset(Dataset):
    def __init__(self, record_numbers, path, labels, window_size=256, stride=128):
        self.segments = []
        self.labels = []
        self.window_size = window_size
        self.stride = stride
        
        for record_number in record_numbers:
            record_path = os.path.join(path, record_number)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            signal = np.array(record.p_signal[:, 0])  # Используем только первый канал
            
            ann_symbols = annotation.symbol
            ann_sample = annotation.sample
            
            # Создание сегментов и меток
            for i in range(0, len(signal) - window_size, stride):
                segment = signal[i:i + window_size]
                labels_in_window = [symbol for symbol, sample in zip(ann_symbols, ann_sample) if
                                    i <= sample < i + window_size]
                
                # Используем первую аннотацию в окне как метку
                label = labels_in_window[0] if labels_in_window and labels_in_window[0] in labels else 'N'
                
                # Если метка в интересующем нас списке, добавляем сегмент и метку
                if label in labels:
                    self.segments.append(segment)
                    self.labels.append(label)
        
        # Преобразование меток в числовой формат
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]
        return torch.tensor(segment, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
