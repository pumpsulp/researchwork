from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def load_record_numbers(path: Path):
    """Загрузка номеров записей в наборе данных"""
    records = []
    for file in path.glob('*.dat'):
        records.append(file.stem)
    return records


def show_confusion_matrix(y_true, y_pred, labels):
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Нормализация матрицы ошибок (по строкам)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Отображение нормализованной матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Reds', xticklabels=labels,
                yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Percentage)')
    plt.show()


def train_metrics(train_losses, val_losses, val_accuracies):
    # Визуализация результатов
    plt.figure(figsize=(12, 4))
    
    # Кривые обучения
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Метрика точности на валидации
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()