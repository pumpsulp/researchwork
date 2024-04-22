from dataclasses import dataclass, field

from sklearn.metrics import accuracy_score

import torch

from objects.Dataset import Dataset


@dataclass
class TeacherTrain:
    model: torch.nn.Module
    criterion: torch.nn.modules.loss._Loss
    optimizer: torch.optim.Optimizer
    cuda: bool = field(default=False)
    
    def __post_init__(self):
        if self.cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print("Cuda is not available. Set device to CPU.")
                self.device = torch.device("cpu")
    
    def train(self, data: Dataset, num_epochs: int):
        # TODO: нужен рефакторинг
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        train_loader = data.train.dataloader
        
        try:
            val_loader = data.valid.dataloader
        except KeyError:
            val_loader = None
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            if val_loader is not None:
                # Оценка модели на наборе валидации
                self.model.eval()
                all_pred_val = []
                all_labels_val = []
                
                with torch.no_grad():
                    for images_val, labels_val, _ in val_loader:
                        images_val, labels_val = images_val.to(self.device), labels_val.to(self.device)
                        
                        outputs_val = self.model(images_val)
                        _, preds_val = torch.max(outputs_val, 1)
                        
                        all_pred_val.extend(preds_val.cpu().numpy())
                        all_labels_val.extend(labels_val.cpu().numpy())
                
                accuracy_val = accuracy_score(all_labels_val, all_pred_val)
                val_loss = self.criterion(outputs_val, labels_val).item()
                # Сохранение val метрик
                val_losses.append(val_loss)
                val_accuracies.append(accuracy_val)
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, "
                    "Validation Accuracy: {accuracy_val * 100:.2f}%")
            
            # Сохранение train метрик
            train_losses.append(running_loss / len(train_loader))
        
        return train_losses, val_losses
    
    @property
    def get_model(self):
        return self.model
