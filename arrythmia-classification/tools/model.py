from datetime import datetime

import torch
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, trainloader, valloader, criterion, optimizer, num_epochs, device='cuda'):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        
        running_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        model.eval()
        all_preds_val = []
        all_labels_val = []
        
        with torch.no_grad():
            for inputs, labels in valloader:
                images_val, labels_val = inputs.to(device), labels.to(device)
                
                outputs_val = model(images_val)
                _, preds_val = torch.max(outputs_val, 1)
                
                all_preds_val.extend(preds_val.cpu().numpy())
                all_labels_val.extend(labels_val.cpu().numpy())
        
        accuracy_val = accuracy_score(all_labels_val, all_preds_val)
        val_loss = criterion(outputs_val, labels_val).item()
        
        # Сохранение метрик
        train_losses.append(running_loss / len(trainloader))
        val_losses.append(val_loss)
        val_accuracies.append(accuracy_val)
        
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader)}, Validation Accuracy: {accuracy_val * 100:.2f}%")
    
    return train_losses, val_losses, val_accuracies


def test(model, test_loader, device='cuda'):
    # Оценка модели на тестовом наборе
    model.eval()
    all_preds_test = []
    all_labels_test = []
    
    with torch.no_grad():
        for data_test, labels_test in test_loader:
            data_test, labels_test = data_test.to(device), labels_test.to(device)
            
            outputs_test = model(data_test)
            _, preds_test = torch.max(outputs_test, 1)
            
            all_preds_test.extend(preds_test.cpu().numpy())
            all_labels_test.extend(labels_test.cpu().numpy())
        
        accuracy_test = accuracy_score(all_labels_test, all_preds_test)
        print(f"Final Test Accuracy: {accuracy_test * 100:.2f}%")
    
    return round(accuracy_test, 2), all_labels_test, all_preds_test


def save_weights(model, path, name: str):
    torch.save(model.state_dict(),
               f'{path}/{name}-{datetime.date.today().strftime("%Y-%m-%d")}.pth')
