# utils/utils.py
import torch

def calculate_metrics(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    metrics = {"val_loss": avg_loss}
    return metrics