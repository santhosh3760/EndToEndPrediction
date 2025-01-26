# train.py
import torch
import mlflow
from models.mlp import MLP
from data.dataset import load_data
from utils.utils import calculate_metrics
import yaml

def train_model(config):
    # Load data
    train_loader, test_loader = load_data(
        config["data"]["dataset_path"],
        config["data"]["train_test_split"],
        config["data"]["batch_size"]
    )

    # Initialize model
    model = MLP(
        input_size=config["model"]["input_size"],
        hidden_sizes=config["model"]["hidden_sizes"],
        output_size=config["model"]["output_size"],
        activation=config["model"]["activation"]
    )

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    best_val_loss = float("inf")
    train_losses = []  # Store training losses for each epoch
    val_losses = []    # Store validation losses for each epoch

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Log training loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        mlflow.log_metric("train_loss", avg_epoch_loss, step=epoch)

        # Evaluate on validation set
        model.eval()
        val_metrics = calculate_metrics(model, test_loader, criterion)
        val_losses.append(val_metrics["val_loss"])
        mlflow.log_metric("val_loss", val_metrics["val_loss"], step=epoch)

        # Track best validation loss
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]

        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, Train Loss: {avg_epoch_loss}, Val Loss: {val_metrics['val_loss']}")

    # Save model
    mlflow.pytorch.log_model(model, "model")

    # Log training and validation loss curves
    log_loss_curves(train_losses, val_losses)

    return best_val_loss

def log_loss_curves(train_losses, val_losses):
    """Log train and validation loss curves as images to MLflow."""
    import matplotlib.pyplot as plt

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plot_path = "loss_curves.png"
    plt.savefig(plot_path)
    plt.close()

    # Log the plot to MLflow
    mlflow.log_artifact(plot_path)