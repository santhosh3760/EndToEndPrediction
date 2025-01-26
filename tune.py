# tune.py
import optuna
import mlflow
import torch
from train import train_model
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def objective(trial):
    # Define hyperparameters to tune
    lr = trial.suggest_float("lr", *config["tuning"]["lr_range"], log=True)
    hidden_sizes = trial.suggest_categorical("hidden_sizes", config["tuning"]["hidden_sizes_options"])
    batch_size = trial.suggest_categorical("batch_size", config["tuning"]["batch_size_options"])

    # Ensure hidden_sizes is a list of integers
    if isinstance(hidden_sizes, str):
        # Convert string representation of list to actual list
        hidden_sizes = eval(hidden_sizes)
    elif isinstance(hidden_sizes, tuple):
        # Convert tuple to list
        hidden_sizes = list(hidden_sizes)

    # Update config with suggested hyperparameters
    config["training"]["learning_rate"] = lr
    config["model"]["hidden_sizes"] = hidden_sizes
    config["data"]["batch_size"] = batch_size

    # Train the model and get validation metrics
    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "lr": lr,
            "hidden_sizes": hidden_sizes,
            "batch_size": batch_size
        })
        val_loss = train_model(config)
        mlflow.log_metric("val_loss", val_loss)

    return val_loss

# Initialize MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# Run Optuna study
with mlflow.start_run():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config["tuning"]["n_trials"])

    # Log best trial
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_val_loss", study.best_value)

    print(f"Best trial: {study.best_trial.params}")
    print(f"Best validation loss: {study.best_value}")

    # Train the final model with the best hyperparameters
    config["training"]["learning_rate"] = study.best_params["lr"]
    config["model"]["hidden_sizes"] = list(study.best_params["hidden_sizes"])  # Convert tuple back to list
    config["data"]["batch_size"] = study.best_params["batch_size"]

    with mlflow.start_run(nested=True, run_name="final_model"):
        final_val_loss = train_model(config)
        mlflow.log_metric("final_val_loss", final_val_loss)
        print(f"Final model validation loss: {final_val_loss}")