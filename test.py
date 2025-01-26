# test.py
import torch
import mlflow
from models.mlp import MLP
from data.dataset import load_data
from utils.utils import calculate_metrics
import yaml

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# Load data
_, test_loader = load_data(
    config["data"]["dataset_path"],
    config["data"]["train_test_split"],
    config["data"]["batch_size"]
)

# Load the final model from MLflow
final_run_id = mlflow.search_runs(filter_string="tags.mlflow.runName = 'final_model'").iloc[0].run_id
model_uri = f"runs:/{final_run_id}/model"
model = mlflow.pytorch.load_model(model_uri)

# Evaluate the model
with mlflow.start_run():
    metrics = calculate_metrics(model, test_loader, torch.nn.MSELoss())
    mlflow.log_metrics(metrics)
    print(f"Test Metrics: {metrics}")