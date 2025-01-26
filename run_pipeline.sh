#!/bin/bash

# Activate virtual environment (if any)
# source venv/bin/activate

# # Install dependencies
# pip install -r requirements.txt

# Run hyperparameter tuning
python tune.py

# Train the final model with best hyperparameters
python train.py

# Test the model
python test.py