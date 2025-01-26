# data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RandomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    


def load_data(dataset_path, split_ratio, batch_size):
    # Load dataset
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["readmitted"]).values
    y = df["readmitted"].values
    

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_ratio, random_state=42)
    
        # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create datasets and dataloaders
    train_dataset = RandomDataset(X_train, y_train)
    test_dataset = RandomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader





