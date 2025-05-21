import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from tqdm import tqdm

# Define the model
class RLCNet(nn.Module):
    def __init__(self):
        super(RLCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# Entry point for Windows
if __name__ == '__main__':
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = pd.read_csv("rlc_transient.csv")

    # Features and target
    X = df[["R", "L", "C", "Time_s", "Vin_V"]].values
    y = df[["Vout_V"]].values

    # Normalize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Save scalers
    joblib.dump(scaler_x, "vanilla_rlc_transient_scaler_X.pkl")
    joblib.dump(scaler_y, "vanilla_rlc_transient_scaler_y.pkl")

    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=0)  # num_workers=0 for Windows

    # Initialize model
    model = RLCNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    epochs = 150
    losses = []

    print("Starting training...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_X, batch_y in loop:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    # Save model
    torch.save(model.state_dict(), "vanilla_rlc_transient.pt")

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Test on a small sample
    print("Evaluating model...")

    X_test = df[["R", "L", "C", "Time_s", "Vin_V"]].values[10000:10100]
    y_test = df[["Vout_V"]].values[10000:10100]

    X_test_scaled = scaler_x.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Plot prediction
    plt.figure(figsize=(10, 5))
    plt.plot(X_test[:, 3], y_test, 'o', label='True Vout')
    plt.plot(X_test[:, 3], y_pred, '-', label='Predicted Vout', alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Vout (V)")
    plt.title("RLC Circuit - Predicted vs Real Vout")
    plt.legend()
    plt.grid(True)
    plt.show()



