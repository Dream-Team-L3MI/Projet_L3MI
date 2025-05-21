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

# مدل شبکه
class RLCNet(nn.Module):
    def __init__(self):
        super(RLCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# physics loss براساس مشتق‌گیری خودکار
def compute_physics_loss(X, y_pred):
    X.requires_grad_(True)
    Vout = y_pred

    dVout_dt = torch.autograd.grad(
        Vout, X, grad_outputs=torch.ones_like(Vout),
        create_graph=True, retain_graph=True
    )[0][:, 3]  # مشتق نسبت به زمان

    d2Vout_dt2 = torch.autograd.grad(
        dVout_dt, X, grad_outputs=torch.ones_like(dVout_dt),
        create_graph=True, retain_graph=True
    )[0][:, 3]

    R = X[:, 0]
    L = X[:, 1]
    C = X[:, 2]
    Vin = X[:, 4]
    Vout = Vout.squeeze()

    residual = (L * C * d2Vout_dt2) + (R * C * dVout_dt) + Vout - Vin
    return torch.mean(residual ** 2)

# اجرای اصلی
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("rlc_transient3.csv")

    X_all = df[["R", "L", "C", "Time", "Vin"]].values
    y_all = df[["Vout_V"]].values

    # نرمال‌سازی
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X_all)
    y_scaled = scaler_y.fit_transform(y_all)

    joblib.dump(scaler_x, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

    # تبدیل به tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # تقسیم داده‌ها: 1000 نمونه برچسب‌دار، بقیه فقط ورودی
    X_labeled = X_tensor[:1000]
    y_labeled = y_tensor[:1000]
    X_unlabeled = X_tensor

    labeled_loader = DataLoader(TensorDataset(X_labeled, y_labeled), batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(TensorDataset(X_unlabeled), batch_size=256, shuffle=True)

    # تعریف مدل
    model = RLCNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    lambda_phys = 50.0
    epochs = 100
    losses = []

    print("Starting semi-supervised training with physics-informed loss...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # گام ۱: داده برچسب‌دار (loss_data)
        for X_batch, y_batch in labeled_loader:
            X_batch = X_batch.to(device).detach().requires_grad_(True)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss_data = criterion(pred, y_batch)
            loss_data.backward()
            optimizer.step()
            total_loss += loss_data.item()

        # گام ۲: کل داده‌ها برای physics loss
        for (X_ph,) in unlabeled_loader:
            X_ph = X_ph.to(device).detach().requires_grad_(True)

            optimizer.zero_grad()
            pred_ph = model(X_ph)
            loss_phys = compute_physics_loss(X_ph, pred_ph)
            total_loss_batch = lambda_phys * loss_phys
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()

        avg_loss = total_loss / (len(labeled_loader) + len(unlabeled_loader))
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {avg_loss:.6f}")

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    torch.save(model.state_dict(), "semi_phys_model.pt")

    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title("Total Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_scaled[:1000], dtype=torch.float32).to(device)
        y_test_true = y_all[:1000]
        y_pred_scaled = model(X_test).cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

    plt.figure(figsize=(10, 5))
    plt.plot(X_all[:1000, 3], y_test_true, 'o', label="True Vout")
    plt.plot(X_all[:1000, 3], y_pred, '-', label="Predicted Vout")
    plt.title("Prediction vs True (1000 labeled)")
    plt.xlabel("Time (s)")
    plt.ylabel("Vout (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    




