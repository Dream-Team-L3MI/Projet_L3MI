import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------- Phase 1: Data Loading and Preprocessing --------
df = pd.read_csv("data.csv")

time = df["Time"].values.reshape(-1, 1)
current = df["I"].values.reshape(-1, 1)
R = df["R"].values.reshape(-1, 1)
L = df["L"].values.reshape(-1, 1)
C = df["C"].values.reshape(-1, 1)

# Normalizers
scaler_time = MinMaxScaler()
scaler_current = MinMaxScaler()
scaler_R = MinMaxScaler()
scaler_L = MinMaxScaler()
scaler_C = MinMaxScaler()

# Normalize
time_norm = scaler_time.fit_transform(time)
current_norm = scaler_current.fit_transform(current)
R_norm = scaler_R.fit_transform(R)
L_norm = scaler_L.fit_transform(L)
C_norm = scaler_C.fit_transform(C)

# Time scale for physics loss
t_min, t_max = time_norm.min(), time_norm.max()
dt_dt_norm = (scaler_time.data_max_ - scaler_time.data_min_)[0]

# Combine input features
inputs_norm = np.hstack([time_norm, R_norm, L_norm, C_norm])
targets_norm = current_norm

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    inputs_norm, targets_norm, test_size=0.2, random_state=42
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

# -------- Phase 2: PINN Model with Swish and deeper network --------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, 128),
            Swish(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------- Phase 3: Physics Residual --------
def physics_residual(model, inputs):
    t = inputs[:, 0:1].clone().detach().requires_grad_(True).to(device)
    RL = inputs[:, 1:4].to(device)
    model_input = torch.cat([t, RL], dim=1)

    i_norm = model(model_input)

    # Derivatives w.r.t. normalized time
    di_dt = torch.autograd.grad(i_norm, t, torch.ones_like(i_norm), create_graph=True)[0]
    d2i_dt2 = torch.autograd.grad(di_dt, t, torch.ones_like(di_dt), create_graph=True)[0]

    # Time scaling
    dt_scale = (scaler_time.data_max_[0] - scaler_time.data_min_[0])
    di_dt_phys = di_dt / dt_scale
    d2i_dt2_phys = d2i_dt2 / (dt_scale ** 2)

    # Differentiable de-normalization (NO detach, NO numpy)
    R = RL[:, 0:1] * (scaler_R.data_max_[0] - scaler_R.data_min_[0]) + scaler_R.data_min_[0]
    L = RL[:, 1:2] * (scaler_L.data_max_[0] - scaler_L.data_min_[0]) + scaler_L.data_min_[0]
    C = RL[:, 2:3] * (scaler_C.data_max_[0] - scaler_C.data_min_[0]) + scaler_C.data_min_[0]

    i = i_norm * (scaler_current.data_max_[0] - scaler_current.data_min_[0]) + scaler_current.data_min_[0]

    # Physics residual
    eps = 1e-8
    residual = L * d2i_dt2_phys + R * di_dt_phys + (1.0 / (C + eps)) * i

    return residual


# -------- Phase 4: Loss Function --------
def pinn_loss(model, X_data, y_data, X_colloc, lambda_data=1.0, lambda_phys=0.1):
    y_pred = model(X_data)
    data_loss = F.mse_loss(y_pred, y_data)
    physics_res = physics_residual(model, X_colloc)
    physics_loss = torch.mean(torch.log(1 + physics_res ** 2))
    total_loss = lambda_data * data_loss + lambda_phys * physics_loss
    return total_loss, data_loss, physics_loss

# -------- Phase 5: Collocation Sampling (Uniform) --------
N_colloc = 5000
t_colloc = np.random.uniform(0, 1, (N_colloc, 1))
R_colloc = np.random.uniform(0, 1, (N_colloc, 1))
L_colloc = np.random.uniform(0, 1, (N_colloc, 1))
C_colloc = np.random.uniform(0, 1, (N_colloc, 1))
colloc_inputs_np = np.hstack([t_colloc, R_colloc, L_colloc, C_colloc])
colloc_inputs_tensor = torch.tensor(colloc_inputs_np, dtype=torch.float32).to(device)

# -------- Phase 6: Model & Optimizer --------
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# -------- Phase 7: Training --------
epochs = 1000
lambda_data = 1.0
lambda_phys = 0.1

for epoch in tqdm(range(epochs), desc="Training Epochs"):
    model.train()
    optimizer.zero_grad()

    total_loss, data_loss, physics_loss = pinn_loss(
        model, X_train, y_train, colloc_inputs_tensor,
        lambda_data=lambda_data, lambda_phys=lambda_phys
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} | "
              f"Data Loss: {data_loss.item():.6f} | Physics Loss: {physics_loss.item():.6f}")
        sample_residual = physics_residual(model, colloc_inputs_tensor[0:1])
        print(f"Sample physics residual: {sample_residual.item():.6e}")

# -------- Phase 8: Evaluation --------
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_mse = F.mse_loss(y_pred_test, y_test).item()

    y_test_inv = scaler_current.inverse_transform(y_test.cpu().numpy())
    y_pred_inv = scaler_current.inverse_transform(y_pred_test.cpu().numpy())

    ss_res = np.sum((y_test_inv - y_pred_inv) ** 2)
    ss_tot = np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
    r2_score = 1 - ss_res / ss_tot

print("\n--- Evaluation Results ---")
print(f"Test MSE: {test_mse:.6f}")
print(f"R² Score: {r2_score:.4f}")


torch.save(model.state_dict(), "trained_pinn.pt")

# Optionally also save the scalers (to use during testing)
import joblib
joblib.dump(scaler_current, "scaler_current.pkl")
joblib.dump(scaler_time, "scaler_time.pkl")
joblib.dump(scaler_R, "scaler_R.pkl")
joblib.dump(scaler_L, "scaler_L.pkl")
joblib.dump(scaler_C, "scaler_C.pkl")
