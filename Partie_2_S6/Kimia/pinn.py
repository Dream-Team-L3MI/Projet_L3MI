import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------- Phase 1: Data Loading and Preprocessing ----------
df = pd.read_csv("data.csv")

time = df["Time"].values.reshape(-1, 1)
current = df["I"].values.reshape(-1, 1)
R = df["R"].values.reshape(-1, 1)
L = df["L"].values.reshape(-1, 1)
C = df["C"].values.reshape(-1, 1)

# Normalize inputs and outputs (time and current only)
scalers = {
    "time": MinMaxScaler(),
    "current": MinMaxScaler()
}

time_norm = scalers["time"].fit_transform(time)
current_norm = scalers["current"].fit_transform(current)

# Keep R, L, C in physical units for physics residual
R_phys = torch.tensor(R, dtype=torch.float32)
L_phys = torch.tensor(L, dtype=torch.float32)
C_phys = torch.tensor(C, dtype=torch.float32)

# Inputs to model: normalized time plus physical R,L,C values
inputs_norm = np.hstack([time_norm, R, L, C])
targets_norm = current_norm

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    inputs_norm, targets_norm, test_size=0.2, random_state=42
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

# For physics residual collocation points, also keep physical R,L,C from training data
R_train_phys = torch.tensor(X_train_np[:, 1], dtype=torch.float32).unsqueeze(1)
L_train_phys = torch.tensor(X_train_np[:, 2], dtype=torch.float32).unsqueeze(1)
C_train_phys = torch.tensor(X_train_np[:, 3], dtype=torch.float32).unsqueeze(1)

# ---------- Phase 2: PINN Definition ----------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Phase 3: Physics Residual ----------
def physics_residual(model, inputs, R_phys, L_phys, C_phys):
    # inputs shape: [N, 4] with normalized time and physical R,L,C

    # Use normalized time for gradient computations
    t = inputs[:, [0]].clone().detach().requires_grad_(True).to(device)
    
    # Use physical R, L, C tensors, detach since not trainable
    R = R_phys.to(device)
    L = L_phys.to(device)
    C = C_phys.to(device)

    # Form full input to model (normalized time + physical R,L,C)
    model_input = torch.cat([t, R, L, C], dim=1)

    i_hat = model(model_input)

    di_dt = torch.autograd.grad(
        outputs=i_hat, inputs=t,
        grad_outputs=torch.ones_like(i_hat),
        create_graph=True, retain_graph=True
    )[0]

    d2i_dt2 = torch.autograd.grad(
        outputs=di_dt, inputs=t,
        grad_outputs=torch.ones_like(di_dt),
        create_graph=True, retain_graph=True
    )[0]

    # Physics residual
    residual = L * d2i_dt2 + R * di_dt + (1.0 / (C + 1e-6)) * i_hat

    return residual

# ---------- Phase 4: Loss Function ----------
def pinn_loss(model, X_data, y_data, X_colloc, R_colloc, L_colloc, C_colloc,
              lambda_data=1.0, lambda_phys=1.0):
    y_pred = model(X_data)
    data_loss = F.mse_loss(y_pred, y_data)

    physics_res = physics_residual(model, X_colloc, R_colloc, L_colloc, C_colloc)
    physics_loss = F.mse_loss(physics_res, torch.zeros_like(physics_res))

    total_loss = lambda_data * data_loss + lambda_phys * physics_loss
    return total_loss, data_loss, physics_loss

# ---------- Phase 5: Collocation Points ----------
N_colloc = 2000
t_min, t_max = time_norm.min(), time_norm.max()
t_colloc = np.random.uniform(t_min, t_max, (N_colloc, 1))

# Sample physical R, L, C values randomly from training set for collocation points
random_indices = np.random.choice(X_train_np.shape[0], size=N_colloc, replace=True)
R_colloc_np = X_train_np[random_indices, 1].reshape(-1, 1)
L_colloc_np = X_train_np[random_indices, 2].reshape(-1, 1)
C_colloc_np = X_train_np[random_indices, 3].reshape(-1, 1)

colloc_inputs_np = np.hstack([t_colloc, R_colloc_np, L_colloc_np, C_colloc_np])
colloc_inputs_tensor = torch.tensor(colloc_inputs_np, dtype=torch.float32).to(device)

R_colloc = torch.tensor(R_colloc_np, dtype=torch.float32).to(device)
L_colloc = torch.tensor(L_colloc_np, dtype=torch.float32).to(device)
C_colloc = torch.tensor(C_colloc_np, dtype=torch.float32).to(device)

# ---------- Phase 6: Model & Optimizer ----------
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------- Phase 7: Training ----------
epochs = 1000
lambda_data = 1.0
lambda_phys = 1.0  # start smaller

for epoch in tqdm(range(epochs), desc="Training Epochs"):
    model.train()
    optimizer.zero_grad()

    total_loss, data_loss, physics_loss = pinn_loss(
        model, X_train, y_train, colloc_inputs_tensor,
        R_colloc, L_colloc, C_colloc,
        lambda_data=lambda_data, lambda_phys=lambda_phys
    )

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} | "
              f"Data Loss: {data_loss.item():.6f} | Physics Loss: {physics_loss.item():.6f}")
        sample_residual = physics_residual(model, colloc_inputs_tensor[0:1],
                                          R_colloc[0:1], L_colloc[0:1], C_colloc[0:1])
        print(f"Sample physics residual: {sample_residual.item():.6e}")

# ---------- Phase 8: Evaluation ----------
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_mse = F.mse_loss(y_pred_test, y_test).item()

    # Inverse transform for R² score
    y_test_inv = scalers["current"].inverse_transform(y_test.cpu().numpy())
    y_pred_inv = scalers["current"].inverse_transform(y_pred_test.cpu().numpy())
    ss_res = np.sum((y_test_inv - y_pred_inv) ** 2)
    ss_tot = np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
    r2_score = 1 - ss_res / ss_tot

print("\n--- Evaluation Results ---")
print(f"Test MSE: {test_mse:.6f}")
print(f"R² Score: {r2_score:.4f}")
