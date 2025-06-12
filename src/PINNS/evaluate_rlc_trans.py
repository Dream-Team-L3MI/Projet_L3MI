import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import torch.nn as nn


# Load data and scalers
df = pd.read_csv("data.csv")

scaler_time = joblib.load("scaler_time.pkl")
scaler_current = joblib.load("scaler_current.pkl")
scaler_R = joblib.load("scaler_R.pkl")
scaler_L = joblib.load("scaler_L.pkl")
scaler_C = joblib.load("scaler_C.pkl")


# Load model definition and weights
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
model.load_state_dict(torch.load("trained_pinn.pt"))
model.eval()

# Prepare normalized inputs for prediction
time_norm = scaler_time.transform(df["Time"].values.reshape(-1,1))
R_norm = scaler_R.transform(df["R"].values.reshape(-1,1))
L_norm = scaler_L.transform(df["L"].values.reshape(-1,1))
C_norm = scaler_C.transform(df["C"].values.reshape(-1,1))

X = np.hstack([time_norm, R_norm, L_norm, C_norm])
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_norm = model(X_tensor).cpu().numpy()

# Invert normalization to get predicted current in physical units
y_pred = scaler_current.inverse_transform(y_pred_norm)
y_true = df["I"].values.reshape(-1,1)

# Add predictions to dataframe for convenience
df["I_pred"] = y_pred

# Create a column that identifies unique circuits
# For example, based on unique (R,L,C) triplets rounded to some decimal places:
df["circuit_id"] = df[["R", "L", "C"]].round(decimals=5).astype(str).agg("-".join, axis=1)

# Prepare normalized time grid [0,1] over 5 tau
t_grid = np.linspace(0, 1, 100)

true_curves = []
pred_curves = []

for cid, group in df.groupby("circuit_id"):
    t = group["Time"].values
    R_val = group["R"].values[0]
    L_val = group["L"].values[0]
    
    tau = L_val / R_val
    t_norm = t / (5 * tau)  # Normalize time over 5 tau
    
    # Filter points where t_norm <= 1 (within 5 tau)
    mask = t_norm <= 1
    if np.sum(mask) < 10:
        # Skip if too few points in range
        continue
    
    t_norm = t_norm[mask]
    i_true = group["I"].values[mask]
    i_pred = group["I_pred"].values[mask]
    
    # Interpolate true and predicted currents onto common grid
    f_true = interp1d(t_norm, i_true, kind='linear', fill_value="extrapolate")
    f_pred = interp1d(t_norm, i_pred, kind='linear', fill_value="extrapolate")
    
    true_curves.append(f_true(t_grid))
    pred_curves.append(f_pred(t_grid))

true_curves = np.array(true_curves)
pred_curves = np.array(pred_curves)

# Compute mean and std
true_mean = np.mean(true_curves, axis=0)
true_std = np.std(true_curves, axis=0)
pred_mean = np.mean(pred_curves, axis=0)
pred_std = np.std(pred_curves, axis=0)

# Plot mean with std shaded area
plt.figure(figsize=(10,6))
plt.plot(t_grid * 5, true_mean, label="True Current", color="blue")
plt.fill_between(t_grid * 5, true_mean - true_std, true_mean + true_std, color="blue", alpha=0.3)

plt.plot(t_grid * 5, pred_mean, label="Predicted Current", color="red")
plt.fill_between(t_grid * 5, pred_mean - pred_std, pred_mean + pred_std, color="red", alpha=0.3)

plt.xlabel("Normalized Time (tau units)")
plt.ylabel("Current (I)")
plt.title("Average Current Response over Normalized Time (0 to 5 tau)")
plt.legend()
plt.grid(True)
plt.show()
