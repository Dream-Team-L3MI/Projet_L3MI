import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Chargement du modèle et des scalers
from torch import nn
import torch.nn.functional as F

class RegressionModel(nn.Module):
    def __init__(self, in_features, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, out_features)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        x = self.fc6(x)
        return x

# Chemins
model_path = "pinn_RC_AC_transient_500K_TEST.pt"
scaler_x_path = "pinn_RC_AC_transient_scaler_X_500K_TEST.pkl"
scaler_y_path = "pinn_RC_AC_transient_scaler_y_500K_TEST.pkl"

# Chargement
model = RegressionModel(in_features=4)
model.load_state_dict(torch.load(model_path))
model.eval()

scaler_x = joblib.load(scaler_x_path)
scaler_y = joblib.load(scaler_y_path)

# Génération des données d'extrapolation
R = 1000
C = 1e-6
Vin = 5
frequencies = np.linspace(1e6, 5e6, 300)  # extrapolation vers plus haute fréquence

X_extra = np.column_stack([
    np.full_like(frequencies, R),
    np.full_like(frequencies, C),
    np.full_like(frequencies, Vin),
    frequencies
])

# Normalisation
X_extra_scaled = scaler_x.transform(X_extra)
X_extra_tensor = torch.tensor(X_extra_scaled, dtype=torch.float32)

# Prédiction
with torch.no_grad():
    predictions = model(X_extra_tensor).numpy()

# Dénormalisation des sorties
preds_denorm = scaler_y.inverse_transform(predictions)

gain_basse, gain_haute, phase_R, phase_C = preds_denorm.T

# === Visualisation ===
plt.figure(figsize=(10, 6))
plt.plot(frequencies, gain_basse, label="Gain Basse", color='blue')
plt.plot(frequencies, gain_haute, label="Gain Haute", color='green')
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Gain")
plt.title("Extrapolation des gains")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(frequencies, phase_R, label="Phase R", color='orange')
plt.plot(frequencies, phase_C, label="Phase C", color='red')
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Phase (rad)")
plt.title("Extrapolation des phases")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
