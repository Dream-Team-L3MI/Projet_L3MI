import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from pinn import Net  # Assure-toi que ce fichier existe

# Détection du device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

sns.set_theme()

# Import des données
df = pd.read_csv("rlc_ac_results.csv")

# Échantillonnage
df_sample = df.sample(n=10000, random_state=42)
X = df_sample[["R", "L", "C", "Frequency", "Vin"]].values
y = df_sample[["V_C"]].values

# Création et entraînement du réseau classique
net = Net(5, 1, loss2=None, epochs=40, batch_size=64, lr=1e-3).to(device)
losses = net.fit(X, y)
plt.plot(losses)
plt.title("Loss - Réseau classique")
#plt.show()

# Sauvegarde
torch.save(net.state_dict(), "vanilla_rlc_ac.pt")
joblib.dump(net.scaler_X, "vanilla_rlc_ac_X.pkl")
joblib.dump(net.scaler_y, "vanilla_rlc_ac_scaler_y.pkl")

# Prédiction sur un petit intervalle
X_preds = df[["R", "L", "C", "Frequency", "Vin"]].values[100:130]
y_preds = df[["V_C"]].values[100:130]
preds = net.predict(X_preds)

# Équation analytique
omega = 2 * np.pi * X_preds[:, 3]
R, L, C, Vin = X_preds[:, 0], X_preds[:, 1], X_preds[:, 2], X_preds[:, 4]
denom = np.sqrt((1 - omega**2 * L * C)**2 + (omega * R * C)**2)
equation = Vin / denom

# Tracé
plt.plot(X_preds[:, 3], equation, label='Equation', alpha=0.6)
plt.plot(X_preds[:, 3], y_preds, 'o', label='Données')
plt.plot(X_preds[:, 3], preds, label='Réseau', alpha=0.6)
plt.legend()
plt.ylabel("Tension (V)")
plt.xlabel("Fréquence (Hz)")
plt.title("Réponse AC du circuit RLC (V_C)")
plt.grid(True)
#plt.show()

# Fonction de perte PINN
def rlc_ac_loss(model: torch.nn.Module, X):
    X_np = X.detach().cpu().numpy()

    # 1. Dé-normaliser les features pour le calcul analytique
    X_np = model.scaler_X.inverse_transform(X_np)
    R, L, C, f, Vin = X_np[:, 0], X_np[:, 1], X_np[:, 2], X_np[:, 3], X_np[:, 4]
    omega = 2 * np.pi * f
    denom = np.sqrt((1 - omega**2 * L * C)**2 + (omega * R * C)**2)
    vc_physique = Vin / denom

    # 2. Re-normaliser la cible physique avec scaler_y
    vc_physique = model.scaler_y.transform(vc_physique.reshape(-1, 1)).astype(np.float32)

    # 3. Conversion en tenseur torch
    vc_physique = torch.tensor(vc_physique, dtype=torch.float32).to(X.device).view(-1, 1)

    # 4. Prédiction du modèle (sur X déjà normalisé)
    vc_predict = model(X)

    # 5. Comparaison dans le même espace
    loss_physique = F.mse_loss(vc_predict, vc_physique)
    return loss_physique



# Création + entraînement du modèle PINN
net_pinn = Net(5, 1, loss2=rlc_ac_loss, loss2_weight=0.01, epochs=40, batch_size=64, lr=1e-3).to(device)
losses_pinn = net_pinn.fit(X, y)
plt.plot(losses_pinn)
plt.title("Loss - PINN")
#plt.show()

# Comparaison sur un sous-ensemble
X_preds = df[["R", "L", "C", "Frequency", "Vin"]].values[100:130, :]
y_preds = df[["V_C"]].values[100:130, :]

preds_net = net.predict(X_preds)
preds_pinn = net_pinn.predict(X_preds)

# Équation théorique
omega = 2 * np.pi * X_preds[:, 3]
R, L, C, Vin = X_preds[:, 0], X_preds[:, 1], X_preds[:, 2], X_preds[:, 4]
denom = np.sqrt((1 - omega*2 * L * C)**2 + (omega * R * C)**2)
equation = Vin / denom

# Tracé
plt.plot(X_preds[:, 3], equation, label="Equation", alpha=0.6)
plt.plot(X_preds[:, 3], y_preds, 'o', label="Données")
plt.plot(X_preds[:, 3], preds_net, label="Réseau classique", alpha=0.6)
plt.plot(X_preds[:, 3], preds_pinn, label="PINN", alpha=0.6)
plt.legend()
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude V_C")
plt.title("Réponse en fréquence du circuit RLC")
plt.grid(True)
#plt.show()

# ----- Extension de la fréquence -----
X_train = df[["R", "L", "C", "Frequency", "Vin"]].values[100:130, :]
y_train = df[["V_C"]].values[100:130, :]

# Dernière ligne
R_last, L_last, C_last, f_last, Vin_last = X_train[-1]
f_step = 100
f_future = np.arange(f_last + f_step, f_last + 100 * f_step, f_step)

X_future = np.column_stack([
    np.full_like(f_future, R_last),
    np.full_like(f_future, L_last),
    np.full_like(f_future, C_last),
    f_future,
    np.full_like(f_future, Vin_last)
])

X_total = np.vstack([X_train, X_future])

preds_net_total = net.predict(X_total)
preds_pinn_total = net_pinn.predict(X_total)

# Solution analytique
omega = 2 * np.pi * X_total[:, 3]
R, L, C, Vin = X_total[:, 0], X_total[:, 1], X_total[:, 2], X_total[:, 4]
denom = np.sqrt((1 - omega*2 * L * C)**2 + (omega * R * C)**2)
equation_total = Vin / denom
# Tracé final
plt.figure(figsize=(10, 6))
plt.plot(X_train[:, 3], y_train, 'o', label="Données d'entraînement")
plt.plot(X_total[:, 3], equation_total, label="Equation", alpha=0.6)
plt.plot(X_total[:, 3], preds_net_total, label="Réseau classique", alpha=0.6)
plt.plot(X_total[:, 3], preds_pinn_total, label="PINN", alpha=0.6)
plt.title("Réponse AC RLC - Extension en fréquence")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude de V_C")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()