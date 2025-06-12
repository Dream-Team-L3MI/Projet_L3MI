import pandas as pd
import numpy as np
import sys
import shutil
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# === Logger TensorBoard ===
#Afin de visualiser les métriques
logger = TensorBoardLogger("tb_logs", name="PINN_rl_experiment")

# === Chargement et filtrage des données ===
data = pd.read_csv("rl_transit.csv")
data = data[data['Vin_V'] > 0.01] # suppression des phases Vin ≈ 0 pour éviter le bruit

I_max = data['I_A'].max()
print(f"I_max détecté automatiquement : {I_max:.4f}")

# === Préparation des features ===
#Séparation des variables entrée / sortie
inputs = data[['R', 'L', 'Time_norm', 'Vin_V']].values
outputs = data[['I_A']].values

# === Normalisation ===
scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()
inputs_normalized = scaler_in.fit_transform(inputs)
outputs_normalized = scaler_out.fit_transform(outputs)

# === Split train/test ===
in_train, in_test, out_train, out_test = train_test_split(
    inputs_normalized, outputs_normalized, test_size=0.1, random_state=42
)

# === Conversion en tenseurs PyTorch ===
in_train_tensor = torch.tensor(in_train, dtype=torch.float32)
out_train_tensor = torch.tensor(out_train, dtype=torch.float32)
in_test_tensor = torch.tensor(in_test, dtype=torch.float32)
out_test_tensor = torch.tensor(out_test, dtype=torch.float32)

# === Dataset et DataLoader ===
train_dataset = TensorDataset(in_train_tensor, out_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

val_dataset = TensorDataset(in_test_tensor, out_test_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# === PINN corrigé avec dérivée physique (sans pénalité sur I<0) ===
class PINN(pl.LightningModule):
    def __init__(self, input_dim=4, hidden_dim=50, output_dim=1, layers=4,
                 lambda_phys=1.0, scaler_in=None):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.scaler_in = scaler_in
        self.scale_t = torch.tensor(scaler_in.scale_[2], dtype=torch.float32)
        self.I_max = I_max

        #Architechture du réseau
        layers_list = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 1):
            layers_list += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        layers_list.append(nn.Tanh())  # Activation finale: meilleure que Sigmoid

        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        #Projection finale dans [0, I_max]
        return self.I_max * (self.net(x) + 1) / 2  # Projection dans [0, I_max]

    def training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad_(True)

        i_pred = self(x)
        data_loss = nn.MSELoss()(i_pred, y)

        #Dérivée temporelle de I(t) normalisé
        dI_dtnorm = torch.autograd.grad(
            i_pred, x,
            grad_outputs=torch.ones_like(i_pred),
            create_graph=True, retain_graph=True
        )[0][:, 2]
        scale_t = self.scale_t.to(dI_dtnorm.device)
        dI_dt_phys = dI_dtnorm / scale_t

        #Calcul du résidu physique
        R = x[:, 0]
        L = x[:, 1]
        Vin = x[:, 3]
        lhs = R * i_pred.squeeze() + L * dI_dt_phys
        physics_residual = Vin - lhs


        #Pondération temporelle : favorise temps élévés
        time_weights = x[:, 2]
        time_weights = 0.5 + 1.0 * time_weights
        physics_loss = torch.mean(time_weights * physics_residual ** 2)

        #Pondération dynamique: augmente vc les epochs
        epoch = self.current_epoch
        lambda_dyn = min(1.0, epoch / 50)

        total_loss = data_loss + lambda_dyn * self.lambda_phys * physics_loss

        self.log_dict({
            'train_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss
        })
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Activation explicite des gradients pour la validation (utile pour autograd)
        torch.set_grad_enabled(True)
        x.requires_grad_(True)

        #prdiction du courant I(t)
        i_pred = self(x)
        # Calcul de la perte data classique (erreur MSE sur la prédiction)
        data_loss = nn.MSELoss()(i_pred, y)

        # Calcul de la dérivée temporelle normalisée dI/dt_norm via autograd
        dI_dtnorm = torch.autograd.grad(
            i_pred, x,
            grad_outputs=torch.ones_like(i_pred),
            create_graph=True, retain_graph=True
        )[0][:, 2]  # On extrait ∂I/∂(Time_norm)

        # Conversion vers une vraie dérivée physique dI/dt en tenant compte du facteur de normalisation
        scale_t = self.scale_t.to(dI_dtnorm.device)
        dI_dt_phys = dI_dtnorm / scale_t

        # Récupération des variables physiques depuis x
        R, L, Vin = x[:, 0], x[:, 1], x[:, 3]

        # Application de la loi de Kirchhoff : Vin = R.I + L.dI/dt
        lhs = R * i_pred.squeeze() + L * dI_dt_phys
        physics_residual = Vin - lhs

        # Pondération temporelle : donne plus d’importance aux points à temps élevé (transitoire)
        time_weights = x[:, 2]
        time_weights = 0.5 + 1.0 * time_weights
        physics_loss = torch.mean(time_weights * physics_residual ** 2)

        # Pondération dynamique croissante avec les epochs (au début focus sur data_loss, puis sur physics_loss)
        epoch = self.current_epoch
        lambda_dyn = min(1.0, epoch / 50)

        # Perte totale = data + pondération dynamique * physique
        total_loss = data_loss + lambda_dyn * self.lambda_phys * physics_loss

        self.log_dict({
            'val_loss': total_loss,
            'val_data_loss': data_loss,
            'val_physics_loss': physics_loss
        }, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)

# === Initialisation du modèle ===
model = PINN(hidden_dim=100, layers=5, lambda_phys=1.0, scaler_in=scaler_in)

# === Log des hyperparamètres dans TensorBoard===
logger.log_hyperparams({
    "hidden_dim": 100,
    "layers": 5,
    "lambda_phys": 1.0,
    "I_max": I_max,
    "activation": "Tanh",
    "batch_size": 64,
    "lr": 1e-3,
    "data_file": "rl_transit.csv"
})

# === Définition des Callbacks ===
checkpoint_callback = ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min')
early_stop_callback = EarlyStopping(monitor='train_loss', patience=10, mode='min')

# === Entraînement ===
trainer = pl.Trainer(max_epochs=100, callbacks=[checkpoint_callback, early_stop_callback],
                     enable_progress_bar=True, logger=logger)
trainer.fit(model, train_loader, val_loader)

# === Copie du script pour traçabilité dans les logs TensorBoard ===
source_file = __file__ if "__file__" in globals() else "ton_script.py"
target_folder = logger.log_dir
try:
    shutil.copy(source_file, os.path.join(target_folder, "used_script.py"))
except Exception as e:
    print("Copie du code source échouée :", e)

# === Évaluation globale/Finale sur le test set ===
model.eval()
with torch.no_grad():
    preds_norm = model(in_test_tensor).cpu().numpy()
preds = scaler_out.inverse_transform(preds_norm)
true = scaler_out.inverse_transform(out_test_tensor.cpu().numpy())
mse = mean_squared_error(true, preds)
print(f"MSE : {mse:.6f}")

# === Prédiction et tracé sur une seule simulation ===
df = pd.read_csv("rl_transit.csv")
df = df[df['Vin_V'] > 0.01]

sim_id = 999  # ou un id donné via sys.argv
sim = df[df['id'] == sim_id].sort_values(by='Time_s')

X_sim = sim[['R', 'L', 'Time_norm', 'Vin_V']].values
X_sim_tensor = torch.tensor(scaler_in.transform(X_sim), dtype=torch.float32)

with torch.no_grad():
    I_pred_norm = model(X_sim_tensor).cpu().numpy()
    I_pred = scaler_out.inverse_transform(I_pred_norm)

plt.figure(figsize=(8, 5))
plt.plot(sim['Time_s'], sim['I_A'], label='I(t) réel')
plt.plot(sim['Time_s'], I_pred, '*', color='purple', label='I(t) prédit (★)', alpha=0.9)
vin_display = sim['Vin_V'].iloc[0]
plt.title(f"I(t) pour id={sim_id} (Vin = {vin_display:.3f} V)")
plt.xlabel("Temps (s)")
plt.ylabel("Courant (A)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
