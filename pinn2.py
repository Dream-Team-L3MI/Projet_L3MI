import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Chargement des données
data = pd.read_csv("rl_trans_I.csv")
# Réduction temporaire du dataset à 10 000 lignes
#data = data[:10000]

data['V_R'] = data['Vin_V'] - data['Vout_V']  # Pas utilisé ici

# Transformation logarithmique du temps pour mieux gérer les échelles
#data['Time_s'] = np.log1p(data['Time_s'])

# Préparation des entrées et sorties
inputs = data[['R', 'L', 'Time_s', 'Vin_V']].values
outputs = data[['I_A']].values


# Normalisation
scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()

inputs_normalized = scaler_in.fit_transform(inputs)
outputs_normalized = scaler_out.fit_transform(outputs)

# Split train/test
in_train, in_test, out_train, out_test = train_test_split(
    inputs_normalized, outputs_normalized, test_size=0.2, random_state=42
)

# Convertir en tenseurs PyTorch
in_train_tensor = torch.tensor(in_train, dtype=torch.float32)
out_train_tensor = torch.tensor(out_train, dtype=torch.float32)
in_test_tensor = torch.tensor(in_test, dtype=torch.float32)
out_test_tensor = torch.tensor(out_test, dtype=torch.float32)

# Dataset et DataLoader
train_dataset = TensorDataset(in_train_tensor, out_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Modèle PINN avec PyTorch Lightning
class PINNLightning(pl.LightningModule):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, layers=8, scaler_in=None, scaler_out=None,):
        super().__init__()
        self.scaler_in = scaler_in  #store the scaler
        layers_list = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 1):
            layers_list += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        i_pred = self(x)
        data_loss = nn.MSELoss()(i_pred, y)
        # Time scale factor for chain rule: ∂I/∂t_physical = ∂I/∂t_scaled / Δt
        time_scale = self.scaler_in.data_max_[2] - self.scaler_in.data_min_[2]

        x.requires_grad_(True)
        i_pred_colloc = self(x)
        dI_dt = torch.autograd.grad(
            i_pred_colloc, x,
            grad_outputs=torch.ones_like(i_pred_colloc),
            create_graph=True, retain_graph=True
        )[0][:, 2]  # dérivée par rapport à Time_s (3e colonne)

        dI_dt = dI_dt / time_scale

        R = x[:, 0]
        L = x[:, 1]
        Vin = x[:, 3]

        # === Unscale the physical parameters (they were scaled with MinMaxScaler)
        R_unscaled = R * (self.scaler_in.data_max_[0] - self.scaler_in.data_min_[0]) + self.scaler_in.data_min_[0]
        L_unscaled = L * (self.scaler_in.data_max_[1] - self.scaler_in.data_min_[1]) + self.scaler_in.data_min_[1]
        Vin_unscaled = Vin * (self.scaler_in.data_max_[3] - self.scaler_in.data_min_[3]) + self.scaler_in.data_min_[3]


        #lhs = R * i_pred_colloc.squeeze() + L * dI_dt
        #physics_residual = Vin - lhs

        physics_residual = R_unscaled * (i_pred_colloc.squeeze())+ L_unscaled * dI_dt - Vin_unscaled
        physics_loss = torch.mean(physics_residual ** 2)

        #total_loss = data_loss + physics_loss

        lambda_phys = 0.1
        total_loss = data_loss + lambda_phys * physics_loss

        self.log('train_loss', total_loss)
        self.log('data_loss', data_loss)
        self.log('physics_loss', physics_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Instanciation du modèle
model = PINNLightning(hidden_dim=64, layers=8, scaler_in=scaler_in)

# Callbacks
checkpoint_callback = ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min')
early_stop_callback = EarlyStopping(monitor='train_loss', patience=10, mode='min')

# Entraînement
trainer = pl.Trainer(max_epochs=100,
                     callbacks=[checkpoint_callback, early_stop_callback],
                     log_every_n_steps=10)

trainer.fit(model, train_loader)

# Évaluation sur le jeu de test
model.eval()
with torch.no_grad():
    preds_norm = model(in_test_tensor).cpu().numpy()

# Dénormalisation
preds = scaler_out.inverse_transform(preds_norm)
true = scaler_out.inverse_transform(out_test_tensor.cpu().numpy())

# Tri des résultats pour affichage ordonné dans le temps
test_time = in_test[:, 2]  # 3e colonne = Time_s
sorted_indices = test_time.argsort()

#preds_sorted = preds[sorted_indices]
#true_sorted = true[sorted_indices]



# Visualisation triée par le temps pour éviter le "gribouillage"
#sorted_indices = in_test[:, 2].argsort()  # Time_s est 3e colonne → index 2
true_sorted = true[sorted_indices]
preds_sorted = preds[sorted_indices]
time_sorted = scaler_in.inverse_transform(in_test)[sorted_indices, 2]  # temps réel

# Calcul de l'erreur
mse = mean_squared_error(true_sorted, preds_sorted)
print(f"MSE sur test set : {mse:.6f}")


"""# Affichage des résultats
plt.figure(figsize=(12, 6))
N = 2000  # Affiche seulement les 2000 premiers points pour la clarté
plt.plot(true_sorted[:N], label='I(t) réel')
plt.plot(preds_sorted[:N], label='I(t) prédit', alpha=0.7)
plt.legend()
plt.grid(True)
plt.title("Prédiction du courant I(t) avec PyTorch Lightning PINN")
plt.xlabel("Index (ordonné par temps)")
plt.ylabel("Courant (A)")
plt.show()"""
trainer.logger = pl.loggers.CSVLogger("logs/", name="pinn_logs")

#Après dénormalisation, avant le plot:
plt.figure(figsize=(12, 6))
N = min(len(true_sorted), 1000)
#plt.plot(test_time[sorted_indices][:N], true_sorted[:N], label='I(t) réel')
#plt.plot(test_time[sorted_indices][:N], preds_sorted[:N], '--', label='I(t) prédit', alpha=0.7)
plt.plot(time_sorted, true_sorted, label='I(t) réel')
plt.plot(time_sorted, preds_sorted, '--', label='I(t) prédit', alpha=0.7)
plt.legend()
plt.grid(True)
plt.title(f"Prédiction du courant I(t) (MSE: {mse:.2e})")
plt.xlabel("Temps (s)")  # Utilisez le temps réel plutôt que l'index
plt.ylabel("Courant (A)")
plt.tight_layout()
plt.show()
