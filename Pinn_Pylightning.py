import pandas as pd
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
data = data[:10000]

data['V_R'] = data['Vin_V'] - data['Vout_V']  # Tension aux bornes R (pas utilisée ici mais c'est ok)

inputs = data[['R', 'L', 'Time_s',  'Vin_V',]].values
outputs = data[['I_A']].values  # Courant à prédire

# Normalisation
scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()

inputs_normalized = scaler_in.fit_transform(inputs)
outputs_normalized = scaler_out.fit_transform(outputs)

# Split train/test
in_train, in_test, out_train, out_test = train_test_split(
    inputs_normalized, outputs_normalized, test_size=0.2, random_state=42
)

# Convert to torch tensors
in_train_tensor = torch.tensor(in_train, dtype=torch.float32)
out_train_tensor = torch.tensor(out_train, dtype=torch.float32)
in_test_tensor = torch.tensor(in_test, dtype=torch.float32)
out_test_tensor = torch.tensor(out_test, dtype=torch.float32)

# Dataset and DataLoader
train_dataset = TensorDataset(in_train_tensor, out_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Modèle Lightning
class PINNLightning(pl.LightningModule):
    def __init__(self, input_dim=4, hidden_dim=10, output_dim=1, layers=5):
        super().__init__()
        layers_list = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 1):
            layers_list += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)

    def pinn_loss(self, batch):
        x, y = batch
        i_pred = self(x)
        data_loss = nn.MSELoss()(i_pred, y)

        # Termes physique
        x.requires_grad_(True)
        i_pred_colloc = self(x)
        dI_dt = torch.autograd.grad(
            i_pred_colloc, x,
            grad_outputs=torch.ones_like(i_pred_colloc),
            create_graph=True, retain_graph=True
        )[0][:, 2]  # dérivée par rapport au temps (4e colonne)

        R = x[:, 0]
        L = x[:, 1]
        Time = x[:, 2]
        Vin = x[:, 3]

        lhs = R * i_pred_colloc.squeeze() + L * dI_dt
        physics_residual = lhs - Vin
        physics_loss = torch.mean(physics_residual ** 2)

        return data_loss + physics_loss

    def training_step(self, batch, batch_idx):
        loss = self.pinn_loss(batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Instanciation du modèle
model = PINNLightning()

# Callbacks
checkpoint_callback = ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min')
early_stop_callback = EarlyStopping(monitor='train_loss', patience=10, mode='min')

# Trainer Lightning
trainer = pl.Trainer(max_epochs=100,
                     callbacks=[checkpoint_callback, early_stop_callback],
                     log_every_n_steps=10)

# Entraînement
trainer.fit(model, train_loader)

# Evaluation sur test set
model.eval()
with torch.no_grad():
    preds_norm = model(in_test_tensor).cpu().numpy()

# Dénormalisation
preds = scaler_out.inverse_transform(preds_norm)
true = scaler_out.inverse_transform(out_test_tensor.cpu().numpy())

mse = mean_squared_error(true, preds)
print(f"MSE sur test set: {mse:.6f}")

# Visualisation

plt.plot(true, label='I(t) réel')
plt.plot(preds, label='I(t) prédit')
plt.legend()
plt.title("Prédiction du courant I(t) avec PyTorch Lightning PINN")
plt.show()
