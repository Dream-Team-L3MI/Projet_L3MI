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

# === Load and Normalize Data ===
data = pd.read_csv("rl_trans_I.csv")
inputs = data[['R', 'L', 'Time_s', 'Vin_V']].values
outputs = data[['I_A']].values

scaler_in = MinMaxScaler()
scaler_out = MinMaxScaler()
inputs_norm = scaler_in.fit_transform(inputs)
outputs_norm = scaler_out.fit_transform(outputs)

in_train, in_test, out_train, out_test = train_test_split(inputs_norm, outputs_norm, test_size=0.1, random_state=42)

in_train_tensor = torch.tensor(in_train, dtype=torch.float32)
out_train_tensor = torch.tensor(out_train, dtype=torch.float32)
in_test_tensor = torch.tensor(in_test, dtype=torch.float32)
out_test_tensor = torch.tensor(out_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(in_train_tensor, out_train_tensor), batch_size=64, shuffle=True)

# === PINN Lightning Module ===
class PINNLightning(pl.LightningModule):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1, layers=8, scaler_in=None, scaler_out=None):
        super().__init__()
        self.scaler_in = scaler_in
        self.scaler_out = scaler_out

        layers_list = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(layers - 1):
            layers_list += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers_list.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers_list)
        self.automatic_optimization = False

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        opt1, opt2 = self.optimizers()

        y_pred = self(x)
        data_loss = nn.MSELoss()(y_pred, y)

        # === Collocation: draw new random points for physics loss ===
        n_phys = 3 * x.size(0)
        x_colloc = torch.rand(n_phys, 4, device=self.device)
        for i in range(4):
            x_colloc[:, i] = x_colloc[:, i] * (self.scaler_in.data_max_[i] - self.scaler_in.data_min_[i]) + self.scaler_in.data_min_[i]
        x_colloc = torch.tensor(self.scaler_in.transform(x_colloc.cpu()), dtype=torch.float32, device=self.device)
        x_colloc.requires_grad_(True)
        y_colloc = self(x_colloc)

        # === Unscale physical params ===
        Δ = self.scaler_in.data_max_ - self.scaler_in.data_min_
        ΔI = self.scaler_out.data_max_[0] - self.scaler_out.data_min_[0]
        I_min = self.scaler_out.data_min_[0]

        R = x_colloc[:, 0] * Δ[0] + self.scaler_in.data_min_[0]
        L = x_colloc[:, 1] * Δ[1] + self.scaler_in.data_min_[1]
        Vin = x_colloc[:, 3] * Δ[3] + self.scaler_in.data_min_[3]

        I_pred = y_colloc.squeeze() * ΔI + I_min
        dI_dt_norm = torch.autograd.grad(
            y_colloc, x_colloc,
            grad_outputs=torch.ones_like(y_colloc),
            create_graph=True, retain_graph=True
        )[0][:, 2]
        dI_dt = dI_dt_norm * ΔI / Δ[2]

        residual = R * I_pred + L * dI_dt - Vin
        physics_loss = torch.mean(residual ** 2)

        # === Adaptive lambda ===
        with torch.no_grad():
            λ = (data_loss / physics_loss).clamp(0.01, 100)

        total_loss = data_loss + λ * physics_loss

        # === Manual optimization logic
        current_epoch = self.trainer.current_epoch
        if current_epoch < 100:
            opt = opt1  # Adam
        else:
            opt = opt2  # L-BFGS

        opt.zero_grad()

        if current_epoch < 100:
            self.manual_backward(total_loss)
            opt.step()
        else:
            def closure():
                opt.zero_grad()
                self.manual_backward(total_loss)
                return total_loss

            opt.step(closure)

        self.log_dict({'train_loss': total_loss, 'data_loss': data_loss, 'phys_loss': physics_loss, 'lambda_phys': λ}, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=1e-3)
        lbfgs = torch.optim.LBFGS(self.parameters(), lr=1.0, max_iter=500, history_size=50)
        return [adam, lbfgs]  # Use Adam for 100 steps, then switch to L-BFGS

# === Training ===
model = PINNLightning(hidden_dim=64, layers=8, scaler_in=scaler_in, scaler_out=scaler_out)
trainer = pl.Trainer(
    max_epochs=200,
    callbacks=[
        ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min'),
        EarlyStopping(monitor='train_loss', patience=20, mode='min')
    ],
    logger=pl.loggers.CSVLogger("logs/", name="pinn_logs"),
    log_every_n_steps=10
)
trainer.fit(model, train_loader)

# === Evaluation ===
model.eval()
with torch.no_grad():
    preds_norm = model(in_test_tensor).cpu().numpy()

preds = scaler_out.inverse_transform(preds_norm)
true = scaler_out.inverse_transform(out_test_tensor.cpu().numpy())
test_time = in_test[:, 2]
sorted_idx = test_time.argsort()
time_sorted = scaler_in.inverse_transform(in_test)[sorted_idx, 2]
true_sorted = true[sorted_idx]
preds_sorted = preds[sorted_idx]

mse = mean_squared_error(true_sorted, preds_sorted)
print(f"MSE sur test set : {mse:.6f}")

# === Plot ===
plt.figure(figsize=(12, 6))
N = min(len(true_sorted), 1000)
plt.plot(time_sorted[:N], true_sorted[:N], label='I(t) réel')
plt.plot(time_sorted[:N], preds_sorted[:N], '--', label='I(t) prédit', alpha=0.7)
plt.title(f"PINN I(t) Prediction (MSE: {mse:.2e})")
plt.xlabel("Temps (s)")
plt.ylabel("Courant (A)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
