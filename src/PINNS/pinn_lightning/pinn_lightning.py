import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
import lightning as L

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Net(L.LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim,
        x,
        y,
        batch_size=32,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=1,
    ) -> None:
        super().__init__()

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.out = nn.Linear(256, output_dim)

        self._prepare_data()

    def _prepare_data(self):
        x_tensor = torch.tensor(self.scaler_x.fit_transform(self.x), dtype=torch.float32, requires_grad=True).to(device)
        y_tensor = torch.tensor(self.scaler_y.fit_transform(self.y), dtype=torch.float32).to(device)

        dataset = TensorDataset(x_tensor, y_tensor)
        train_size = int(0.95 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=self.lr)

        return optimiser

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch

        outputs = self(x_batch)

        loss = self.loss(y_batch, outputs)
        if self.loss2:
            loss += self.loss2_weight * self.loss2(self, x_batch)

        self.log("train_loss", loss, prog_bar=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch

        outputs = self(x_batch)

        loss = self.loss(y_batch, outputs)
        if self.loss2:
            loss += self.loss2_weight * self.loss2(self, x_batch)

        self.log("val_loss", loss, prog_bar=True)

        return {'val_loss': loss}

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)



# class Net(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         output_dim,
#         epochs=1000,
#         batch_size=32,
#         loss=nn.MSELoss(),
#         lr=1e-3,
#         loss2=None,
#         loss2_weight=0.1,
#         scaler=MinMaxScaler(),
#     ) -> None:
#         super().__init__()
#
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.loss = loss
#         self.loss2 = loss2
#         self.loss2_weight = loss2_weight
#         self.lr = lr
#         self.scaler = scaler
#
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.Tanh(),
#             nn.Linear(256, 256),
#             nn.Tanh(),
#             nn.Linear(256, 256),
#             nn.Tanh(),
#             nn.Linear(256, 256),
#             nn.Tanh(),
#             nn.Linear(256, 256),
#             nn.Tanh(),
#         )
#         self.out = nn.Linear(256, output_dim)
#
#     def forward(self, x):
#         h = self.layers(x)
#         out = self.out(h)
#
#         return out
#
#     def fit(self, X_np, y_np):
#         self.scaler_X = self.scaler
#         self.scaler_y = self.scaler
#
#         X_tensor = torch.tensor(self.scaler_X.fit_transform(X_np), dtype=torch.float32, requires_grad=True).to(device)
#         y_tensor = torch.tensor(self.scaler_y.fit_transform(y_np), dtype=torch.float32, requires_grad=True).to(device)
#
#         dataset = TensorDataset(X_tensor, y_tensor)
#
#         data = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
#         optimiser = optim.Adam(self.parameters(), lr=self.lr)
#         self.train()
#         losses = []
#         for ep in range(self.epochs):
#             for X_batch, y_batch in data:
#                 optimiser.zero_grad()
#                 outputs = self.forward(X_batch)
#                 loss = self.loss(y_batch, outputs)
#                 if self.loss2:
#                     loss += self.loss2_weight * self.loss2(self, X_batch)
#                 loss.backward()
#                 optimiser.step()
#                 losses.append(loss.item())
#             if ep % int(self.epochs / 10) == 0:
#                 print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.4f}")
#         return losses
#
#     def predict(self, X):
#         X_tensor = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32).to(device)
#         self.eval()
#         out = self.forward(X_tensor)
#         out = out.detach().cpu().numpy()
#         return self.scaler_y.inverse_transform(out)
