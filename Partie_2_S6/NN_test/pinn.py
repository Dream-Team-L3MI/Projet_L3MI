import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        batch_size=32,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def fit(self, X_np, y_np):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_tensor = torch.tensor(self.scaler_X.fit_transform(X_np), dtype=torch.float32, requires_grad=True).to(device)
        y_tensor = torch.tensor(self.scaler_y.fit_transform(y_np), dtype=torch.float32, requires_grad=True).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)

        data = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            for X_batch, y_batch in data:
                optimiser.zero_grad()
                outputs = self.forward(X_batch)
                loss = self.loss(y_batch, outputs)
                if self.loss2:
                    loss += self.loss2_weight * self.loss2(self, X_batch)
                loss.backward()
                optimiser.step()
                losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.4f}")
        return losses

    def predict(self, X):
        X_tensor = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32).to(device)
        self.eval()
        out = self.forward(X_tensor)
        out = out.detach().cpu().numpy()
        return self.scaler_y.inverse_transform(out)
