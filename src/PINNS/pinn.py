import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# On choisit le device disponible parmi MPS (Apple Silicon), CUDA (GPU Nvidia) ou CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Classe générale, customisable pour créer des pinns
######### Entrées #########
#   input_dim (int) : Dimension des entrées.
#   output_dim (int) : Dimension des sorties.
#   epochs (int : 1000) : Nombre d'epochs pour l'entrainement.
#   batch_size (int : 32) : Taille des batchs.
#   loss (function : nn.MSELoss()) : Fonction de perte principale (données).
#   lr (float : 1e-3) : Taux d'apprentissage.
#   loss2 (function : None) : Fonction de perte physique optionnelle.
#   loss2_weight (float : 1e-1) : Poids de la perte physique.
###########################
class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
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

        # 5 couches cachées linéaires de 256 neurones, avec fonction d'activation tanh entre chaque
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

        # Couche de sortie linéaire
        self.out = nn.Linear(256, output_dim)

    # Fonction forward: calcule la sortie du réseau pour une entrée donnée
    ######### Entrées #########
    #   x (tensor) : tensor d'entrée
    ###########################
    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    # Fonction d'entrainement du modèle
    ######### Entrées #########
    #   X_np (np.vector) : Données d'entrainement en entrée
    #   y_np (np.vector) : Données d'entrainement en sortie
    ###########################
    def fit(self, X_np, y_np):

        # Normalisation des données d'entrée et sortie avec MinMaxScaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Conversion des vecteurs en tensor PyTorch
        X_tensor = torch.tensor(self.scaler_X.fit_transform(X_np), dtype=torch.float32, requires_grad=True).to(device)
        y_tensor = torch.tensor(self.scaler_y.fit_transform(y_np), dtype=torch.float32, requires_grad=True).to(device)

        # Création du dataset et du dataloader
        dataset = TensorDataset(X_tensor, y_tensor)

        data = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Définition de l'optimiseur, ici Adam
        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []

        # Boucle d'entrainement
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

    # Fonction de prédiction : Prédit la sortie à partir d'une entrée X, à utiliser après l'entrainement
    ######### Entrées #########
    #   X (np.Vector) : Vecteur d'entrée.
    ###########################
    def predict(self, X):
        # Normalisation et conversion en tensor
        X_tensor = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32).to(device)
        self.eval()

        # Prédiciton
        out = self.forward(X_tensor)

        # Dénormalisation et conversion en vecteur numpy
        out = out.detach().cpu().numpy()
        return self.scaler_y.inverse_transform(out)
