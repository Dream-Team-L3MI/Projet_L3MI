import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

import random 
import torch.nn.functional as F

#Suivant cet structure : 
#1 preparation des données , 2 . modele de regression , 3. entrainement du modele , 4. évalution , 5. visualisation des résultats
 # Initialise random seed for model weights and activations

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # If using torch.backends (optional for CPU, more relevant for CUDA)
    if torch.backends.mps.is_available():
        torch.use_deterministic_algorithms(True)

set_seed(42)
#chargement de la dataframe 
df = pd.read_csv( "AC_RL.csv" )
print( df.shape )


# Normalisation des données 
X = df[ [ "R" , "L" , "V_in" , "Frequency_Hz" , "V_R" , "V_L"  ] ].values
Y = df[ [ "gain_bas", "gain_haut" , "phase_V_R_rad" , "phase_V_L_rad" ] ].values
# gain_bas,gain_bas,phase_V_R_rad,phase_V_L_rad
Y_GB = df[ [ "gain_bas" ] ].values
Y_GH = df[ [ "gain_bas" ] ].values
Y_PR = df[ [ "phase_V_R_rad" ] ].values
Y_PL = df[ [ "phase_V_L_rad" ] ].values


scaler_x = StandardScaler()

scaler_y = StandardScaler()

scaler_y_GB = StandardScaler()
scaler_y_GH = StandardScaler()
scaler_y_PR = StandardScaler()
scaler_y_PL = StandardScaler()

X_tensor = torch.tensor( scaler_x.fit_transform(X) , dtype = torch.float32 )

Y_tensor = torch.tensor( scaler_y.fit_transform(Y) , dtype = torch.float32 )
Y_tensor_GB = torch.tensor( scaler_y_GB.fit_transform(Y_GB) , dtype = torch.float32 )
Y_tensor_GH = torch.tensor( scaler_y_GH.fit_transform(Y_GH) , dtype = torch.float32 )
Y_tensor_PR = torch.tensor( scaler_y_PR.fit_transform(Y_PR) , dtype = torch.float32 )
Y_tensor_PL = torch.tensor( scaler_y_PL.fit_transform(Y_PL) , dtype = torch.float32 )

#print(X_tensor[:,0])
#print(Y_tensor)

dataset = TensorDataset( X_tensor , Y_tensor )

dataset_GB = TensorDataset( X_tensor , Y_tensor_GB )
dataset_GH = TensorDataset( X_tensor , Y_tensor_GH )
dataset_PR = TensorDataset( X_tensor , Y_tensor_PR )
dataset_PL = TensorDataset( X_tensor , Y_tensor_PL )

#loader = DataLoader(dataset, batch_size=32, shuffle=True)

R = X_tensor[ :,0 ]
L = X_tensor[ :,1 ]
Frequency = X_tensor[ :,3 ]
print( R )
print( L )


print( ( R*L ).shape )


class RegressionModel(nn.Module):
    def __init__( self , in_features , out_features = 4 ):
        super().__init__()

        """"
        nn.Linear( a , b ) crée une couche fully connected

            a : neurones en input
            b : neurones en output
        """
        self.fc1 = nn.Linear( in_features , 64 )
        self.fc2 = nn.Linear( 64 , 64 )
        self.fc3 = nn.Linear( 64 , 64 )
        self.fc4 = nn.Linear( 64 , 64 )
        self.fc5 = nn.Linear( 64 , 64 )
        self.fc6 = nn.Linear( 64 , out_features )
        


    def forward( self , x ):
        x = F.tanh( self.fc1( x ) )
        x = F.tanh( self.fc2( x ) )
        x = F.tanh( self.fc3( x ) )
        x = F.tanh( self.fc4( x ) )
        x = F.tanh( self.fc5( x ) )
        x = self.fc6( x )  # No activation on output for regression
        return x


# Définir input et output dimension selon tes données
input_dim = X_tensor.shape[1]  
output_dim = Y_tensor.shape[1]  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialisation du modèle
model = RegressionModel(input_dim, output_dim).to(device)

# Définition de l'optimiseur
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Boucle d'entraînement classique
n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss = F.mse_loss(pred, Y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item()}")

#evaluation 
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    loss = F.mse_loss(predictions, Y_tensor)
    print(f"Validation Loss: {loss.item()}")

#denormalisation des données 
Y_pred_denorm = scaler_y.inverse_transform(predictions.cpu().numpy())
Y_reel_denorm = scaler_y.inverse_transform(Y_tensor.cpu().numpy())

#visualisation des résultats 
# Visualisation : courbes ET scatter plots
labels = ["gain_bas", "gain_bas", "phase_V_R_rad", "phase_V_L_rad"]
for i in range(4):
    # Courbe (vrai vs prédit pour chaque point)
    plt.figure(figsize=(6, 4))
    plt.plot(Y_reel_denorm[:, i], label="Vrai", color='blue')
    plt.plot(Y_pred_denorm[:, i], label="Prédit", color='orange', linestyle='--')
    plt.title(f"Comparaison {labels[i]} (courbe)")
    plt.xlabel("Exemples")
    plt.ylabel(labels[i])
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Scatter (vrai vs prédit dans l'espace)
    plt.figure(figsize=(5, 5))
    plt.scatter(Y_reel_denorm[:, i], Y_pred_denorm[:, i], alpha=0.5, color='purple')
    plt.plot([Y_reel_denorm[:, i].min(), Y_reel_denorm[:, i].max()],
             [Y_reel_denorm[:, i].min(), Y_reel_denorm[:, i].max()],
             color='red', linestyle='--', label="y = x")
    plt.title(f"Scatter : Vrai vs Prédit ({labels[i]})")
    plt.xlabel("Vrai")
    plt.ylabel("Prédit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.axis("equal")
    plt.show()
