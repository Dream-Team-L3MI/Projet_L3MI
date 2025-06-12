"""
    This file is for testing the trained model on RC transient circuit.
    No training in this file.
    All data must be newly read and standardise before evaluating.
    Normalisation must use the scalers used in training.

"""

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv 
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import random


# Check Hardware (mps)

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# Read data from csv file

df = pd.read_csv("rc_transient.csv")
#print(df)
print(df.shape)

df.describe()



# Data normalisation
test_size = 1000

start = 1000000
end = 1100000

R = df[["R"]].values[start:end]
C = df[["C"]].values[start:end]
t = df[["Time"]].values[start:end]
Vin = df[["Vin"]].values[start:end]
Vout = df[["Vout"]].values[start:end]


# Load data scalers

scaler_R = joblib.load('scaler_R.pkl')
scaler_C = joblib.load('scaler_C.pkl')
scaler_t = joblib.load('scaler_t.pkl')
scaler_vin = joblib.load('scaler_vin.pkl')
scaler_vout = joblib.load('scaler_vout.pkl')

R_tensor = torch.tensor(scaler_R.transform(R), dtype = torch.float64)
C_tensor = torch.tensor(scaler_C.transform(C), dtype = torch.float64)
t_tensor = torch.tensor(scaler_t.transform(t), dtype = torch.float64)
Vin_tensor = torch.tensor(scaler_vin.transform(Vin), dtype = torch.float64)
Vout_tensor = torch.tensor(scaler_vout.transform(Vout), dtype = torch.float64)

# Model class for instanciation

nb_neu = 128
nb_neu2 = 128

class RegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, nb_neu)
        self.fc2 = nn.Linear(nb_neu,nb_neu2)
        self.fc3 = nn.Linear(nb_neu, out_features)
        


    def forward(self, r, c, t, v):
        inp_combined = torch.cat([r, c, t, v], dim = 1)
        x = F.silu(self.fc1(inp_combined))
        x = F.silu(self.fc2(x))
        x = (self.fc3(x))  # No activation on output for regression
        return x



# Load the saved model and evaluate

my_model = RegressionModel(4, 1).double()
my_model.load_state_dict(torch.load("My_model", weights_only=True))
my_model.eval()

criterion = nn.MSELoss()


with torch.no_grad():
    # Forward pass
    pred = my_model(R_tensor, C_tensor, t_tensor, Vin_tensor)  

    

    # Calculate the loss (optional)
    loss = criterion(pred, Vout_tensor)
    print(f"Evaluation Loss: {loss.item():.3f}")

    # Inverse transform predictions
    pred2_ori = scaler_vout.inverse_transform(pred.numpy())


# Plot the tested data
"""plt.title("Tested data")
plt.xlabel("Time")
plt.ylabel("U")
plt.plot(t[:72], pred2_ori[:72], label = 'predicted',  color = 'deeppink')
plt.plot(t[:72], Vout[:72], label = 'true', color = 'blue', marker = '*', linestyle = 'None')
plt.legend()

plt.show()
"""

# Corelation graph 
plt.figure(figsize=(6, 6))
plt.scatter(Vout, pred2_ori, alpha=0.5, label = "predicted", color = 'blue', s=0.5)
plt.plot([Vout.min(), pred2_ori.max()], [Vout.min(), pred2_ori.max()], '--', label = "test data", color = 'lime')  # Perfect line
plt.xlabel("True Vout")
plt.ylabel("Predicted Vout")
plt.title("Scatter Plot of Predictions (Corelation)")
plt.legend()
plt.grid(True)
plt.show()