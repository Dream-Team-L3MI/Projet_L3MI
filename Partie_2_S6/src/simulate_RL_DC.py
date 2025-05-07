import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V , u_Ohm, u_Hz, u_H
import csv
import numpy as np

def simulate_rl_dc(R_value, L_value, Vin_value):
    """
    Simulates an RL circuit in DC regime (steady-state).
    Returns R, L, Vin, I_out, V_R, V_L, V_out
    """
    Vin = Vin_value @ u_V

    circuit = Circuit(f'RL Circuit R={R_value} L={L_value}')
    circuit.V(1, 'vin', circuit.gnd, Vin_value)
    circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L_value @ u_H)

    simulator = circuit.simulator()
    analysis = simulator.operating_point()

    # Le courant dans l’inductance est le courant de la source
    V_R = abs(analysis['vin'] - analysis['n1'])[0]  # tension aux bornes de R
    V_L = abs(analysis['n1'])[0]

    return {'R': float(R_value), 'L': float(L_value), 'V_in': float(Vin), 'V_R': float(V_R), 'V_L': float(V_L)}

# Paramètres de balayage
vin_values = np.linspace(1.0, 10.0, 5)     # Vin de 1V à 10V
r_values = np.linspace(100, 1000, 50)  # Ohms
l_values = np.linspace(1e-3, 10e-3, 50)  # Henries

dataset = []
print("Lancement des simulations RL en régime DC...")

for Vin in vin_values:
    for R in r_values:
        for L in l_values:
            result = simulate_rl_dc(R, L, Vin)
            dataset.append(result)

# Sauvegarde en CSV
with open("rl_dataset_dc.csv", "w", newline='') as f:
    fieldnames = ['R', 'L', 'V_in', 'V_R', 'V_L']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dataset)

print("Simulation RL DC terminée. Résultats dans rl_dataset_dc.csv")
