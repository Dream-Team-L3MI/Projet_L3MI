import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_H, u_s, u_ms
import numpy as np
import csv
from itertools import product
from tqdm import tqdm

logger = Logging.setup_logging()

def simulate_rl_transient(R, L, Vin, t_max, t_step):
    """Simulation RL transitoire avec sortie standardisée"""
    circuit = Circuit('RL Transient')
    circuit.PulseVoltageSource(1, 'vin', circuit.gnd,
                              initial_value=0@u_V, pulsed_value=Vin@u_V,
                              pulse_width=t_max@u_s, period=(2*t_max)@u_s)
    circuit.R(1, 'vin', 'n1', R@u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L@u_H)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=t_step@u_s, end_time=t_max@u_s)

    # Extraction des résultats (tableau numpy)
    time = np.array(analysis.time)
    V_in = np.array(analysis['vin'])
    V_R = np.array(analysis['vin'] - analysis['n1'])
    V_L = np.array(analysis['n1'])
    I_L = np.array(analysis['n1'] / R)  # I = V_L / R (approximation)

    return {
        'R': R,
        'L': L,
        'Vin': Vin,
        'time': time,
        'V_in': V_in,
        'V_R': V_R,
        'V_L': V_L,
        'I_L': I_L
    }

def save_to_csv(data, filename):
    """Sauvegarde les données transitoires dans un CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['R', 'L', 'Vin', 'time', 'V_in', 'V_R', 'V_L', 'I_L'])
        for i in range(len(data['time'])):
            writer.writerow([
                data['R'], data['L'], data['Vin'],
                data['time'][i], data['V_in'][i],
                data['V_R'][i], data['V_L'][i], data['I_L'][i]
            ])

# Paramètres de simulation
r_values = [100, 1000]  # 2 résistances pour l'exemple
l_values = [1e-3, 10e-3]  # 2 inductances
vin_values = [5, 10]  # 2 tensions d'entrée
t_max = 10e-3  # 10 ms de simulation
t_step = 1e-5  # Pas de 10 µs

dataset = []
print("Lancement des simulations RL transitoires...")

# Boucle de simulation
for R, L, Vin in tqdm(product(r_values, l_values, vin_values),
                      total=len(r_values)*len(l_values)*len(vin_values)):
    result = simulate_rl_transient(R, L, Vin, t_max, t_step)
    save_to_csv(result, f"rl_transient_R{R}_L{L}_Vin{Vin}.csv")
    dataset.append(result)  # Pour analyse ultérieure

print("Simulations terminées !")

