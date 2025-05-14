import numpy as np
import csv
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_s, u_Ohm, u_H
from PySpice.Probe import plot
import matplotlib.pyplot as plt

def simulate_rl_transient(R_value, L_value, Vin_value, t_stop, n_points=1000):
    circuit = Circuit('RL Transient')
    circuit.PulseVoltageSource(1, 'vin', circuit.gnd,
                                initial_value=0@u_V, pulsed_value=Vin_value@u_V,
                                pulse_width=10@u_s, period=20@u_s, rise_time=1@u_s, fall_time=1@u_s,
                                delay_time=0@u_s)

    circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L_value @ u_H)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=(t_stop/n_points)@u_s, end_time=t_stop@u_s)

    time = np.array(analysis.time)
    current = np.array(analysis.branches['v1'])  # courant source = courant boucle

    I_final = Vin_value / R_value
    I_max = np.max(current)
    tau = L_value / R_value
    try:
        t_63_idx = np.where(current >= 0.63 * I_final)[0][0]
        t_63 = time[t_63_idx]
    except IndexError:
        t_63 = None  # courant n'a pas encore atteint 63% de I_final

    return {
        'R': R_value, 'L': L_value, 'Vin': Vin_value,
        't_stop': t_stop,
        'I_final': I_final, 'I_max': I_max, 't_63': t_63, 'tau': tau
    }

# Paramètres de balayage
r_values = np.linspace(100, 1000, 5)
l_values = np.linspace(1e-3, 10e-3, 5)
vin_values = np.linspace(1.0, 10.0, 3)
t_stop_values = np.linspace(1e-3, 10e-3, 4)

dataset = []
print("Début des simulations transitoires...")

for R in r_values:
    for L in l_values:
        for Vin in vin_values:
            for t_stop in t_stop_values:
                result = simulate_rl_transient(R, L, Vin, t_stop)
                dataset.append(result)

# Écriture du CSV
with open("rl_dataset_transient.csv", "w", newline='') as f:
    fieldnames = ['R', 'L', 'Vin', 't_stop', 'I_final', 'I_max', 't_63', 'tau']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dataset)

print("Simulation transitoire RL terminée. Résultats dans rl_dataset_transient.csv")
