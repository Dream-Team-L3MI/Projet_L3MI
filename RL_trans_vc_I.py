import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_H, u_s
from PySpice.Probe.Plot import plot
import numpy as np
import csv
from tqdm import tqdm
from itertools import product
import signal
import sys

# === Gestion de l'interruption clavier ===
def signal_handler(sig, frame):
    print('\nArrêt demandé, sauvegarde des données...')
    save_data(dataset)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Fonction de sauvegarde CSV ===
def save_data(data):
    with open("rl_trans_I.csv", "w", newline='') as f:
        fieldnames = ['R', 'L', 'Time_s', 'Vin_V', 'Vout_V','I_A']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Données sauvegardées ({len(data)} lignes) dans rl_trans_I.csv")

# === Simulation transitoire RL ===
def simulate_rl_transient(R_value, L_value, Vin_value):
    circuit = Circuit(f"RL Transient R={R_value} L={L_value}")

    # Source en échelon de tension (pulse rapide)
    """circuit.SinusoidalVoltageSource('input', 'vin', circuit.gnd,
                                    amplitude=Vin_value,
                                    frequency=frequency,
                                    offset=0)"""
    circuit.PulseVoltageSource('input', 'vin', circuit.gnd,
                               initial_value=0@u_V,
                               pulsed_value=Vin_value @ u_V,
                               pulse_width=10@u_s,
                               period=20@u_s)

    circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L_value @ u_H)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # Calcul de la constante de temps τ = L / R
    tau = L_value / R_value  # en secondes

    stop_time = 3 * tau       # Observation sur 3τ
    #step_time = tau / 100     # 100 points par τ
    step_time = tau / 200  # → ~300 points (au lieu de 300)

    try:
        analysis = simulator.transient(step_time=step_time @ u_s, end_time=stop_time @ u_s)

        times = analysis.time
        vin = analysis['vin']
        vout = analysis['n1']  # Tension sur la bobine
        iL = analysis['L1']   #Courant ds la bobine

        results = []
        for t, v_in, v_out, iL in zip(times, vin, vout, iL):
            #current = (float(v_in) - float(v_out)) / float(R_value)  # I = (Vin - Vout) / R

            results.append({
                'R': float(R_value),
                'L': float(L_value),
                'Time_s': float(t),
                'Vin_V': float(v_in),
                'Vout_V': float(v_out),
                'I_A': float(iL)  # <--- Ajout du courant
            })
        return results

    except Exception as e:
        print(f"Erreur pour R={R_value}, L={L_value} : {str(e)}")
        return None

# === Paramètres à tester ===

vin_values = np.round(np.linspace(1, 10, 2), 2)       # inchangé (2)
r_values = np.round(np.geomspace(10, 1000, 20), 2)  # 20 valeurs
l_values = np.round(np.geomspace(1e-4, 1e-2, 25), 6)  # 25 valeurs
# Total combinaisons : 2 × 20 × 25 = 1000

# Chaque combinaison → ~1000 lignes (300~400 points)
# On peut choisir de tronquer le nombre de points à ~1000 par combinaison


# === Combinaisons des paramètres ===
param_combinations = list(product(r_values, l_values, vin_values))
total_combinations = len(param_combinations)
print(f"Nombre total de combinaisons : {total_combinations}")
print(f"Estimation approximative du nombre total de lignes : ~{total_combinations * 1000}")

dataset = []

# === Lancement des simulations ===
with tqdm(total=total_combinations, desc="Progression") as pbar:
    for R, L, Vin in param_combinations:
        result = simulate_rl_transient(R, L, Vin)
        if result is not None:
            dataset.extend(result)
        pbar.update(1)

        # Sauvegarde temporaire toutes les 100 combinaisons
        if len(dataset) > 0 and pbar.n % 100 == 0:
            save_data(dataset)

# === Sauvegarde finale ===
save_data(dataset)
print("\nSimulation terminée. Données sauvegardées dans rl_trans_I.csv")
