import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging() #pr activer les logs de la simulation

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_H, u_s
from PySpice.Probe.Plot import plot #(non utilisé ici)
import numpy as np
import csv
from tqdm import tqdm
from itertools import product #Produit cartésien des paramètres
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
    with open("rl_tran_vc_periode.csv", "w", newline='') as f:
        fieldnames = ['id', 'R', 'L', 'Time_s', 'Vin_V', 'Vout_V','I_A', 'tau', 'Time_norm']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Données sauvegardées ({len(data)} lignes) dans rl_tran_vc_periode.csv")

# === Simulation transitoire RL ===
def simulate_rl_transient(R_value, L_value, Vin_value, sim_id):
    circuit = Circuit(f"RL Transient R={R_value} L={L_value}")

    tau = L_value / R_value  # Constante de temps RL
    stop_time = 3 * tau   #Transitoire jusqu'à 3*tau
    step_time = tau / 200    #Resolution temporelle fine (comme avant)

    #  Convertir en µs pour les unités PySpice
    #On simule un échelon de tension en utilisant Pulse vc une période très grande pr éviter que le I(t) devienne négatif
    pulse_width = (stop_time + tau) @ u_s         # large pulse
    period = (10 * stop_time) @ u_s               # jamais atteint dans la simu
    #Période très longue => on reste sur la 1ère impulsion

    # Signal = échelon (step-like) en utilisant Pulse avec période très grande
    circuit.PulseVoltageSource('input', 'vin', circuit.gnd,
                               initial_value=0@u_V,
                               pulsed_value=Vin_value @ u_V,
                               pulse_width=pulse_width,
                               period=period)

    #Définition des composantes du RL
    circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L_value @ u_H)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    try:
        #Simulation transitoire
        analysis = simulator.transient(step_time=step_time @ u_s, end_time=stop_time @ u_s)

        #Récupération des signaux simulés
        times = analysis.time
        vin = analysis['vin']   #Tension d'entrée
        vout = analysis['n1']   #Tension sur la bobine
        iL = analysis['L1']     #Courant dans l'inductance

        #Création des lignes de données
        results = []
        for t, v_in, v_out, i in zip(times, vin, vout, iL):
            results.append({
                'id': sim_id,
                'R': float(R_value),
                'L': float(L_value),
                'Time_s': float(t),
                'Vin_V': float(v_in),
                'Vout_V': float(v_out),
                'I_A': float(i),
                'tau': float(tau),
                'Time_norm': float(t) / float(tau)  #le tmp normalisé
            })
        return results

    except Exception as e:
        print(f"Erreur pour R={R_value}, L={L_value} : {str(e)}")
        return None


# === Paramètres à tester ===
vin_values = np.round(np.linspace(1, 10, 5), 2)     # 5 valeurs de Vin entre 1V et 10V
r_values = np.round(np.geomspace(10, 1000, 15), 2)     # 15 valeurs de R (10 à 1000 Ohms)
l_values = np.round(np.geomspace(1e-4, 1e-2, 15), 6)    # 15 valeurs de L  (100microH à am mH)


# Produit cartésien des combinaisons (5 × 15 × 15)
param_combinations = list(product(r_values, l_values, vin_values))
total_combinations = len(param_combinations)
print(f"Nombre total de combinaisons : {total_combinations}")
print(f"Estimation approximative du nombre total de lignes : ~{total_combinations * 1000}")

dataset = []
sim_id = 0

# === Lancement des simulations vc barre de progression ===
with tqdm(total=total_combinations, desc="Progression") as pbar:
    for R, L, Vin in param_combinations:
        result = simulate_rl_transient(R, L, Vin, sim_id)
        if result is not None:
            dataset.extend(result)
            sim_id += 1
        pbar.update(1)

        #Sauvegarde périodique toute les 100 simulations
        if len(dataset) > 0 and pbar.n % 100 == 0:
            save_data(dataset)

# === Sauvegarde finale ===
save_data(dataset)
print("\nSimulation terminée. Données sauvegardées dans rl_tran_vc_periode.csv")
