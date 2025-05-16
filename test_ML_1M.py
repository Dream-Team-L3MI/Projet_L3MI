import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_Hz, u_H, u_rad
import csv
import numpy as np
import cmath
from tqdm import tqdm
import sys
import signal
import os
from itertools import product
from multiprocessing import Pool, cpu_count, Manager

# Préparer répertoire temporaire si besoin
os.environ['PYSPICE_SIMULATION_TEMP_DIR'] = os.path.join(os.getcwd(), 'tmp')
os.makedirs(os.environ['PYSPICE_SIMULATION_TEMP_DIR'], exist_ok=True)

# Pour capture propre Ctrl+C
def signal_handler(sig, frame):
    print('\nArrêt demandé, sauvegarde des données...')
    save_data(shared_dataset)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Sauvegarde incrémentielle
def save_data(data):
    filename = "rl_dataset_ac_extended.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline='') as f:
        fieldnames = ['R', 'L', 'V_in', 'Frequency_Hz', 'V_R', 'V_L',
                      'gain_bas', 'gain_haut', 'phase_V_R_rad', 'phase_V_L_rad']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)
    print(f"\nDonnées sauvegardées (+{len(data)} points)")

# Simulation unitaire
def simulate_rl_ac_batch(params):
    R, L, Vin, freq = params
    circuit = Circuit(f'RL Circuit R={R} L={L} Vin={Vin} Freq={freq}')
    circuit.SinusoidalVoltageSource(1, 'vin', circuit.gnd, amplitude=Vin, frequency=freq)
    circuit.R(1, 'vin', 'n1', R @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L @ u_H)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    try:
        analysis = simulator.ac(start_frequency=freq, stop_frequency=freq, number_of_points=1, variation='lin')
        vin = analysis['vin'].as_ndarray()[0]
        vn1 = analysis['n1'].as_ndarray()[0]
        V_R = vin - vn1
        V_L = vn1

        return {
            'R': float(R), 'L': float(L), 'V_in': float(Vin), 'Frequency_Hz': float(freq),
            'V_R': float(abs(V_R)), 'V_L': float(abs(V_L)),
            'gain_bas': float(abs(V_R) / abs(vin)),
            'gain_haut': float(abs(V_L) / abs(vin)),
            'phase_V_R_rad': float(cmath.phase(V_R)),
            'phase_V_L_rad': float(cmath.phase(V_L))
        }
    except Exception:
        return None

# Valeurs
vin_values = np.round(np.linspace(1, 24, 20), 2)
r_values = np.round(np.geomspace(10, 10e3, 50), 2)
l_values = np.round(np.geomspace(1e-4, 1, 50), 6)
frequency_values = np.round(np.geomspace(1, 10e6, 20), 2)

param_combinations = list(product(r_values, l_values, vin_values, frequency_values))
total_combinations = len(param_combinations)

print(f"\nGénération de données RL AC étendue")
print(f"Total combinaisons: {total_combinations:,}")

# MULTIPROCESSING
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    batch_size = 100
    shared_dataset = []

    with Pool(processes=cpu_count()) as pool:
        results = []
        with tqdm(total=total_combinations, desc="Simulation", unit="point") as pbar:
            for i, result in enumerate(pool.imap_unordered(simulate_rl_ac_batch, param_combinations), 1):
                if result is not None:
                    shared_dataset.append(result)

                if i % batch_size == 0 and shared_dataset:
                    save_data(shared_dataset)
                    shared_dataset = []

                pbar.update(1)

        # Sauvegarde finale
        if shared_dataset:
            save_data(shared_dataset)

    print("\nRécapitulatif final :")
    print(f"Combinaisons totales : {total_combinations}")
    print(f"Simulations réussies : {i} (voir CSV pour détails)")

    try:
        with open("rl_dataset_ac_extended.csv", 'r') as f:
            line_count = sum(1 for _ in f) - 1
        print(f"\nLignes dans le CSV : {line_count:,}")
    except Exception as e:
        print(f"Erreur lecture CSV : {e}")

    print("Traitement terminé.")
