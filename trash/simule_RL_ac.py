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

# Configuration pour maximiser les performances
os.environ['PYSPICE_USE_GPU'] = '1'  # Si disponible
os.environ['PYSPICE_SIMULATION_TEMP_DIR'] = os.path.join(os.getcwd(), 'tmp')


def signal_handler(sig, frame):
    print('\nArrêt demandé, sauvegarde des données...')
    save_data(dataset)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def save_data(data):
    """Sauvegarde incrémentielle des données"""
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


def simulate_rl_ac_batch(params):
    """Simulation par lot pour meilleure performance"""
    R, L, Vin, freq = params
    circuit = Circuit(f'RL Circuit R={R} L={L} Vin={Vin} Freq={freq}')

    circuit.SinusoidalVoltageSource(1, 'vin', circuit.gnd,
                                    amplitude=Vin,
                                    frequency=freq)
    circuit.R(1, 'vin', 'n1', R @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L @ u_H)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    try:
        analysis = simulator.ac(start_frequency=freq,
                                stop_frequency=freq,
                                number_of_points=1,
                                variation='lin')

        vin = analysis['vin'].as_ndarray()[0]
        vn1 = analysis['n1'].as_ndarray()[0]
        V_R = vin - vn1
        V_L = vn1

        return {
            'R': float(R),
            'L': float(L),
            'V_in': float(Vin),
            'Frequency_Hz': float(freq),
            'V_R': float(abs(V_R)),
            'V_L': float(abs(V_L)),
            'gain_bas': float(abs(V_R) / abs(vin)),
            'gain_haut': float(abs(V_L) / abs(vin)),
            'phase_V_R_rad': float(cmath.phase(V_R)),
            'phase_V_L_rad': float(cmath.phase(V_L))
        }
    except Exception as e:
        print(f"x", end='', flush=True)  # Indicateur d'erreur minimal
        return None


# Paramètres étendus (×2 à ×5 plus de valeurs que précédemment)
vin_values = np.round(np.linspace(1, 24, 5), 2)  # 12 valeurs de 1V à 24V
r_values = np.round(np.geomspace(10, 10e3, 10), 2)  # 20 valeurs de 10Ω à 10kΩ (échelle log)
l_values = np.round(np.geomspace(1e-4, 1, 10), 6)  # 25 valeurs de 0.1mH à 1H (échelle log)
frequency_values = np.round(np.geomspace(1, 10e6, 20), 2)  # 30 valeurs de 1Hz à 10MHz

# Préparation des combinaisons de paramètres
from itertools import product

param_combinations = product(r_values, l_values, vin_values, frequency_values)
total_combinations = len(r_values) * len(l_values) * len(vin_values) * len(frequency_values)

print(f"\nGénération de données RL AC étendue")
print(f"Total combinaisons: {total_combinations:,}")
print(f"V_in: {len(vin_values)} valeurs de {min(vin_values)}V à {max(vin_values)}V")
print(f"R: {len(r_values)} valeurs de {min(r_values)}Ω à {max(r_values)}Ω")
print(f"L: {len(l_values)} valeurs de {min(l_values):.4f}H à {max(l_values):.4f}H")
print(f"Fréquences: {len(frequency_values)} de {min(frequency_values)}Hz à {max(frequency_values)}Hz")

dataset = []
batch_size = 100  # Sauvegarde par lots pour performance
batch_count = 0

with tqdm(total=total_combinations, desc="Simulation", unit="point") as pbar:
    for params in product(r_values, l_values, vin_values, frequency_values):
        result = simulate_rl_ac_batch(params)
        if result is not None:
            dataset.append(result)
            batch_count += 1

            if batch_count % batch_size == 0:
                save_data(dataset)
                dataset = []  # Réinitialise pour le prochain lot

        pbar.update(1)

# Sauvegarde finale des données restantes
if dataset:
    save_data(dataset)

# Analyse des résultats
print("\nRécapitulatif final:")
print(f"Combinaisons totales essayées: {total_combinations:,}")
print(f"Simulations réussies: {batch_count:,} ({batch_count / total_combinations:.1%})")

"""if batch_count > 0:
    phases_vr = [x['phase_V_R_rad'] for x in dataset]
    phases_vl = [x['phase_V_L_rad'] for x in dataset]
    print(f"\nPlage des phases VR: {min(phases_vr):.4f} à {max(phases_vr):.4f} rad")
    print(f"Plage des phases VL: {min(phases_vl):.4f} à {max(phases_vl):.4f} rad")
    print("Données sauvegardées dans rl_dataset_ac_extended.csv")
"""

# Vérification basique du CSV
try:
    with open("rl_dataset_ac_extended.csv", 'r') as f:
        line_count = sum(1 for _ in f) - 1  # Compte les lignes (moins l'en-tête)
    print(f"\nLignes dans le CSV : {line_count:,}")
    print("Les colonnes 'phase_V_R_rad' et 'phase_V_L_rad' contiennent les valeurs brutes.")
except Exception as e:
    print(f"\nErreur de vérification du CSV : {str(e)}")

print("Traitement terminé avec succès!")

