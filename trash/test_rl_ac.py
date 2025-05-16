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


def signal_handler(sig, frame):
    print('\nArrêt demandé, sauvegarde des données...')
    save_data(dataset)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def save_data(data):
    """Sauvegarde les données"""
    with open("rl_dataset_ac_final.csv", "w", newline='') as f:
        fieldnames = ['R', 'L', 'V_in', 'Frequency_Hz', 'V_R', 'V_L',
                      'gain_bas', 'gain_haut', 'phase_V_R_rad', 'phase_V_L_rad']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Données sauvegardées ({len(data)} points)")


def simulate_rl_ac(R_value, L_value, Vin_value, frequency):
    """
    Simulates an RL circuit in AC regime with proper phase calculation.
    """
    circuit = Circuit(f'RL Circuit R={R_value} L={L_value} Freq={frequency}')

    # Source AC avec spécification de phase via offset (solution alternative)
    circuit.SinusoidalVoltageSource(1, 'vin', circuit.gnd,
                                    amplitude=Vin_value,
                                    frequency=frequency,
                                    offset=0)  # Important pour la référence de phase

    circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L_value @ u_H)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # Analyse AC avec calcul des phases
    try:
        analysis = simulator.ac(start_frequency=frequency,
                                stop_frequency=frequency,
                                number_of_points=1,
                                variation='dec')

        # Extraction des tensions complexes (méthode robuste)
        vin = analysis['vin']
        vn1 = analysis['n1']

        # Conversion en nombres complexes
        V_in = vin.as_ndarray()[0]
        V_R = (vin - vn1).as_ndarray()[0]
        V_L = vn1.as_ndarray()[0]

        # Calcul des phases
        phase_V_R = cmath.phase(V_R)
        phase_V_L = cmath.phase(V_L)

        # Calcul des amplitudes
        V_R_amp = abs(V_R)
        V_L_amp = abs(V_L)
        V_in_amp = abs(V_in)

        # Calcul des gains
        gain_bas = V_R_amp / V_in_amp
        gain_haut = V_L_amp / V_in_amp

        return {
            'R': float(R_value),
            'L': float(L_value),
            'V_in': float(Vin_value),
            'Frequency_Hz': float(frequency),
            'V_R': float(V_R_amp),
            'V_L': float(V_L_amp),
            'gain_bas': float(gain_bas),
            'gain_haut': float(gain_haut),
            'phase_V_R_rad': float(phase_V_R),
            'phase_V_L_rad': float(phase_V_L)
        }

    except Exception as e:
        print(f"Erreur pour R={R_value}, L={L_value}, Freq={frequency}: {str(e)}")
        return None


# Paramètres de simulation garantissant des phases non nulles
vin_values = [5.0]  # 5V pour des résultats clairs
r_values = [100, 500, 1000]  # Plage typique
l_values = [1e-3, 5e-3, 10e-3]  # 1mH à 10mH
frequency_values = [10, 100, 1000, 10000, 100000]  # De 10Hz à 100kHz

dataset = []
print("Lancement des simulations RL avec calcul de phase...")

total_simulations = len(vin_values) * len(r_values) * len(l_values) * len(frequency_values)

with tqdm(total=total_simulations, desc="Progression") as pbar:
    for Vin in vin_values:
        for R in r_values:
            for L in l_values:
                for freq in frequency_values:
                    result = simulate_rl_ac(R, L, Vin, freq)
                    if result is not None:
                        dataset.append(result)
                        # Vérification immédiate des phases
                        if abs(result['phase_V_R_rad']) < 1e-6:
                            print(f"\nAttention: phase VR nulle pour R={R}, L={L}, Freq={freq}")
                        if abs(result['phase_V_L_rad']) < 1e-6:
                            print(f"\nAttention: phase VL nulle pour R={R}, L={L}, Freq={freq}")
                    pbar.update(1)

# Sauvegarde finale
save_data(dataset)

# Affichage de vérification
print("\nVérification des résultats:")
print(f"Nombre total de simulations: {total_simulations}")
print(f"Nombre de points valides: {len(dataset)}")
if len(dataset) > 0:
    test_point = dataset[0]
    print("\nExemple de point de données:")
    print(f"Fréquence: {test_point['Frequency_Hz']} Hz")
    print(f"Phase VR: {test_point['phase_V_R_rad']:.4f} rad")
    print(f"Phase VL: {test_point['phase_V_L_rad']:.4f} rad")
    print(f"Gain bas: {test_point['gain_bas']:.4f}")
    print(f"Gain haut: {test_point['gain_haut']:.4f}")

print("\nSimulation terminée. Données sauvegardées dans rl_dataset_ac_final.csv")