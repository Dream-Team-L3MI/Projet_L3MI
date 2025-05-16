import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_Hz, u_H, u_rad
import csv
import numpy as np
import cmath  # Pour gérer les nombres complexes


def simulate_rl_ac(R_value, L_value, Vin_value, frequency):
    """
    Simulates an RL circuit in AC regime and calculates gains and phases.
    Returns a dictionary with all required parameters.
    """
    circuit = Circuit(f'RL Circuit R={R_value} L={L_value} Freq={frequency}')

    # Source AC avec amplitude Vin_value et phase 0
    circuit.SinusoidalVoltageSource(1, 'vin', circuit.gnd, amplitude=Vin_value, frequency=frequency)
    circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L_value @ u_H)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # Analyse AC avec une seule fréquence
    analysis = simulator.ac(start_frequency=frequency, stop_frequency=frequency, number_of_points=1, variation='dec')

    # Fréquence en Hz
    freq = float(frequency @ u_Hz)

    # Tensions complexes
    V_in_complex = complex(analysis['vin'][0])
    V_R_complex = complex(analysis['vin'][0] - analysis['n1'][0])
    V_L_complex = complex(analysis['n1'][0])

    # Amplitudes
    V_R = abs(V_R_complex)
    V_L = abs(V_L_complex)

    # Calcul des gains simulés
    gain_bas = V_R / abs(V_in_complex)  # VR / Vin
    gain_haut = V_L / abs(V_in_complex)  # VL / Vin
    # Phases en radians
    phase_V_R = cmath.phase(V_R_complex)
    phase_V_L = cmath.phase(V_L_complex)

    # Calcul des gains
    """gain_bas = R_value / np.sqrt(
        R_value ** 2 + (2 * np.pi * frequency * L_value) ** 2)  # Gain à fréquence nulle théorique
    gain_haut = 1.0  # Gain à fréquence infinie théorique (pour VL/Vin)"""

    return {
        'R': float(R_value),
        'L': float(L_value),
        'V_in': float(Vin_value),
        'Frequency_Hz': float(frequency),
        'V_R': float(V_R),
        'V_L': float(V_L),
        'gain_bas': float(gain_bas),
        'gain_haut': float(gain_haut),
        'phase_V_R_rad': float(phase_V_R),
        'phase_V_L_rad': float(phase_V_L)
    }


# Paramètres de balayage (réduits pour tester)
vin_values = np.linspace(1.0, 10.0, 2)  # Vin de 1V à 10V (amplitude)
r_values = np.linspace(100, 1000, 5)  # Ohms
l_values = np.linspace(1e-3, 10e-3, 5)  # Henries
frequency_values = np.logspace(1, 6, 5)  # Fréquences de 10Hz à 1MHz (échelle logarithmique)

dataset = []
print("Lancement des simulations RL en régime AC...")

for Vin in vin_values:
    for R in r_values:
        for L in l_values:
            for freq in frequency_values:
                try:
                    result = simulate_rl_ac(R, L, Vin, freq)
                    dataset.append(result)
                except Exception as e:
                    print(f"Erreur avec R={R}, L={L}, Vin={Vin}, Freq={freq}: {str(e)}")
                    continue

# Sauvegarde en CSV dans l'ordre demandé
with open("rl_dataset_ac.csv", "w", newline='') as f:
    fieldnames = ['R', 'L', 'V_in', 'Frequency_Hz', 'V_R', 'V_L',
                  'gain_bas', 'gain_haut', 'phase_V_R_rad', 'phase_V_L_rad']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dataset)

print(f"Simulation RL AC terminée. {len(dataset)} points calculés. Résultats dans rl_dataset_ac.csv")