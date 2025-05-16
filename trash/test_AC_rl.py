import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import numpy as np

def simulate_rl_ac(R_value, L_value, frequency_hz):
    # Créer un circuit
    circuit = Circuit(f"RL Circuit AC - R={R_value}Ω L={L_value}H f={frequency_hz}Hz")

    # ✅ Source AC d'amplitude 1V, utilisée dans analyse AC
    circuit.V('input', 'vin', circuit.gnd, 0@u_V)  # Source DC 0V
    circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
    circuit.L(1, 'n1', circuit.gnd, L_value @ u_H)

    # Créer simulateur
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # Analyse AC (fournit un signal sinusoïdal de 1V AC automatiquement)
    analysis = simulator.ac(start_frequency=frequency_hz, stop_frequency=frequency_hz,
                            number_of_points=1, variation='dec')

    # Extraire les tensions complexes
    V_in = analysis['vin'][0]  # complexe
    V_n1 = analysis['n1'][0]   # complexe

    V_R = V_in - V_n1
    V_L = V_n1

    gain_R = abs(V_R)
    gain_L = abs(V_L)
    phase_R = np.angle(V_R, deg=True)
    phase_L = np.angle(V_L, deg=True)

    return {
        'R': R_value,
        'L': L_value,
        'f': frequency_hz,
        'gain_R': gain_R,
        'gain_L': gain_L,
        'phase_R_deg': phase_R,
        'phase_L_deg': phase_L
    }

# 🔁 Exemple
if __name__ == "__main__":
    print("Lancement des simulations RL en régime AC...")
    result = simulate_rl_ac(R_value=100, L_value=1e-3, frequency_hz=1000)
    print(result)
