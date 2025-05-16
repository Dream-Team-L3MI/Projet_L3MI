import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_Hz, u_H, u_rad
import numpy as np
import cmath
import csv
from itertools import product
import multiprocessing
from tqdm import tqdm
import os
import signal
import sys

# Configuration
logger = Logging.setup_logging()
os.environ['PYSPICE_SIMULATION_TEMP_DIR'] = os.path.join(os.getcwd(), 'tmp')

def signal_handler(sig, frame):
    print("\nArrêt propre en cours...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def simulate_rl_ac(params):
    """Fonction de simulation avec gains"""
    R, L, Vin, freq = params
    try:
        circuit = Circuit('RL Circuit')
        circuit.SinusoidalVoltageSource(1, 'vin', circuit.gnd, amplitude=Vin, frequency=freq)
        circuit.R(1, 'vin', 'n1', R @ u_Ohm)
        circuit.L(1, 'n1', circuit.gnd, L @ u_H)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=freq, stop_frequency=freq, number_of_points=1, variation='lin')

        vin = complex(analysis['vin'][0])
        vn1 = complex(analysis['n1'][0])
        V_R = vin - vn1
        V_L = vn1

        # Calcul des gains (nouveau)
        gain_bas = abs(V_R) / abs(vin)  # VR/Vin
        gain_haut = abs(V_L) / abs(vin)  # VL/Vin

        return {
            'R': float(R),
            'L': float(L),
            'V_in': float(Vin),
            'Frequency_Hz': float(freq),
            'V_R': float(abs(V_R)),
            'V_L': float(abs(V_L)),
            'gain_bas': float(gain_bas),        # Ajouté
            'gain_haut': float(gain_haut),      # Ajouté
            'phase_V_R_rad': float(cmath.phase(V_R)),
            'phase_V_L_rad': float(cmath.phase(V_L))
        }
    except Exception:
        return None

def writer_worker(output_file, queue):
    """Mise à jour avec les nouvelles colonnes"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'R', 'L', 'V_in', 'Frequency_Hz',
            'V_R', 'V_L', 'gain_bas', 'gain_haut',  # Colonnes ajoutées
            'phase_V_R_rad', 'phase_V_L_rad'
        ])
        writer.writeheader()
        while True:
            result = queue.get()
            if result == 'STOP':
                break
            writer.writerow(result)
            f.flush()  # ← Nécessaire pour éviter la perte de données

if __name__ == '__main__':
    # Paramètres pour 1 million de points (50×50×20×20)
    r_values = np.round(np.geomspace(10, 10e3, 50), 2)
    l_values = np.round(np.geomspace(1e-4, 1, 50), 6)
    vin_values = np.round(np.linspace(1, 24, 20), 2)
    freq_values = np.round(np.geomspace(1, 10e6, 20), 2)

    param_combinations = list(product(r_values, l_values, vin_values, freq_values))

    print(f"Lancement de {len(param_combinations):,} simulations...")

    result_queue = multiprocessing.Queue()
    writer_process = multiprocessing.Process(
        target=writer_worker, args=("rl_ac_1M.csv", result_queue)
    )
    writer_process.start()

    with multiprocessing.Pool() as pool:
        try:
            for result in tqdm(
                pool.imap_unordered(simulate_rl_ac, param_combinations),
                total=len(param_combinations),
                desc="Progression"
            ):
                if result:
                    result_queue.put(result)
        finally:
            result_queue.put('STOP')
            writer_process.join()

    print("Terminé ! Fichier : rl_ac_1M.csv")