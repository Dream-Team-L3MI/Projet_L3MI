import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_Hz, u_H, u_F
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
    print("\nArr\u00eat propre en cours...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def simulate_rlc_ac(params):
    """Simulation d'un circuit RLC en r\u00e9gime AC"""
    R, L, C, Vin, freq = params
    try:
        circuit = Circuit('RLC Series Circuit')
        circuit.SinusoidalVoltageSource(1, 'vin', circuit.gnd, amplitude=Vin, frequency=freq)
        circuit.R(1, 'vin', 'n1', R @ u_Ohm)
        circuit.L(1, 'n1', 'n2', L @ u_H)
        circuit.C(1, 'n2', circuit.gnd, C @ u_F)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=freq, stop_frequency=freq, number_of_points=1, variation='lin')

        vin = complex(analysis['vin'][0])
        vn1 = complex(analysis['n1'][0])
        vn2 = complex(analysis['n2'][0])

        V_R = vin - vn1
        V_L = vn1 - vn2
        V_C = vn2

        gain_bas = abs(V_C) / abs(vin)
        gain_haut = abs(V_L) / abs(vin)
        gain_bande = abs(V_R) / abs(vin)

        return {
            'R': float(R),
            'L': float(L),
            'C': float(C),
            'V_in': float(Vin),
            'Frequency_Hz': float(freq),
            'V_R': float(abs(V_R)),
            'V_L': float(abs(V_L)),
            'V_C': float(abs(V_C)),
            'gain_bas': float(gain_bas),
            'gain_haut': float(gain_haut),
            'gain_bande': float(gain_bande),
            'phase_V_R_rad': float(cmath.phase(V_R)),
            'phase_V_L_rad': float(cmath.phase(V_L)),
            'phase_V_C_rad': float(cmath.phase(V_C))
        }
    except Exception:
        return None

def writer_worker(output_file, queue):
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'R', 'L', 'C', 'V_in', 'Frequency_Hz',
            'V_R', 'V_L', 'V_C',
            'gain_bas', 'gain_haut', 'gain_bande',
            'phase_V_R_rad', 'phase_V_L_rad', 'phase_V_C_rad'
        ])
        writer.writeheader()
        while True:
            result = queue.get()
            if result == 'STOP':
                break
            writer.writerow(result)
            f.flush()

if __name__ == '__main__':
    r_values = np.round(np.geomspace(10, 10e3, 20), 2)
    l_values = np.round(np.geomspace(1e-4, 1, 20), 6)
    c_values = np.round(np.geomspace(1e-9, 1e-4, 10), 9)
    vin_values = np.round(np.linspace(1, 24, 10), 2)
    freq_values = np.round(np.geomspace(1, 10e6, 25), 2)

    param_combinations = list(product(r_values, l_values, c_values, vin_values, freq_values))

    print(f"Lancement de {len(param_combinations):,} simulations...")

    result_queue = multiprocessing.Queue()
    writer_process = multiprocessing.Process(
        target=writer_worker, args=("rlc_ac.csv", result_queue)
    )
    writer_process.start()

    with multiprocessing.Pool() as pool:
        try:
            for result in tqdm(
                pool.imap_unordered(simulate_rlc_ac, param_combinations),
                total=len(param_combinations),
                desc="Progression"
            ):
                if result:
                    result_queue.put(result)
        finally:
            result_queue.put('STOP')
            writer_process.join()

    print("Termin\u00e9 ! Fichier : rlc_ac_mil.csv")
