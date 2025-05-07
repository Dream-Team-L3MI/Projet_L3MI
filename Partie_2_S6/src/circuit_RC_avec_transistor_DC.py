import csv
import numpy as np
from itertools import product
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_F
import multiprocessing
from tqdm import tqdm

def generate_full_parameter_grid(r_count=10, c_count=10, vin_count=10):
    r_vals   = np.linspace(1e2, 1e5, r_count)
    c_vals   = np.linspace(1e-9, 1e-5, c_count)
    vin_vals = np.linspace(0.1, 10.0, vin_count)
    return list(product(r_vals, c_vals, vin_vals))

def simulate_rc_transistor_dc(params):
    R_value, C_value, Vin = params
    try:
        circuit = Circuit(f'RC + NPN R={R_value:.1f}Ω C={C_value:.1e}F Vin={Vin:.1f}V')

        # Source
        circuit.V(1, 'vin', circuit.gnd, Vin @ u_V)

        # RC
        circuit.R(1, 'vin', 'vout', R_value @ u_Ohm)
        circuit.C(1, 'vout', circuit.gnd, C_value @ u_F)

        # Transistor NPN générique
        circuit.model('NPN', 'NPN')
        circuit.R(2, 'vout', 'base', 1e3 @ u_Ohm)
        circuit.R(3, 'vin', 'collector', 1e3 @ u_Ohm)
        circuit.R(4, 'emitter', circuit.gnd, 1e3 @ u_Ohm)
        circuit.BJT(1, 'collector', 'base', 'emitter', model='NPN')

        sim = circuit.simulator()
        analysis = sim.operating_point()

        V_in   = float(analysis.nodes['vin'])
        V_out  = float(analysis.nodes['vout'])
        V_R    = V_in - V_out
        V_C    = V_out

        return {
            'R': R_value,
            'C': C_value,
            'Vin': Vin,
            'V_in': V_in,
            'V_out': V_out,
            'V_R': V_R,
            'V_C': V_C,
            'V_base': float(analysis.nodes['base']),
            'V_collector': float(analysis.nodes['collector']),
            'V_emitter': float(analysis.nodes['emitter']),
            'error': ''
        }

    except Exception as e:
        return {
            'R': R_value,
            'C': C_value,
            'Vin': Vin,
            'V_in': None,
            'V_out': None,
            'V_R': None,
            'V_C': None,
            'V_base': None,
            'V_collector': None,
            'V_emitter': None,
            'error': str(e)
        }

if __name__ == "__main__":
    param_grid = generate_full_parameter_grid(100, 100, 100)
    total = len(param_grid)
    print(f"🔄 Lancement de {total:,} simulations RC+Transistor en parallèle...")

    output_file = "Final_rc_transistor_dataset.csv"
    fieldnames = [
        'R', 'C', 'Vin', 'V_in', 'V_out', 'V_R', 'V_C',
        'V_base', 'V_collector', 'V_emitter', 'error'
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with multiprocessing.Pool() as pool:
            for result in tqdm(pool.imap_unordered(simulate_rc_transistor_dc, param_grid, chunksize=100),
                               total=total, desc="Simulations"):
                writer.writerow(result)

    print(f" {total:,} simulations terminées. Résultats enregistrés dans '{output_file}'.")