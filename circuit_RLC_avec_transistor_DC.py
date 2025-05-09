import csv
import numpy as np
from itertools import product
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_H, u_F
import multiprocessing
from tqdm import tqdm


def generate_full_parameter_grid(r_count=10, l_count=10, c_count=10, vin_count=10):
    r_vals   = np.linspace(1, 1000, 10)
    l_vals = np.linspace(1e-6, 1e-3, 10)
    c_vals   = np.linspace(1e-9, 1e-6, 10)

    vin_vals = np.linspace(1, 10, 10)
    return list(product(r_vals,l_vals, c_vals, vin_vals))

def simulate_rc_transistor_dc(params):
    R_value, L_value, C_value, Vin = params
    try:
        circuit = Circuit(f'RC + NPN R={R_value:.1f}Ω L={L_value:.1e}H C={C_value:.1e}F Vin={Vin:.1f}V')

        # Source
        circuit.V(1, 'vin', circuit.gnd, Vin @ u_V)

        # RLC
        circuit.R(1, 'vin', 'vout', R_value @ u_Ohm)
        circuit.L(1, 'vin', 'vout', L_value @ u_H)
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
        V_L = None
        V_C    = V_out

        #V_R = float(analysis.nodes['vin']) - float(analysis.nodes['n1'])
        #V_L = float(analysis.nodes['n1']) - float(analysis.nodes['n2'])
        #V_C = float(analysis.nodes['n2'])

        return {
            'R': R_value,
            'L': L_value,
            'C': C_value,
            'Vin': Vin,
            'V_in': V_in,
            'V_out': V_out,
            #'V_in': float(analysis.nodes['vin']),
            #'V_out': float(analysis.nodes['n2']),
            'V_R': V_R,
            'V_L': V_L,
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
            'L': L_value,
            'Vin': Vin,
            'V_in': None,
            'V_out': None,
            'V_R': None,
            'V_L': None,
            'V_C': None,
            'V_base': None,
            'V_collector': None,
            'V_emitter': None,
            'error': str(e)
        }

if __name__ == "__main__":
    param_grid = generate_full_parameter_grid(10,10,10, 10)
    total = len(param_grid)
    print(f"🔄 Lancement de {total:,} simulations RC+Transistor en parallèle...")

    output_file = "Final_rlc_transistor_dataset.csv"
    fieldnames = [
        'R', 'L', 'C', 'Vin', 'V_in', 'V_out', 'V_R', 'V_L', 'V_C',
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