import csv
import numpy as np
from itertools import product
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_H, u_F
from tqdm import tqdm

def simulate_dc_op(params):
    V_in, R, L, C = params
    try:
        circuit = Circuit('RLC DC Steady-State')
        circuit.V(1, 'input', circuit.gnd, V_in @ u_V)
        circuit.R(1, 'input', 'n1', R @ u_Ohm)
        circuit.L(1, 'n1', 'n2', L @ u_H)
        circuit.C(1, 'n2', circuit.gnd, C @ u_F)

        sim = circuit.simulator()
        op = sim.operating_point()

        V_input = float(op['input'])
        V_n1 = float(op['n1'])
        V_n2 = float(op['n2'])

        V_R = V_input - V_n1
        V_L = V_n1 - V_n2
        V_C = V_n2

        return {
            'V_in': V_in,
            'R': R,
            'L': L,
            'C': C,
            'V_R': V_R,
            'V_L': V_L,
            'V_C': V_C,
            'error': ''
        }

    except Exception as e:
        return {
            'V_in': V_in,
            'R': R,
            'L': L,
            'C': C,
            'V_R': None,
            'V_L': None,
            'V_C': None,
            'error': str(e)
        }

if __name__ == '__main__':
    V_range = np.linspace(1, 10, 100)          
    R_range = np.linspace(1, 1000, 100)         
    L_range = np.linspace(1e-6, 1e-3, 10)       
    C_range = np.linspace(1e-9, 1e-6, 10)    

    param_grid = list(product(V_range, R_range, L_range, C_range))
    total = len(param_grid)

    output_file = 'rlc_dc_operating_point.csv'
    fieldnames = ['V_in', 'R', 'L', 'C', 'V_R', 'V_L', 'V_C', 'error']

    print(f"Running {total} steady-state DC simulations...")

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in tqdm(map(simulate_dc_op, param_grid), total=total, desc='Simulating'):
            writer.writerow(result)

    print(f"Simulations completed. Results saved to '{output_file}'.")
