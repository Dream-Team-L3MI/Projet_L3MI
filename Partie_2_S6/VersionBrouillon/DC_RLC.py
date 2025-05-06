import csv
import numpy as np
from itertools import product
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_H, u_F
import multiprocessing
from tqdm import tqdm

def simulate_rlc_dc(params):
    V_in, R, L, C = params
    try:
        circuit = Circuit('DC RLC')
        circuit.V(1, 'input', circuit.gnd, V_in @ u_V)
        circuit.R(1, 'input', 'n1', R @ u_Ohm)
        circuit.L(1, 'n1', 'n2', L @ u_H)
        circuit.C(1, 'n2', circuit.gnd, C @ u_F)

        sim = circuit.simulator()
        analysis = sim.operating_point()

        V_R = float(analysis['n1'])         
        V_C = float(analysis['n2'])     
        V_L = V_C - V_R                      

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
    V_range = np.linspace(1, 10, 10)           
    R_range = np.linspace(1, 1000, 100)        
    L_range = np.linspace(1e-6, 1e-3, 100)    
    C_range = np.linspace(1e-9, 1e-6, 10)      

    param_grid = list(product(V_range, R_range, L_range, C_range))
    total = len(param_grid)

    print(f"Running {total:,} RLC DC simulations with multiprocessing...")

    output_file = 'rlc_dc_1M.csv'
    fieldnames = ['V_in', 'R', 'L', 'C', 'V_R', 'V_L', 'V_C', 'error']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with multiprocessing.Pool(processes=4) as pool:
            for result in tqdm(pool.imap_unordered(simulate_rlc_dc, param_grid, chunksize=100),
                               total=total, desc='Simulating'):
                writer.writerow(result)

    print("All simulations completed and saved.")


