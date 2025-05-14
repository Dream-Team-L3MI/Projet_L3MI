import csv
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_H
from itertools import product
import multiprocessing
from tqdm import tqdm
import time

def generate_full_parameter_grid():
    r_vals = np.linspace(1e2, 1e5, 100)     # 100 valeurs de R
    l_vals = np.linspace(1e-6, 1e-2, 100)   # 100 valeurs de L (de 1 µH à 10 mH)
    vin_vals = np.linspace(0.1, 10.0, 100)  # 100 valeurs de Vin
    return list(product(r_vals, l_vals, vin_vals))  # => 1 million de combinaisons

def simulate_rl_dc(parametres):
    R_value, L_value, Vin_value = parametres
    try:
        circuit = Circuit(f'RL Circuit R={R_value} L={L_value} Vin={Vin_value}')
        circuit.V(1, 'vin', circuit.gnd, Vin_value @ u_V)
        circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
        circuit.L(1, 'n1', circuit.gnd, L_value @ u_H)

        simulator = circuit.simulator()
        analysis = simulator.operating_point()

        V_in = float(Vin_value)
        V_R = abs(analysis.nodes['vin'][0] - analysis.nodes['n1'][0])
        V_L = abs(analysis['n1'])[0]


        return {
            'R': R_value,
            'L': L_value,
            'Vin': V_in,
            'V_R': V_R,
            'V_L': V_L,
            'error': None
        }

    except Exception as e:
        return {
            'R': R_value,
            'L': L_value,
            'Vin': Vin_value,
            'V_R': None,
            'V_L': None,
            'error': str(e)
        }

if __name__ == "__main__":
    param_grid = generate_full_parameter_grid()
    print(f"Lancement de {len(param_grid):,} simulations RL en régime DC...")

    start = time.time()
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap_unordered(simulate_rl_dc, param_grid), total=len(param_grid)))

    # Statistiques
    success_count = sum(1 for r in results if r['V_R'] is not None)
    fail_count = len(results) - success_count
    print(f"Simulations réussies : {success_count}")
    print(f"Simulations échouées : {fail_count}")

    # Sauvegarde
    with open("bigRl_dataset.csv", "w", newline='') as f:
        fieldnames = ['R', 'L', 'Vin', 'V_R', 'V_L']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'R': r['R'],
                'L': r['L'],
                'Vin': r['Vin'],
                'V_R': r['V_R'],
                'V_L': r['V_L'],
            })

    end = time.time()
    print(f"Données enregistrées dans 'bigRl_dataset.csv'")
    print(f"Temps total d'exécution : {end - start:.2f} secondes")
