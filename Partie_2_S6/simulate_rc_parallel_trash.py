# simulate_rc_parallel.py

import csv
import numpy as np
from itertools import product
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_F
import multiprocessing
from tqdm import tqdm

# simulate_rc_parallel.py

import csv
import numpy as np
from itertools import product
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_F
import multiprocessing
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 1) Génération de la grille de paramètres
# ─────────────────────────────────────────────────────────────────────────────
def generate_full_parameter_grid(r_count=1000, c_count=1000, vin_count=100):
    """
    Retourne la liste de tous les tuples (R, C, Vin) formant
    un produit cartésien de tailles r_count × c_count × vin_count.
    """
    r_vals   = np.linspace(1e2, 1e5, r_count)     # de 100 Ω à 100 kΩ
    c_vals   = np.linspace(1e-9, 1e-5, c_count)   # de 1 nF à 10 µF
    vin_vals = np.linspace(0.1, 10.0, vin_count)  # de 0.1 V à 10 V
    return list(product(r_vals, c_vals, vin_vals))

# ─────────────────────────────────────────────────────────────────────────────
# 2) Fonction de simulation DC pour un circuit RC
# ─────────────────────────────────────────────────────────────────────────────
def simulate_rc_dc(params):
    """
    Simule un circuit RC en régime DC pour les valeurs données de
    R (Ohm), C (F) et Vin (V). Retourne un dict de résultats.
    """
    R_value, C_value, Vin = params
    try:
        # Construction du circuit
        circuit = Circuit(f'RC R={R_value:.1f}Ω C={C_value:.1e}F Vin={Vin:.1f}V')
        circuit.V(1, 'vin', circuit.gnd, Vin @ u_V)
        circuit.R(1, 'vin', 'vout', R_value @ u_Ohm)
        circuit.C(1, 'vout', circuit.gnd, C_value @ u_F)
        sim      = circuit.simulator()
        analysis = sim.operating_point()

        # Lecture des tensions
        V_in  = float(analysis.nodes['vin'])
        V_out = float(analysis.nodes['vout'])
        V_R   = V_in - V_out      # tension aux bornes de la résistance
        V_C   = V_out             # tension aux bornes du condensateur (vs gnd)

        return {
            'R':    R_value,
            'C':    C_value,
            'Vin':  Vin,
            'V_in':  V_in,
            'V_out': V_out,
            'V_R':   V_R,
            'V_C':   V_C,
            'error': ''
        }

    except Exception as e:
        # En cas d'erreur de convergence ou autre
        return {
            'R':    R_value,
            'C':    C_value,
            'Vin':  Vin,
            'V_in':  None,
            'V_out': None,
            'V_R':   None,
            'V_C':   None,
            'error': str(e)
        }

# ─────────────────────────────────────────────────────────────────────────────
# 3) Point d'entrée principal : lancement des simulations
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 3.1) Générer la grille complète (100×100×100 = 1 000 000 combinaisons)
    param_grid = generate_full_parameter_grid(100, 100, 100)
    total = len(param_grid)
    print(f"🔄 Lancement de {total:,} simulations RC en parallèle...")

    output_file = "TRASH_rc_dataset_1M_2.csv"
    fieldnames = ['R', 'C', 'Vin', 'V_in', 'V_out', 'V_R', 'V_C', 'error']

    # 3.2) Ouvrir le CSV et écrire l'en-tête
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 3.3) Créer la pool de processus
        with multiprocessing.Pool() as pool:
            # Utilisation de imap_unordered pour écrire les résultats dès qu'ils sont prêts
            for result in tqdm(pool.imap_unordered(simulate_rc_dc, param_grid, chunksize=100),
                               total=total, desc="Simulations"):
                writer.writerow(result)

    print(f"✅ 1 000 000 simulations terminées. Données enregistrées dans '{output_file}'.")
