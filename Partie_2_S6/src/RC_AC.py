import csv
import numpy as np
from itertools import product
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_F, u_Hz
import multiprocessing
from tqdm import tqdm

#dans le cas d'un regime AC on doit plutot utiliser une source sinusoidale 
#analyse en frequence 
#les amplitudes et les phases des tensions à une frq donnée 

#######################################################################
#Les etapes :
#Une fonction pour générer tous les paramètres
#Une fonction de simulation DC bien conçue avec gestion des erreurs
#Les calculs pour tensions aux bornes des composants
######################################################################

# ─────────────────────────────────────────────────────────────────────────────
# 1) Génération de la grille de paramètres
# ─────────────────────────────────────────────────────────────────────────────
def generate_full_parameter_grid(r_count=100, c_count=100, vin_count=100):
    """
    Retourne la liste de tous les tuples (R, C, Vin) formant
    un produit cartésien de tailles r_count × c_count × vin_count.
    """
    r_vals   = np.logspace(1, 3, r_count)     # de 10 Ω à 1000 Ω (logarithmique)
    c_vals   = np.logspace(-9, -5, c_count)   # de 1 nF à 10 µF (logarithmique)
    vin_vals = np.logspace(0.1, 2, vin_count)  # de 0.1 V à 100 V (logarithmique)
    return list(product(r_vals, c_vals, vin_vals))

# ─────────────────────────────────────────────────────────────────────────────
# 2) Fonction de simulation AC pour un circuit RC
# ─────────────────────────────────────────────────────────────────────────────
def simulate_rc_ac(params):
    """
    Simule un circuit RC en régime AC pour les valeurs données de
    R (Ohm), C (F) et Vin (V). Retourne un dict de résultats.
    """
    R_value, C_value, Vin = params
    try:
        # Construction du circuit
        circuit = Circuit(f'RC R={R_value:.1f}Ω C={C_value:.1e}F Vin={Vin:.1f}V')
        circuit.V(1, 'vin', circuit.gnd, Vin@u_V)
        circuit.R(1, 'vin', 'vout', R_value@u_Ohm)
        circuit.C(1, 'vout', circuit.gnd, C_value@u_F)

        # Simulation AC
        sim = circuit.simulator()
        analysis = sim.ac(start_frequency=1e3@u_Hz,  # Fréquence de départ (1 kHz)
                          stop_frequency=1e6@u_Hz,  # Fréquence d'arrêt (1 MHz)
                          number_of_points=100,       # 100 points sur la plage
                          variation='dec')            # Variation logarithmique

        # Récupérer les résultats de l'analyse
        #pour le gain le vout est comme une valeur complexe cest l'amplitude + phase du signal de sortie 
        #donc c'est le rapoort d'amplitude entre la sortie et l'entrée 
        gain = abs(analysis['vout'] / analysis['vin'])  # Gain |Vout/Vin|
        phase = np.angle(analysis['vout'], deg=True) - np.angle(analysis['vin'], deg=True)  # Phase (en degrés)

        return {
            'R':    R_value,
            'C':    C_value,
            'Vin':  Vin,
            'gain': gain[0],   # Gain à la première fréquence (par exemple)
            'phase_deg': phase[0],  # Phase à la première fréquence
            'error': ''
        }

    except Exception as e:
        # En cas d'erreur
        return {
            'R':    R_value,
            'C':    C_value,
            'Vin':  Vin,
            'gain': None,
            'phase_deg': None,
            'error': str(e)
        }

# ─────────────────────────────────────────────────────────────────────────────
# 3) Point d'entrée principal : lancement des simulations
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 3.1) Générer la grille complète (100×100×100 = 1 000 000 combinaisons)
    param_grid = generate_full_parameter_grid(100, 100, 100)
    total = len(param_grid)
    print(f" Lancement de {total:,} simulations RC en parallèle...")

    output_file = "rc_dataset_1M_ac.csv"
    fieldnames = ['R', 'C', 'Vin', 'gain', 'phase_deg', 'error']

    # 3.2) Ouvrir le CSV et écrire l'en-tête
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 3.3) Créer la pool de processus
        with multiprocessing.Pool() as pool:
            # Utilisation de imap_unordered pour écrire les résultats dès qu'ils sont prêts
            for result in tqdm(pool.imap_unordered(simulate_rc_ac, param_grid, chunksize=100),
                               total=total, desc="Simulations AC"):
                writer.writerow(result)

    print(f" 1 000 000 simulations terminées. Données enregistrées dans '{output_file}'.")
