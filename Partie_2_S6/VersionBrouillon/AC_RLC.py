from PySpice.Spice.Netlist import Circuit
import csv
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import time

# Fonction de simulation AC pour un circuit RC

# Fonction de simulation AC pour un circuit RLC
def simulate_rlc_ac(params):
    R_value, L_value, C_value, Vin = params

    try:
        circuit = Circuit('RLC Circuit AC Analysis')

        circuit.V('input', 'input', circuit.gnd, f'DC 0 AC {float(Vin)}')
        circuit.R('R1', 'input', 'n1', R_value)
        circuit.L('L1', 'n1', 'n2', L_value)
        circuit.C('C1', 'n2', circuit.gnd, C_value)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)

        analysis = simulator.ac(start_frequency=1,
                                stop_frequency=1e6,
                                number_of_points=100,
                                variation='lin')

        v_in = analysis['input']
        v_out = analysis['n2']
        v_r = analysis['input'] - analysis['n1']
        v_l = analysis['n1'] - analysis['n2']

        frequencies = np.array([float(f) for f in analysis.frequency])
        mag_out = np.abs(v_out)
        mag_r = np.abs(v_r)
        mag_l = np.abs(v_l)

        phase_out = np.angle(v_out, deg=True)
        phase_r = np.angle(v_r, deg=True)
        phase_l = np.angle(v_l, deg=True)

        gain_basse = mag_out / np.abs(v_in)
        gain_basse_db = 20 * np.log10(gain_basse)
        gain_haute = mag_r / np.abs(v_in)
        gain_haute_db = 20 * np.log10(gain_haute)
        gain_bande = mag_out / np.abs(v_in)
        gain_bande_db = 20 * np.log10(gain_bande)


        results = []
        for i, f in enumerate(frequencies):
            results.append({
                'R': R_value,
                'L': L_value,
                'C': C_value,
                'Vin': Vin,
                'Frequency': f,
                'V_R': float(mag_r[i]),
                'V_L': float(mag_l[i]),
                'V_C': float(mag_out[i]),
                'Phase_R': phase_r[i],
                'Phase_L': phase_l[i],
                'Phase_C': phase_out[i],
                'Gain_basse': float(gain_basse[i]),
                'Gain_basse_dB': gain_basse_db[i],
                'Gain_haute': float(gain_haute[i]),
                'Gain_haute_dB': gain_haute_db[i],
                'Gain_bande': float(gain_bande[i]),
                'Gain_bande_dB': gain_bande_db[i],
            })

        return results

    except Exception as e:
        print(f"Erreur pour (R={R_value}, L={L_value}, C={C_value}, Vin={Vin}): {e}")
        return []



# Génération des combinaisons paramétriques
def generate_dataset(num_simulations):
    r_values = np.logspace(3, 4, 10)
    l_values = np.logspace(-6, -3, 10)  # 1µH à 1mH
    c_values = np.logspace(-8, -6, 10)
    vin_values = np.linspace(0.5, 2.0, 10)

    full_grid = [(r, l, c, v) for r in r_values for l in l_values for c in c_values for v in vin_values]
    return full_grid[:num_simulations]


# Écriture progressive des résultats dans le fichier CSV
def write_batch_to_csv(batch_results, writer):
    for result in batch_results:
        writer.writerows(result)  # Chaque "result" est une liste de 100 lignes

if __name__ == '__main__':
    # Paramètres globaux
    TOTAL_SIMULATIONS = 10000
    BATCH_SIZE = 100
    PROCESSES = 4  

    print(f"Lancement de {TOTAL_SIMULATIONS} simulations AC en batchs de {BATCH_SIZE}...")

    # Création du dataset
    dataset = generate_dataset(TOTAL_SIMULATIONS)
    total_batches = TOTAL_SIMULATIONS // BATCH_SIZE

    start = time.time()

    # Sauvegarde des résultats dans un fichier CSV
    with open('rc_ac_results.csv', 'w', newline='') as f:
        #writer = csv.DictWriter(f, fieldnames=['R', 'C', 'Vin', 'Frequency', 'Magnitude', 'Phase'])
        writer = csv.DictWriter(f, fieldnames=[
            'R', 'L', 'C', 'Vin', 'Frequency',
            'V_R', 'V_L', 'V_C',
            'Phase_R', 'Phase_L', 'Phase_C',
            'Gain_basse', 'Gain_basse_dB',
            'Gain_haute', 'Gain_haute_dB', 'Gain_bande','Gain_bande_dB'
])


        writer.writeheader()

        # Exécution batch par batch
        for batch_index in tqdm(range(total_batches)):
            batch = dataset[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]

            # Création d'une pool de processus pour le batch actuel
            # l'utilisation de with va assurer la fermeture de notre pool après l'exécution 
            # la méthode pool.map va assurer l'application de la ft de simulation sur chaque élément du param_grid 
            # de façon parallèle (càd elle répartit les valeurs de la grille sur les différents ps de la pool )
            # => Ce qui réduit le temps global nécessaire pour l'exécution
            with Pool(processes=PROCESSES) as pool:
                batch_results = pool.map(simulate_rlc_ac, batch)

            # Écriture progressive dans le fichier CSV
            write_batch_to_csv(batch_results, writer)

    end = time.time()
    print("Simulations AC terminées. Données sauvegardées dans rc_ac_results.csv")
    print(f"Temps total d'exécution : {end - start:.2f} secondes")


