from PySpice.Spice.Netlist import Circuit
import csv
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import time

# Fonction de simulation AC pour un circuit RC
def simulate_rc_ac(params):
    R_value, C_value, Vin = params

    try:
        # Création du circuit
        circuit = Circuit('RC Circuit AC Analysis')

        # Ajout des composants avec valeur DC explicite pour compatibilité ngspice
        circuit.V('input', 'input', circuit.gnd, f'DC 0 AC {float(Vin)}')  # 'DC 0 AC {}'.format(Vin)
        circuit.R('R1', 'input', 'output', R_value)
        circuit.C('C1', 'output', circuit.gnd, C_value)

        # Configuration du simulateur
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)

        # Analyse AC compatible NgSpice
        analysis = simulator.ac(start_frequency=1,
                                stop_frequency=1e6,
                                number_of_points=100,
                                variation='dec')
        

        v_in = analysis[ 'input' ]
        v_out = analysis[ 'output' ]
        # Calcul de la tension aux bornes du condensateur
        v_r = v_in - v_out

        # Extraction des données
        frequencies = np.array( [ float( f ) for f in analysis.frequency ] )
        magnitudes = np.array( [ float( m ) for m in abs( analysis[ 'output' ] ) ] )
        mag_r = np.abs( v_r)
        mag_out = abs( analysis[ 'output' ] )
        #mag_r = np.array( [ float( m ) for m in abs( analysis[ 'R' ] ) ] )
        #mag_out = np.array( [ float( m ) for m in abs( analysis[ 'output' ] ) ] )

        phases = np.array( [ float(p) for p in np.angle( analysis[ 'output' ] , deg = True ) ] )


        #phases_r = np.array( [ float( p ) for p in np.angle( analysis[ 'R' ] , deg = True ) ] )
        phases_r = np.angle( v_r, deg = True )
        phases_out = np.array( [ float( p ) for p in np.angle( analysis[ 'output' ], deg = True ) ] )

        # Calcul du gain basse
        gain_basse = np.abs( v_out ) / np.abs( v_in )
        gain_basse_db = 20 * np.log10( gain_basse )

        # calcul du gain hautes
        gain_haute = np.abs( v_r ) / np.abs( v_in )
        gain_gain_haute_db = 20 * np.log10( gain_haute )


        # Création des résultats
        results = []
        i = 0
        for f, mag, phase in zip(frequencies, magnitudes, phases):
            results.append({
                'R': R_value,
                'C': C_value,
                'Vin': Vin,
                'Frequency': f,
                'V_R': float( mag_r[ i ] ),
                'V_C': float( mag_out[ i ] ) ,
                'Phase_R': phases_r[ i ],
                'Phase_C': phases_out[ i ],
                'Gain_basse': float( gain_basse[ i ] ),
                'Gain_basse_dB': gain_basse_db[ i ] ,
                'Gain_haute': float( gain_haute[ i ] ),
                'Gain_haute_dB': gain_gain_haute_db[ i ]
            })
            i += 1

        return results

    except Exception as e:
        print(f"Erreur pour (R={R_value}, C={C_value}, Vin={Vin}): {e}")
        return []

# Génération des combinaisons paramétriques
def generate_dataset(num_simulations):
    r_values = np.logspace(3, 4, 5)          # 1kΩ à 10kΩ
    c_values = np.logspace(-8, -6, 5)        # 10nF à 1µF
    vin_values = np.linspace(0.5, 2.0, 5)    # 0.5V à 2V

    # Création du param_grid
    full_grid = [(r, c, v) for r in r_values for c in c_values for v in vin_values]
    return full_grid[:num_simulations]

# Écriture progressive des résultats dans le fichier CSV
def write_batch_to_csv(batch_results, writer):
    for result in batch_results:
        writer.writerows(result)  # Chaque "result" est une liste de 100 lignes

if __name__ == '__main__':
    # Paramètres globaux
    TOTAL_SIMULATIONS = 1_000_000
    BATCH_SIZE = 1000
    PROCESSES = 4  # à ajuster selon ta machine (4 à 6 recommandé)

    print(f"Lancement de {TOTAL_SIMULATIONS} simulations AC en batchs de {BATCH_SIZE}...")

    # Création du dataset
    dataset = generate_dataset(TOTAL_SIMULATIONS)
    total_batches = TOTAL_SIMULATIONS // BATCH_SIZE

    start = time.time()

    # Sauvegarde des résultats dans un fichier CSV
    with open('rc_ac_results.csv', 'w', newline='') as f:
        #writer = csv.DictWriter(f, fieldnames=['R', 'C', 'Vin', 'Frequency', 'Magnitude', 'Phase'])
        writer = csv.DictWriter(f, fieldnames=[
                                'R', 'C', 'Vin', 'Frequency',
                                'V_R', 'V_C', 'Phase_R', 'Phase_C' , 
                                'Gain_basse', 'Gain_basse_dB',
                                'Gain_haute', 'Gain_haute_dB'

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
                batch_results = pool.map(simulate_rc_ac, batch)

            # Écriture progressive dans le fichier CSV
            write_batch_to_csv(batch_results, writer)

    end = time.time()
    print("Simulations AC terminées. Données sauvegardées dans rc_ac_results.csv")
    print(f"Temps total d'exécution : {end - start:.2f} secondes")