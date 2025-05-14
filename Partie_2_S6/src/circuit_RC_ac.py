from PySpice.Spice.Netlist import Circuit
import csv
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def simulate_rc_ac(params):
    R_value, C_value, Vin = params

    # Création du circuit
    circuit = Circuit('RC Circuit AC Analysis')

    # Ajout des composants avec valeur DC explicite pour compatibilité ngspice
    circuit.V('input', 'input', circuit.gnd, 'DC 0 AC {}'.format(Vin))
    circuit.R('R1', 'input', 'output', R_value)
    circuit.C('C1', 'output', circuit.gnd, C_value)

    # Configuration du simulateur
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # Analyse AC compatible NgSpice v44
    analysis = simulator.ac(start_frequency=1,
                            stop_frequency=1e6,
                            number_of_points=100,
                            variation='lin')

    # Extraction des données
    frequencies = np.array([float(f) for f in analysis.frequency])
    magnitudes = np.array([float(m) for m in abs(analysis['output'])])
    phases = np.array([float(p) for p in np.angle(analysis['output'], deg=True)])

    # Création des résultats
    results = []
    for f, mag, phase in zip(frequencies, magnitudes, phases):
        results.append({
            'R': R_value,
            'C': C_value,
            'Vin': Vin,
            'Frequency': f,
            'Magnitude': mag,
            'Phase': phase
        })

    return results

if __name__ == '__main__':
    # Paramètres de simulation
    r_values = np.logspace(3, 4, 10)  # 1kΩ à 10kΩ
    c_values = np.logspace(-8, -6, 10)  # 10nF à 1µF
    vin_values = np.linspace(0.5, 2.0, 100)  # 0.5V à 2V

    # Création du dataset
    dataset = [(r, c, v) for r in r_values for c in c_values for v in vin_values]
    print(f"Lancement de {len(dataset)} simulations en parallèle...")

    # Simulation parallèle
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(simulate_rc_ac, dataset), total=len(dataset)))

    # Aplatissement des résultats
    flat_results = [item for sublist in results for item in sublist]

    # Sauvegarde des résultats
    with open('rc_ac_results_1M.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['R', 'C', 'Vin', 'Frequency', 'Magnitude', 'Phase'])
        writer.writeheader()
        writer.writerows(flat_results)

    print("✅ Simulations AC terminées. Données sauvegardées dans rc_ac_results_1M.csv")
