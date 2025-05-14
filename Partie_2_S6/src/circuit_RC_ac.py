from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_F, u_Hz
import csv
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def simulate_rc_ac(params):
    R_value, C_value, Vin = params  # Décompacte le tuple

    # Création du circuit RC
    circuit = Circuit('RC AC Analysis')
    circuit.SinusoidalVoltageSource(1, 'vin', circuit.gnd, amplitude=Vin @ u_V, frequency=1 @ u_Hz)

    circuit.R(1, 'vin', 'vout', R_value @ u_Ohm)
    circuit.C(1, 'vout', circuit.gnd, C_value @ u_F)

    # Simulation AC
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.ac(start_frequency=1 @ u_Hz, stop_frequency=1e6 @ u_Hz, number_of_points=100, variation='dec')

    # Traitement des résultats
    freqs = np.array([float(f) for f in analysis.frequency])
    vin_complex = np.array([complex(analysis['vin'][i]) for i in range(len(freqs))])
    vout_complex = np.array([complex(analysis['vout'][i]) for i in range(len(freqs))])

    vin_mag = np.abs(vin_complex)
    vout_mag = np.abs(vout_complex)
    gain_db = 20 * np.log10(vout_mag / vin_mag)
    # Ancienne ligne :
    # phase_deg = np.angle(vout_complex / vin_complex, deg=True)

    # Nouvelle version correcte :
    phase_vin = np.angle(vin_complex, deg=True)
    phase_vout = np.angle(vout_complex, deg=True)
    phase_deg = phase_vout - phase_vin


    v_r_mag = np.abs(vin_complex - vout_complex)
    v_c_mag = np.abs(vout_complex)

    # Construction des résultats pour chaque fréquence
    result = []
    for f, vo, vr, vc, g, p in zip(freqs, vout_mag, v_r_mag, v_c_mag, gain_db, phase_deg):
        result.append({
            'R': R_value,
            'C': C_value,
            'Vin': Vin,
            'Frequency': f,
            'Vout': vo,
            'V_R': vr,
            'V_C': vc,
            'Gain_dB': g,
            'Phase_deg': p
        })

    return result


if __name__ == '__main__':
    # Définition des plages de paramètres
    r_values = np.linspace(1e3, 10e3, 10)         # 10 valeurs de résistance
    c_values = np.linspace(1e-8, 1e-6, 10)        # 10 valeurs de capacité
    vin_values = np.linspace(0.5, 2.0, 5)         # 5 valeurs de Vin

    # Création de la liste des combinaisons (500 au total)
    dataset = [(R, C, Vin) for R in r_values for C in c_values for Vin in vin_values]

    print(f"Lancement de {len(dataset)} simulations en parallèle...")

    # Simulation parallèle avec barre de progression
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(simulate_rc_ac, dataset), total=len(dataset)))

    # Aplatir la liste des résultats
    flat_dataset = [item for sublist in results for item in sublist]

    # Écriture dans le fichier CSV
    with open("rc_dataset_ac_complet.csv", "w", newline='') as file:
        fieldnames = ['R', 'C', 'Vin', 'Frequency', 'Vout', 'V_R', 'V_C', 'Gain_dB', 'Phase_deg']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_dataset)

    print("✅ Simulations AC terminées. Données sauvegardées dans rc_dataset_ac_complet.csv")
