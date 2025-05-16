import numpy as np
import pandas as pd
import multiprocessing as mp
from PySpice.Spice.Netlist import Circuit
from PySpice.Logging.Logging import setup_logging
from itertools import product
from tqdm import tqdm  # for progress bar (optional)

# Fréquences en échelle log : de 10 Hz à 10 kHz avec 100 points (≈33 par décade)
frequencies = np.logspace(1, 4, num=100)


def run_simulation(params):
    R, L, C, V_in = params
    results = []
    try:
        circuit = Circuit(f"RLC Series - R={R}, L={L}, C={C}, V={V_in}")
        circuit.SinusoidalVoltageSource('input', 'vin', circuit.gnd, amplitude=V_in)
        circuit.R(1, 'vin', 'n1', R)
        circuit.L(1, 'n1', 'n2', L)
        circuit.C(1, 'n2', circuit.gnd, C)

        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=frequencies[0],
                                stop_frequency=frequencies[-1],
                                number_of_points=len(frequencies),
                                variation='dec')

        V_in_complex = analysis['vin']
        V_n1 = analysis['n1']
        V_n2 = analysis['n2']

        for f_index, freq in enumerate(analysis.frequency):
            V_R = abs(V_in_complex[f_index] - V_n1[f_index])
            V_L = abs(V_n1[f_index] - V_n2[f_index])
            V_C = abs(V_n2[f_index])
            gain_bas = V_C / V_in
            gain_haut = V_L / V_in
            gain_bande = V_R / V_in

            results.append({
                'R': R,
                'L': L,
                'C': C,
                'V_in': V_in,
                'Frequency_Hz': float(freq),
                'V_R': V_R,
                'V_L': V_L,
                'V_C': V_C,
                'gain_bas': gain_bas,
                'gain_haut': gain_haut,
                'gain_bande': gain_bande
            })
    except Exception as e:
        return []  # Quiet failure to avoid crash
    return results


if __name__ == "__main__":
    setup_logging()

    # Frequencies (log scale): from 10 Hz to 10 kHz
    print("Nombre de fréquences dans sweep :", len(frequencies))


    # Parameter values
    R_values = np.linspace(1, 100, 10)  # 1Ω to 100Ω
    L_values = np.linspace(1e-3, 100e-3, 10)  # 1mH to 100mH
    C_values = np.linspace(100e-9, 10e-6, 10)  # 100nF to 10μF
    V_values = np.linspace(1, 10, 10)  # 1V to 10V

    # Generate parameter combinations
    combinations = list(product(R_values, L_values, C_values, V_values))
    print(f"Total combinations: {len(combinations) } (expect ~1M rows after sweep)")

    # Make sure frequencies is global for run_simulation
    #global frequencies

    with mp.Pool(processes=mp.cpu_count()) as pool:
        all_results_nested = list(tqdm(pool.imap(run_simulation, combinations), total=len(combinations)))

    # Flatten the nested results
    all_results = [row for sublist in all_results_nested for row in sublist]

    df = pd.DataFrame(all_results)

    print(df.head())
    print(f"Shape du DataFrame : {df.shape} (lignes, colonnes)")


    df.to_csv("AC_RLC_parallel.csv", index=False)
    print(f"Simulation complete! {len(df)} rows saved to 'AC_RLC_parallel.csv'")
