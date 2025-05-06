import numpy as np
import pandas as pd
import multiprocessing
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_H, u_F, u_Ohm, u_Hz
from PySpice.Logging.Logging import setup_logging

logger = setup_logging()

# Parameter bounds
R_bounds = (1, 1000)             # Ohms
L_bounds = (0.001, 0.1)          # Henries
C_bounds = (1e-9, 1e-6)          # Farads
V_bounds = (1, 10)               # Volts
f_bounds = (10, 10000)           # Hz

def simulate_batch(batch_combos):
    results = []
    for R, L, C, V_in, freq in batch_combos:
        circuit = Circuit("RLC Series")
        circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd,
                                        amplitude=V_in @ u_V,
                                        frequency=freq @ u_Hz)
        circuit.R(1, 'in', 'n1', R @ u_Ohm)
        circuit.L(1, 'n1', 'n2', L @ u_H)
        circuit.C(1, 'n2', circuit.gnd, C @ u_F)

        simulator = circuit.simulator()
        try:
            analysis = simulator.ac(start_frequency=freq @ u_Hz,
                                    stop_frequency=freq @ u_Hz,
                                    number_of_points=1, variation='dec')
            V_R = abs(analysis['in'] - analysis['n1'])[0]
            V_L = abs(analysis['n1'] - analysis['n2'])[0]
            V_C = abs(analysis['n2'])[0]

            gain_bas = V_C / V_in
            gain_haut = V_L / V_in
            gain_bande = V_R / V_in

            results.append([R, L, C, V_in, V_R, V_L, V_C, freq,
                            gain_bas, gain_haut, gain_bande])
        except Exception:
            continue
    return results

def run_simulation(n_total=1_000_000, n_processes=4):
    np.random.seed(42)
    R_vals = np.random.uniform(*R_bounds, n_total)
    L_vals = np.random.uniform(*L_bounds, n_total)
    C_vals = np.random.uniform(*C_bounds, n_total)
    V_vals = np.random.uniform(*V_bounds, n_total)
    F_vals = np.random.uniform(*f_bounds, n_total)

    combos = list(zip(R_vals, L_vals, C_vals, V_vals, F_vals))

    # Divide into chunks
    chunk_size = n_total // n_processes
    chunks = [combos[i * chunk_size: (i + 1) * chunk_size if i != n_processes - 1 else n_total]
              for i in range(n_processes)]

    with multiprocessing.Pool(n_processes) as pool:
        all_results = pool.map(simulate_batch, chunks)

    flat_results = [row for batch in all_results for row in batch]
    df = pd.DataFrame(flat_results, columns=[
        'R', 'L', 'C', 'V_in', 'V_R', 'V_L', 'V_C', 'frequency',
        'gain_bas', 'gain_haut', 'gain_bande'
    ])
    df.to_csv('AC_RLC_ML_dataset.csv', index=False)
    print("✅ Simulation complete. Results saved to 'AC_RLC_ML_dataset.csv'.")

if __name__ == '__main__':
    run_simulation()


