import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_H, u_F, u_s
from PySpice.Probe.Plot import plot
import numpy as np
import csv
from tqdm import tqdm
from itertools import product
import signal
import sys

# === Handle Keyboard Interrupt ===
def signal_handler(sig, frame):
    print('\nInterrupt received. Saving data...')
    save_data(dataset)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === CSV Saving Function ===
def save_data(data):
    with open("rlc_transit.csv", "w", newline='') as f:
        fieldnames = ['id', 'R', 'L', 'C', 'Time_s', 'Vin_V', 'Vout_V', 'I_A', 'tau', 'Time_norm']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} rows to rlc_transit.csv")

# === RLC Transient Simulation ===
def simulate_rlc_transient(R_value, L_value, C_value, Vin_value, sim_id):
    circuit = Circuit(f"RLC Transient R={R_value} L={L_value} C={C_value}")

    circuit.PulseVoltageSource('input', 'vin', circuit.gnd,
                               initial_value=0@u_V,
                               pulsed_value=Vin_value @ u_V,
                               pulse_width=10@u_s,
                               period=20@u_s)

    circuit.R(1, 'vin', 'n1', R_value @ u_Ohm)
    circuit.L(1, 'n1', 'n2', L_value @ u_H)
    circuit.C(1, 'n2', circuit.gnd, C_value @ u_F)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    tau = L_value / R_value  # Still useful for time normalization
    stop_time = 5 * tau  # Longer to capture oscillations
    step_time = tau / 200

    try:
        analysis = simulator.transient(step_time=step_time @ u_s, end_time=stop_time @ u_s)

        times = analysis.time
        vin = analysis['vin']
        vout = analysis['n2']  # across capacitor
        iL = analysis['L1']

        results = []
        for t, v_in, v_out, i in zip(times, vin, vout, iL):
            results.append({
                'id': sim_id,
                'R': float(R_value),
                'L': float(L_value),
                'C': float(C_value),
                'Time_s': float(t),
                'Vin_V': float(v_in),
                'Vout_V': float(v_out),
                'I_A': float(i),
                'tau': float(tau),
                'Time_norm': float(t) / float(tau)
            })
        return results

    except Exception as e:
        print(f"Simulation error for R={R_value}, L={L_value}, C={C_value} : {str(e)}")
        return None

# === Parameter Grid ===
vin_values = np.round(np.linspace(1, 10, 2), 2)
r_values = np.round(np.geomspace(10, 1000, 20), 2)
l_values = np.round(np.geomspace(1e-4, 1e-2, 25), 6)
c_values = np.round(np.geomspace(1e-8, 1e-5, 20), 9)

param_combinations = list(product(r_values, l_values, c_values, vin_values))
total_combinations = len(param_combinations)
print(f"Total combinations: {total_combinations}")
print(f"Estimated total rows: ~{total_combinations * 500}")

dataset = []
sim_id = 0

# === Run Simulations ===
with tqdm(total=total_combinations, desc="RLC Simulation Progress") as pbar:
    for R, L, C, Vin in param_combinations:
        result = simulate_rlc_transient(R, L, C, Vin, sim_id)
        if result is not None:
            dataset.extend(result)
            sim_id += 1
        pbar.update(1)

        if len(dataset) > 0 and pbar.n % 100 == 0:
            save_data(dataset)

# === Final Save ===
save_data(dataset)
print("\nSimulation complete. Data saved to rlc_transit.csv")
