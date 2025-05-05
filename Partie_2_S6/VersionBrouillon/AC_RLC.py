import numpy as np
import pandas as pd
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V , u_s, u_ms, u_Ohm, u_uF, u_mH, u_nF, u_Hz, u_kHz
from PySpice.Probe.Plot import plot
from PySpice.Logging.Logging import setup_logging
from PySpice.Spice.Library import SpiceLibrary

logger = setup_logging()

# Frequency sweep (log scale): 10Hz to 10kHz
frequencies = np.logspace(1, 4, num=100)  # 10^1 to 10^4

# Parameter ranges
R_values = [1, 10, 50, 100]             # Ohms
L_values = [1@u_mH, 10@u_mH, 50@u_mH]   # Henry
C_values = [100@u_nF, 330@u_nF, 1@u_uF] # Farads
V_values = [3, 5, 7, 10]                # Volts

# Prepare DataFrame for output
results = []

# Sweep over all combinations
for R in R_values:
    for L in L_values:
        for C in C_values:
            for V_amp in V_values:
                circuit = Circuit(f"RLC Series - R={R}Ω, L={L}, C={C}, V={V_amp}V")

                # AC voltage source
                circuit.SinusoidalVoltageSource('input', 'in', circuit.gnd, amplitude=V_amp@u_V)

                # RLC components
                circuit.R(1, 'in', 'n1', R@u_Ohm)
                circuit.L(1, 'n1', 'n2', L)
                circuit.C(1, 'n2', circuit.gnd, C)

                # AC analysis
                simulator = circuit.simulator(temperature=25, nominal_temperature=25)
                analysis = simulator.ac(start_frequency=10@u_Hz, stop_frequency=10@u_kHz, number_of_points=100,  variation='dec')

                for f_index, freq in enumerate(np.array(analysis.frequency)):
                    V_R = abs(analysis['in'] - analysis['n1'])[f_index]
                    V_L = abs(analysis['n1'] - analysis['n2'])[f_index]
                    V_C = abs(analysis['n2'])[f_index]

                    results.append({
                        'Frequency_Hz': freq,
                        'V_R': V_R,
                        'V_L': V_L,
                        'V_C': V_C,
                        'R_Ohm': R,
                        'L_H': float(L),
                        'C_F': float(C),
                        'V_amp': V_amp
                    })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('output_rlc.csv', index=False)

print("Simulation complete! Results saved to 'output_rlc.csv'")


