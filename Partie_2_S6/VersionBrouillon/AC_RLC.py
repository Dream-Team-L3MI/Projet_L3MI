import csv
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm, u_F, u_H, u_Hz

filename = 'AC_RLC_results.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['R', 'L', 'C', 'V_in', 'V_R', 'V_L', 'V_C', 'temps', 'frequence', 'gain'])

    for V in np.linspace(1, 10, 10):  # V_in amplitude
        for R in np.linspace(1, 1000, 10):
            for L in np.linspace(1e-6, 1e-3, 10):
                for C in np.linspace(1e-9, 1e-6, 10):
                    circuit = Circuit('RLC Circuit AC')
                    circuit.SinusoidalVoltageSource(1, 'input', circuit.gnd, amplitude=V@u_V)
                    circuit.R(1, 'input', 'n1', R@u_Ohm)
                    circuit.L(1, 'n1', 'n2', L@u_H)
                    circuit.C(1, 'n2', circuit.gnd, C@u_F)

                    simulator = circuit.simulator()
                    try:
                        # Run AC analysis with frequency sweep
                        analysis = simulator.ac(start_frequency=1@u_Hz,
                                                 stop_frequency=1e6@u_Hz,
                                                 number_of_points=100,
                                                 variation='log')
                        
                        # We are interested in values at the first frequency point
                        frequence = float(analysis.frequency[0])  # frequency
                        V_in = V  # input voltage
                        V_R = abs(analysis['input'][0] - analysis['n1'][0])  # Voltage across resistor
                        V_L = abs(analysis['n1'][0] - analysis['n2'][0])  # Voltage across inductor
                        V_C = abs(analysis['n2'][0])  # Voltage across capacitor

                        # Gain calculation
                        gain = 20 * np.log10(V_C / V_in) if V_in != 0 else None

                        # Calculating the time period (temps) from the frequency
                        temps = 1 / frequence if frequence != 0 else None

                        # Write values to CSV
                        writer.writerow([R, L, C, V_in, V_R, V_L, V_C, temps, frequence, gain])
                    except Exception as e:
                        # In case of error, write None values
                        writer.writerow([R, L, C, V, None, None, None, None, None, None])
