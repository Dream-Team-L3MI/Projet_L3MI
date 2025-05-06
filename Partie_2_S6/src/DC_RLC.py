import csv
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V , u_s, u_ms, u_Ohm, u_F, u_H

filename = 'DC_RLC_results.csv'
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['R', 'L', 'C', 'V_in', 'V_R', 'V_L', 'V_C'])

    for V in np.linspace(1, 10, 10):
        for R in np.linspace(1, 1000, 10):
            for L in np.linspace(1e-6, 1e-3, 10):
                for C in np.linspace(1e-9, 1e-6, 10):
                    circuit = Circuit('RLC Circuit')
                    circuit.V(1, 'input', circuit.gnd, V@u_V)
                    circuit.R(1, 'input', 'n1', R@u_Ohm)
                    circuit.L(1, 'n1', 'n2', L@u_H)
                    circuit.C(1, 'n2', circuit.gnd, C@u_F)

                    simulator = circuit.simulator()
                    analysis = simulator.operating_point()

                    try:
                        V_R = float(analysis['n1'])  
                        V_L = float(analysis['n2']) - V_R  
                        V_C = float(analysis['n2'])  
                    except:
                        V_R, V_L, V_C = None, None, None

                    writer.writerow([R, L, C, V, V_R, V_L, V_C])
