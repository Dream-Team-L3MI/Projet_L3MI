import numpy as np
import csv
from PySpice.Spice.Netlist import Circuit
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Simulation import ACAnalysis
from PySpice.Unit import *
from PySpice.Logging.Logging import setup_logging
from PySpice.Unit import u_V , u_s, u_ms, u_Ohm, u_F, u_Hz, u_H

setup_logging()

frequencies = np.logspace(1, 6, num=100)  # فرکانس از 10Hz تا 1MHz
results = []

for R in [1, 10, 100, 100]:  # مقاومت‌ها از 1 تا 1000 اهم
    for C in [1e-3, 1e-1, 1]:  # ظرفیت از 1 میکروفاراد تا 1 فاراد
        for V in [1, 5, 10]:   # ولتاژ ورودی از 1 تا 10 ولت

            circuit = Circuit('AC Analysis of RLC Circuit')
            circuit.SinusoidalVoltageSource('input', 'vin', circuit.gnd, amplitude=V@u_V)
            circuit.R(1, 'vin', 'n1', R@u_Ohm)
            circuit.L(1, 'n1', 'n2', 1@u_H)
            circuit.C(1, 'n2', circuit.gnd, C@u_F)

            simulator = circuit.simulator(temperature=25, nominal_temperature=25)
            analysis = simulator.ac(start_frequency=frequencies[0]@u_Hz,
                                     stop_frequency=frequencies[-1]@u_Hz,
                                     number_of_points=len(frequencies),
                                     variation='dec')

            for f, v in zip(analysis.frequency, analysis['vin']):
                results.append([float(f), R, C, V, abs(v)])

# نوشتن در فایل CSV
with open('ac_rlc_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frequency (Hz)', 'R (Ohm)', 'C (F)', 'Vin (V)', 'Magnitude of Vin'])
    writer.writerows(results)

