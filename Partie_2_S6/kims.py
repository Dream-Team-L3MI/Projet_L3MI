import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V , u_s, u_ms, u_kOhm, u_uF, u_mH, u_us


circuit = Circuit('RLC Series DC Circuit')

V_in = 10 @ u_V
R = 1 @ u_kOhm
L = 1 @ u_mH
C = 1 @ u_uF

circuit.V('input', 'in', circuit.gnd, V_in)
circuit.R(1, 'in', 'n1', R)
circuit.L(1, 'n1', 'n2', L)
circuit.C(1, 'n2', circuit.gnd, C)

# تعریف شبیه‌سازی گذرا (ترانزینت)
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.transient(step_time=1@u_us, end_time=5@u_ms)

# رسم نمودار ولتاژها
plt.figure(figsize=(10, 5))
plt.plot(analysis.time, analysis['in'], label='Vin')
plt.plot(analysis.time, analysis['n1'], label='Node n1')
plt.plot(analysis.time, analysis['n2'], label='Node n2')
plt.title('Voltage at different nodes in RLC Circuit')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.legend()
plt.grid()
plt.show()

# تبدیل داده‌ها به دیتافریم برای ذخیره‌سازی
df = pd.DataFrame({
    'time_s': np.array(analysis.time),
    'Vin': np.array(analysis['in']),
    'V_n1': np.array(analysis['n1']),
    'V_n2': np.array(analysis['n2']),
})

# ذخیره به CSV
df.to_csv('rlc_dc_simulation.csv', index=False)
