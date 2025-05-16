import matplotlib.pyplot as plt
import numpy as np
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

# Créer un circuit
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.probe import (find_probe)
import PySpice.Probe

circuit = """
* Test circuit: Simple RLC circuit
V1 N1 0 DC 10
R1 N1 N2 1k
L1 N2 N3 1mH
C1 N3 0 1uF
.tran 1ms 10ms
.end
"""

# Charger le circuit dans PySpice
from PySpice.Doc.ExampleTools import find_libraries


# Créer la simulation avec le simulateur
circuit = PySpice.Doc.ExampleTools.load_circuit(circuit)
simulator = circuit.simulator(temperature = 25, nominal_temperature = 25)

# Effectuer la simulation
analysis = simulator.transient(step_time = 0.001)

# Tracer la sortie de la simulation
plt.plot(analysis['Time'], analysis['N3'])
plt.title('Voltage at Node N3')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.grid(True)
plt.show()
