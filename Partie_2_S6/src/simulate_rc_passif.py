from PySpice.Spice.Parser import SpiceParser
import matplotlib.pyplot as plt

# Chemin vers la netlist exportée
parser = SpiceParser('rc_passif.cir')
circuit = parser.build_circuit()
simulator = circuit.simulator(temperature=25, nominal_temperature=25)

# Simulation transitoire (attention : end_time = 30ms comme dans ta netlist)
analysis = simulator.transient(step_time=1e-6, end_time=30e-3)

# Affiche les noms de nœuds disponibles pour debug
print("Nœuds disponibles :", list(analysis.nodes.keys()))

# Affiche la tension de sortie (v(_out)) — selon renaming automatique de PySpice
plt.plot(analysis.time, analysis.nodes['/in'], label='Entrée (/in)')
plt.plot(analysis.time, analysis.nodes['/out'], label='Sortie (/out)')

plt.title('Tension en sortie du circuit RC')
plt.xlabel('Temps [s]')
plt.ylabel('Tension [V]')
plt.grid(True)
plt.legend()
plt.show()
