from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V, u_Ohm

def simple_simulation():
    circuit = Circuit('Simple Circuit')
    circuit.V(1, 'input', circuit.gnd, 5 @ u_V)
    circuit.R(1, 'input', 'output', 1e3 @ u_Ohm)
    simulator = circuit.simulator()
    analysis = simulator.operating_point()
    return analysis.nodes

print(simple_simulation())
