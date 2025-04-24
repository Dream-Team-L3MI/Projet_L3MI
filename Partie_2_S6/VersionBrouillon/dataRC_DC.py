import csv
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V , u_s, u_ms, u_Ohm, u_uF, u_F

def simulate_rc_dc(R_value, C_value, Vin=5.0):
    """
    Simulates an RC circuit in DC regime (steady-state) using PySpice.
    Returns the voltage at input and output nodes.
    """
    circuit = Circuit(f'RC Circuit R={R_value} C={C_value}')
    circuit.V(1, 'in', circuit.gnd, Vin @ u_V)
    circuit.R(1, 'in', 'out', R_value @ u_Ohm)
    circuit.C(1, 'out', circuit.gnd, C_value @ u_F)

    simulator = circuit.simulator()
    analysis = simulator.operating_point()  # DC operating point

    V_in = float(analysis.nodes['in'])
    V_out = float(analysis.nodes['out'])

    return {'R': R_value, 'C': C_value, 'V_in': V_in, 'V_out': V_out}

# ----------------------------
# Example usage
# ----------------------------

# Define R and C ranges
r_values = [1e3, 2e3, 5e3]       # Ohms
c_values = [1e-6, 2e-6, 5e-6]    # Farads

# Generate dataset
dataset = []

for R in r_values:
    for C in c_values:
        result = simulate_rc_dc(R, C)
        dataset.append(result)

# Save results to CSV
with open("rc_dataset_dc.csv", "w", newline='') as f:
    fieldnames = ['R', 'C', 'V_in', 'V_out']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(dataset)

print(" Simulation complete. Results saved to rc_dataset_dc.csv")
