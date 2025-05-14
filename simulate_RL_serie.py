#On s'interesse dans ce code à un Vout qui est la tension aux bornes de L ou eventuellement le courant dans le circuit


import numpy as np
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit

def simulate_rl_circuit(R_value, L_value, V_value, step_time, end_time):
    circuit = Circuit('Circuit RL simple')
    #Source de tension continue
    circuit.V('1', 'input', circuit.gnd, f'DC {V_value}V')

    #La résistance R entre le input et le output
    circuit.R('1', 'input', 'output', R_value)

    #L'inductance L entre output et la masse
    circuit.L('1', 'output', circuit.gnd, L_value)

    # Simulateur
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # Condition initiale (facultative ici mais on la met pour être propre)
    simulator.initial_condition(output=0)

    # Simulation transitoire
    analysis = simulator.transient(step_time=step_time, end_time=end_time)

    # Récupération de la tension sur le noeud "output"
    v_out = analysis['output']

#Main : programme principal :

if __name__ == '__main__':
    # Valeurs de R de 100 à 1000 Ohm, par pas de 300
    R_values = np.arange(100, 1001, 300)
    # Valeurs de L de 1mH à 10mH par pas de 3mH
    L_values = np.arange(1e-3, 10.1e-3, 3e-3)

    voltage = 1
    step_time = 1e-3
    end_time = 50e-3

    plt.figure()

    for R in R_values:
        for L in L_values:
            time, v_out = simulate_rl_circuit(R, L, voltage, step_time, end_time)
            label = f'R={R}Ω, L={L * 1000:.1f}mH'
            plt.plot(time, v_out, label=label)

    plt.title('Tension sur L pour différentes valeurs de R et L')
    plt.xlabel('Temps [s]')
    plt.ylabel('Tension [V]')
    plt.grid(True)
    plt.legend(fontsize='small')
    plt.show()
