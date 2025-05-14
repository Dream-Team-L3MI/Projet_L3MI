#On importe la classe Circuits qui permet de créer un nv circuits
import matplotlib.pyplot as plt
from PySpice.Spice.Netlist import Circuit
#from PySpice.Unit import *

#To define a simple RC circuit with a DC voltage source of 1V

circuit = Circuit('Simple RC Circuit')
#Input est le noeud positif càd le pt de départ du courant
circuit.V('1', 'input', circuit.gnd, 'DC 1V')
#circuit.gnd est la masse (0 volt)
#DC 1V : cette source fournit 1 volt en courant continu
circuit.R('1', 'input', 'output', 1e3)
#Résistance : 1 kilo-oum entre le noeud input et output
circuit.C('1', 'output', circuit.gnd, 10e-6)
#Condensateur C entre le output et la masse (gnd) de 10 microF

#Transient: simulation transitoire sert à observer l'évolution d'un circuit dans le tmp
#est utile pr les circuits non pas en régime permanent (exp : charge/ décharge de condensateur)
#dans notre exp ici : le condensateur n'a pas encr de charge au début, (la tension à ses bornes va augmenter
#progressivement qd on applique 1V.
#Création du simulateur SPICE pr le circuit
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
#cette ligne crée un objet Simulator qui utilisera NgSpice pr simuler le circuit.
simulator.initial_condition(output=0)
#Cette ligne indique une condition initiale : ici, la
#tension sur le nœud output (entre R et C) est 0 volt au temps t=0.
#Pour un condensateur, la tension à ses bornes ne peut pas changer instantanément.
#On dit donc à SPICE qu'au début le condensateur est déchargé
#Simulation transitoire:
analysis = simulator.transient(step_time=1e-3, end_time=100e-3) #100ms
#Comme si on dit ici : on calcule les tensions toutes les 1 ms, plus petit plus précis mais plus long
#Et la durée totale de la simulation est 100ms

v_out = analysis['output']
# pr le plot on intègre avec matplotlib

plt.figure()
plt.plot(analysis.time, v_out)
plt.title('le voltage du condensateur au cours du temps')
plt.xlabel('Temps [s]')
plt.ylabel('Tension [V]')
plt.grid(True)
plt.show()






