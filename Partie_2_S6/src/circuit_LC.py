# Condition initiale pour l'inducteur (pas de courant initial dans l'inducteur)
simulator.initial_condition(output=0)

# Simulation transitoire
analysis = simulator.transient(step_time=1e-3, end_time=100e-3)

# Extraction des résultats de simulation pour le condensateur (tension)
v_out = analysis['output']

# Visualisation des résultats avec matplotlib
plt.figure()
plt.plot(analysis.time, v_out)
plt.title('Le voltage du condensateur dans le circuit LC')
plt.xlabel('Temps [s]')
plt.ylabel('Tension [V]')
plt.grid(True)
plt.show()
