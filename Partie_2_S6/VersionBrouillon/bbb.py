import numpy as np
import itertools
import csv
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

# Define the values for V, R, C, and L
V_values = np.linspace(1, 10, 10)  # 10 different values of voltage (1V to 10V)
R_values = np.linspace(1, 100, 10)  # 10 different values of resistance (1 Ohm to 100 Ohm)
C_values = np.linspace(1e-6, 1e-3, 10)  # 10 different values of capacitance (1uF to 1mF)
L_values = np.linspace(1e-6, 1e-3, 10)  # 10 different values of inductance (1uH to 1mH)

# Generate all combinations of V, R, C, and L
combinations = list(itertools.product(V_values, R_values, C_values, L_values))

# Create a CSV file to save the results
with open('simulation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['Circuit', 'V (V)', 'R (Ohm)', 'C (F)', 'L (H)', 'I_R (A)', 'I_C (A)', 'I_L (A)'])
    
    # Simulate each circuit and save the results
    for index, (V, R, C, L) in enumerate(combinations):
        print(f"Simulating circuit {index + 1} with V={V}V, R={R} Ohm, C={C} F, L={L} H")
        
        try:
            if R != 0:
                # Calculate the current through the resistor
                I_R = V / R
            else:
                I_R = 0

            # Capacitor: No current in steady-state DC (open circuit)
            I_C = 0

            # Inductor: In steady-state DC, behaves like a short circuit
            I_L = V / R if R != 0 else 0  # Assuming there's some resistor path for the inductor
            
            # Write the results to the CSV file
            writer.writerow([index + 1, V, R, C, L, I_R, I_C, I_L])
            
        except Exception as e:
            logger.error(f"Error simulating circuit {index + 1}: {e}")


