#en premier lieu , on prepare les paramètres d'entrée 
import csv 
import numpy as np
from PySpice.Spice.Netlist import Circuit
#from PySpice.Probe.Plot import plot
from itertools import product
from PySpice.Unit import u_V, u_Ohm, u_F 
from tqdm import tqdm
import multiprocessing

#######################################################################
#Les etapes :
#Une fonction pour générer tous les paramètres
#Une fonction de simulation DC bien conçue avec gestion des erreurs
#Les calculs pour tensions aux bornes des composants
######################################################################

#1)Génération de la grille de paramètres 
#Avec cette fonction , on crée la grille complète des paramètres 
#Ce qui donne  million de combinaison 
def generate_full_parameter_grid(r_count=10, c_count=10, vin_count=10):
    r_vals = np.linspace(1e2, 1e5, r_count)     # 100 valeurs de R , # de 100 Ω à 100 kΩ
    c_vals = np.linspace(1e-9, 1e-5,c_count)   # 100 valeurs de C ,de 1 nF à 10 µF
    vin_vals = np.linspace(0.1, 10.0, vin_count)  # 100 valeurs de Vin ,de 0.1 V à 10 V
    #on cree un meshgrid 
    grid = list(product(r_vals,c_vals , vin_vals))
    return grid 

'''def generate_parameters(n_samples):
    #on va randomiser les valeurs de R et C 
    r_vals = np.random.uniform(1e2, 1e5, n_samples)
    c_vals = np.random.uniform(1e-9, 1e-5, n_samples)
    return list(zip(r_vals, c_vals))'''
#2)Fonction de Simulation 
#Puis définir notre fonction de simulation (en utilisant le meme code du prototype  + qq modif )
#la fonction prend un tuplet(R,C,Vin)
def simulate_rc_dc(parametres):
    "j'utilise un une analogie des exceptions pour les cas des erreurs aussi"
    R_value, C_value, Vin = parametres
    circuit = Circuit(f'RC Circuit R={R_value} C={C_value:.1e}F Vin={Vin:.1f}V')
    circuit.V(1, 'vin', circuit.gnd, Vin @ u_V)
    circuit.R(1, 'vin', 'vout', R_value @ u_Ohm)
    circuit.C(1, 'vout', circuit.gnd, C_value @ u_F)

    simulator = circuit.simulator()
    analysis = simulator.operating_point()

    #lecture des tensions
    V_in = float(analysis.nodes['vin'])
    V_out = float(analysis.nodes['vout'])
    # Calculs des tensions aux bornes
    V_R = V_in - V_out
    V_C = V_out  # vout - 0

    return {
            'R': R_value,
            'C': C_value,
            'Vin': Vin,
            'V_in': V_in,
            'V_out': V_out,
            'V_R': V_R,
            'V_C': V_C
    }

#3)Lancement des Simulations
#la fonction principale pour le parallèlisme 
if __name__ == "__main__":
    #on commence par générer la grille avec la fonction déjà crée
    param_grid=generate_full_parameter_grid()
    total =len(param_grid)
    print(f"Lancement de {total} simulations" ) 

    output_file = "1MRc_dataset.csv.csv"
    fieldnames = ['R', 'C', 'Vin', 'V_in', 'V_out', 'V_R', 'V_C' ]

    #Lancer les simulations avec le multiprocessing 
    "la classe Pool du module multiprocessing permet de paralléliser des oprs sur plsrs kernel de notre ps"
    "simplement c'est un groupe de ps qui peuevent s'executer au meme temps " 
    #Ouvrir le CSV et écrire l'en-tête
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    #on crée alors une Pool : 
    with multiprocessing.Pool() as pool : 
        #res =pool.map(simulate_rc_dc,param_grid) 
        # l'utilisation de with va asssuer la fermeture de notre pool après l'éxécution 
        #la methode pool.map va assurer l'application de la ft de simulation sur chq elem du param_grid 
        # de façon parallèle (càd elle repartit les valeurs de la grille sur les diff ps de la pool )
        # => Ce qui réduit le tmp global necessaire pour l'execution
        # Utilisation de imap_unordered pour écrire les résultats dès qu'ils sont prêts
        for result in tqdm(pool.imap_unordered(simulate_rc_dc, param_grid, chunksize=100),
                            total=total, desc="Simulations"):
                writer.writerow(result)
    print("Simulation terminé ! , Données dans 1MRc_dataset.csv ")

####################################################################
#Test de fonctionnement 
####################################################################      
# Tester avec une combinaison simple
'''result = simulate_rc_dc((1000, 1e-6, 5.0))
print(result)
# Vérifier la génération de la grille de paramètres
param_grid = generate_full_parameter_grid()
print(f"Nombre de simulations : {len(param_grid)}")
from multiprocessing import Pool

# Test avec un sous-ensemble de param_grid
test_param_grid = param_grid[:20]  # Prendre un petit sous-ensemble de la grille

with Pool() as pool:
    results = pool.map(simulate_rc_dc, test_param_grid)

# Vérifier les résultats
print(results)

#test_final 
with Pool() as pool:
    results = pool.map(simulate_rc_dc, param_grid)

# Sauvegarder les résultats dans un fichier CSV
with open("rc_dataset_dc.csv", "w", newline='') as f:
    fieldnames = ['R', 'C', 'Vin', 'V_in', 'V_out', 'V_R', 'V_C']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("✅ Simulations terminées. Résultats enregistrés dans rc_dataset_dc.csv")
'''

 