#en premier lieu , on prepare les paramètres d'entrée 
import csv 
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Probe.Plot import plot
from itertools import product
from PySpice.Unit import u_V, u_Ohm, u_F 
import multiprocessing

#######################################################################
#Les etapes :
#Une fonction pour générer tous les paramètres
#Une fonction de simulation DC bien conçue avec gestion des erreurs
#Les calculs pour tensions aux bornes des composants
######################################################################

#Avec cette fonction , on crée la grille complète des paramètres 
#Ce qui donne  million de combinaison 
def generate_full_parameter_grid():
    r_vals = np.linspace(1e2, 1e5, 10)     # 100 valeurs de R
    c_vals = np.linspace(1e-9, 1e-5, 10)   # 100 valeurs de C
    vin_vals = np.linspace(0.1, 10.0, 1)  # 100 valeurs de Vin
    #on cree un meshgrid 
    grid = list(product(r_vals,c_vals , vin_vals))
    return grid 

'''def generate_parameters(n_samples):
    #on va randomiser les valeurs de R et C 
    r_vals = np.random.uniform(1e2, 1e5, n_samples)
    c_vals = np.random.uniform(1e-9, 1e-5, n_samples)
    return list(zip(r_vals, c_vals))'''
#Puis définir notre fonction de simulation (en utilisant le meme code du prototype  + qq modif )
#la fonction prend un tuplet(R,C,Vin)
def simulate_rc_dc(parametres):
    "j'utilise un une analogie des exceptions pour les cas des erreurs aussi"
    R_value, C_value, Vin = parametres
    try:
        circuit = Circuit(f'RC Circuit R={R_value} C={C_value} Vin={Vin}')
        circuit.V(1, 'vin', circuit.gnd, Vin @ u_V)
        circuit.R(1, 'vin', 'vout', R_value @ u_Ohm)
        circuit.C(1, 'vout', circuit.gnd, C_value @ u_F)

        simulator = circuit.simulator()
        analysis = simulator.operating_point()

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

    except Exception as e:
        # Pour gérer les erreurs de simulation
        return {
            'R': R_value,
            'C': C_value,
            'Vin': Vin,
            'V_in': None,
            'V_out': None,
            'V_R': None,
            'V_C': None,
            'error': str(e)
        }

#la fonction principale pour le parallèlisme 
if __name__ == "__main__":
    #on commence par générer la grille avec la fonction déjà crée
    param_grid=generate_full_parameter_grid()
    print(f"Lancement de {len(param_grid)} simulations" ) 
    #Lancer les simulations avec le multiprocessing 
    "la classe Pool du module multiprocessing permet de paralléliser des oprs sur plsrs kernel de notre ps"
    "simplement c'est un groupe de ps qui peuevent s'executer au meme temps " 
    #on crée alors une Pool : 
    with multiprocessing.Pool() as pool : 
        res =pool.map(simulate_rc_dc,param_grid) 
        # l'utilisation de with va asssuer la fermeture de notre pool après l'éxécution 
        #la methode pool.map va assurer l'application de la ft de simulation sur chq elem du param_grid 
        # de façon parallèle (càd elle repartit les valeurs de la grille sur les diff ps de la pool )
        # => Ce qui réduit le tmp global necessaire pour l'execution

    with open("bigRc_dataset.csv","w",newline='') as f : 
        fieldnames = ['R', 'C', 'V_in', 'V_out' ,'V_R', 'V_C', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(res)
    print("Simulation terminé ! , Données dans bigRc_dataset.csv ")

####################################################################
#Test de fonctionnement 
####################################################################      
# Tester avec une combinaison simple
result = simulate_rc_dc((1000, 1e-6, 5.0))
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


 