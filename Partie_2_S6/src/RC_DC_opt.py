# en premier lieu , on prepare les paramètres d'entrée 
import csv 
import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Probe.Plot import plot
from itertools import product
from PySpice.Unit import u_V, u_Ohm, u_F 
import multiprocessing
from tqdm import tqdm
import time

#######################################################################
# Les etapes :
# Une fonction pour générer tous les paramètres
# Une fonction de simulation DC bien conçue avec gestion des erreurs
# Les calculs pour tensions aux bornes des composants
######################################################################

#1)Génération de la grille de paramètres 
#Avec cette fonction , on crée la grille complète des paramètres 
#Ce qui donne  million de combinaison 
def generate_full_parameter_grid():
    r_vals = np.linspace(1e2, 1e5, 100)     # 100 valeurs de R
    c_vals = np.linspace(1e-9, 1e-5, 100)   # 100 valeurs de C
    vin_vals = np.linspace(0.1, 10.0, 100)  # 100 valeurs de Vin
    grid = list(product(r_vals, c_vals , vin_vals))
    return grid 

#2)Fonction de Simulation 
#Puis définir notre fonction de simulation (en utilisant le meme code du prototype  + qq modif )
#la fonction prend un tuplet(R,C,Vin)
def simulate_rc_dc(parametres):
    R_value, C_value, Vin = parametres
    try:
        circuit = Circuit(f'RC Circuit R={R_value} C={C_value} Vin={Vin}')
        circuit.V(1, 'vin', circuit.gnd, Vin @ u_V)
        circuit.R(1, 'vin', 'vout', R_value @ u_Ohm)
        circuit.C(1, 'vout', circuit.gnd, C_value @ u_F)

        simulator = circuit.simulator()
        analysis = simulator.operating_point()

        V_in = float(analysis.nodes['vin'][0])    # corrigé pour éviter le warning NumPy
        V_out = float(analysis.nodes['vout'][0])  # idem
        V_R = V_in - V_out # tension aux bornes de la résistance
        V_C = V_out# tension aux bornes du condensateur (vs gnd)


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
        # En cas d'erreur de convergence ou autre
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
#3)Lancement des Simulations
#la fonction principale pour le parallèlisme 
if __name__ == "__main__":
    #on commence par générer la grille avec la fonction déjà crée
    param_grid = generate_full_parameter_grid()
    print(f"Lancement de {len(param_grid)} simulations") 

    start = time.time()
       #on crée alors une Pool : 
    with multiprocessing.Pool() as pool:
        # l'utilisation de with va asssuer la fermeture de notre pool après l'éxécution 
        #la methode pool.map va assurer l'application de la ft de simulation sur chq elem du param_grid 
        # de façon parallèle (càd elle repartit les valeurs de la grille sur les diff ps de la pool )
        # => Ce qui réduit le tmp global necessaire pour l'execution
        # Utilisation de imap_unordered pour écrire les résultats dès qu'ils sont prêts
        res = list(tqdm(pool.imap(simulate_rc_dc, param_grid), total=len(param_grid)))
       

    # Statistiques
    success_count = sum(1 for r in res if r.get('V_in') is not None)
    fail_count = len(res) - success_count
    print(f"✅ Simulations réussies : {success_count}")
    print(f"❌ Simulations échouées : {fail_count}")

    # Écriture des résultats
    with open("bigRc_dataset.csv", "w", newline='') as f:
        
        fieldnames = ['R', 'C', 'Vin', 'V_in', 'V_out', 'V_R', 'V_C', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(res)  # écriture de tous les résultats (y compris les erreurs)

    end = time.time()
    print("Simulation terminée ! Données dans TEST.csv")
    print(f"⏱️ Temps total d'exécution : {end - start:.2f} secondes")

