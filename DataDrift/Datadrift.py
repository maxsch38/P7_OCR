###########################################################################################################################
# Script pour execution du DataDrift.
###########################################################################################################################


###########################################################################################################################
# Importation des librairies : 
import pickle
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import time

###########################################################################################################################
# Fonction chargement pickle : 
def chargement_fichier(chemin): 
    
    with open(chemin, 'rb') as f:
        fichier = pickle.load(f)

    return fichier

###########################################################################################################################
# Fonction Rapport Datadrift : 
def DataDrift_Report(data_ref, data_current): 
    
     # Initialisation du rappport : 
    report = Report(metrics=[
    DataDriftPreset(),
    ])

    # Application du rapport au DataFrame : 
    report.run(reference_data=data_ref, current_data=data_current)
    
    return report

###########################################################################################################################
# Fonction Impression des résultats :
def affichage_result(dico):
    
    # Recupération des données : 
    nbre_col_tot = dico['metrics'][0]['result']['number_of_columns']
    nbre_col_drifted = dico['metrics'][0]['result']['number_of_drifted_columns']
    pourcentage_drift = dico['metrics'][0]['result']['share_of_drifted_columns']

    liste_colonnes_drifted = []
    dico_bis = dico['metrics'][1]['result']['drift_by_columns']

    for key in dico_bis.keys():
        if dico_bis[key]['drift_detected'] == True:
            liste_colonnes_drifted.append(key)
            
    # Affichage des résultats : 
    print(f"""RESULTATS : 
        
        Nombre total de colonnes analyées : {nbre_col_tot}
        Nombre de colonnes avec un DataDrift detecté : {nbre_col_drifted}
            
                --> Soit {pourcentage_drift*100:.2f} %
        """)
    print('--'*50)
    if len(liste_colonnes_drifted) == 0: 
        print("Aucune colonne ne comporte de DataDrift.")
        return
    elif len(liste_colonnes_drifted)  == 1 : 
        print('Colonne avec du DataDrift :\n')
    else: 
        print('Colonnes avec du DataDrift :\n')

    for col in liste_colonnes_drifted: 
        print(f"\t- {col}")
        
###########################################################################################################################
# Fonction principale :
def main(data_ref_path, data_current_path): 
    
    # Chargement des fichiers = 
    data_ref = chargement_fichier(
        chemin=data_ref_path,
    )
    
    data_current = chargement_fichier(
        chemin=data_current_path,
    )
     # Suppression de la colonne TARGET : 
    data_ref = data_ref.drop('TARGET', axis=1)
    
    # Démarrage :
    print('')
    print('--'*50)
    print('Analyse du DataDirft en cours....')
    print('--'*50)
    
    # Analyse DataDrift : 
    start_time = time.perf_counter()
    
    report = DataDrift_Report(
        data_ref=data_ref,
        data_current=data_current,
    )
    
    end_time = time.perf_counter()
    
    print('\nAnalyse du DataDirft terminé !')
    print(f"Temps d'execution : {end_time - start_time :.2f} secondes\n")
    print('--'*50)
    
    # Affichage des resultats : 
    affichage_result(
        dico=report.as_dict(),
    )
    
    # Enregistrement au format HTML : 
    report.save_html('Data_Drift_Report')
    print('')
    print('--'*50)
    print("Rapport DataDrift enregistré au format HMTL\n")

###########################################################################################################################
# Script :

if __name__ == '__main__':
    
    # Demande du chemin du fichier de référence : 
    data_ref_path = input("Entrez le chemin du fichier de reference : ")
    
    # Demande du chemin du fichier à comparer : 
    data_current_path = input("Entrez le chemin du fichier à comparer : ")

    # Execution : 
    main(
        data_ref_path=data_ref_path,
        data_current_path=data_current_path,
        )
