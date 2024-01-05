# Projet 7 OpenClassRooms : Implémentez un modèle de scoring.

## Objectifs

1. Modèle de Scoring : Construire un modèle de scoring pour prédire la probabilité de faillite d'un client.
2. Dashboard Interactif : Développer un dashboard interactif pour interpréter les prédictions du modèle et améliorer la connaissance client.
3. Mise en Production : Mettre en production le modèle de scoring via une API et le dashboard interactif.
   
## Construction

Dans ce dépôt, vous trouverez :

1. un Notebook nommé '1. P7 - Analyse Exploratoire' : pour l'études des données, le feature engineering, et la selection de features.
2. un fichier python nommé 'fct_data' : fichier contenant les fonctiond utilisées pour le notebook Analyse Exploratoire.
3. un Notebook nommé : 2. P7 - Modélisation' : pour l'entrainement, l'optimisation des hyperparamètres et la sélection du modèle à retenir pour l'étude.
4. un fichier python nommé 'fct_modelisation' : fichier contenant les fonctions utilisées pour le notebook Modélisation.
5. un fichier python nommé 'data_preparation' : fichier contenant un script pour automatiser le trainement de l'ensemble des fichiers forunis la société.
6. un dossier 'API' : contenant la configuration locale de l'API.
     - backend réalisé avec FastAPI.
     - frontend réalisé avec Streamlit.
7. un fichier python nommé 'DataDrift.py' : permettant d'établir le rapport html --> Data_Drift_Report entre deux fichier en .pickle
8. Un PDF nommé 'Note Méthodologique' : rapport décrivant les différents points abordés dans ce projet.

## API en prodution : 

l'API en production est réalisée à partir de deux répertoires GitHub distincts :
   - [Backend avec FastAPI](https://github.com/maxsch38/API_backend_P7)
   - [Frontend avec Streamlit](https://github.com/maxsch38/API_Frontend_P7)

[API Deployée](https://apiocrp7maxsch-d299d2d6fa81.herokuapp.com)

## Autres informations : 

Lien de téléchargement des données d'entrées : [lien](https://www.kaggle.com/c/home-credit-default-risk/data)

La partie features engineering n'étant pas la partie la plus importante de ce projet, il nous était proposé d'utiliser des Notebooks disponiblent sur le site de Kaggle,
Notebooks utilisés pour l'Analyse exploratoire : 
  - [SOURCE_1](https://www.kaggle.com/code/ozericyer/homecreditdefaultrisk-test-train-eda-1/notebook)
  - [SOURCE_2](https://www.kaggle.com/code/rishabhrao/home-credit-default-risk-extensive-eda)
  - [SOURCE_3](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script)
