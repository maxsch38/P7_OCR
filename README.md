# Projet 7 - OpenClassRooms : IMPLEMENTEZ UN MODELE DE SCORING

<u>*Auteur : Maxime SCHRODER*</u>

## Contexte

<p align="center">
  <img src="Logo_pret_a_depenser.png" alt="Logo projet">
</p>

Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt. L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.). De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner. Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Données

Les données sont issues de Kaggle et sont disponibles à l'adresse suivante : https://www.kaggle.com/c/home-credit-default-risk/data

## Mission

1. Construire un modèle de scoring pour prédire la probabilité de faillite d'un client.
2. Développer un dashboard interactif pour interpréter les prédictions du modèle et améliorer la connaissance client.
3. Mettre en production le modèle de scoring via une API et le dashboard interactif.
   
## Construction

Dans ce dépôt, vous trouverez :
1. Le notebbok comprtant l'analyse exploratoire des données, la création de features engineering et la sélection de features : Notebook_1_analyse_exploratoire.ipynb



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
