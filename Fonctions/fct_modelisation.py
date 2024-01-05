###########################################################################################################################
# Fichier de fonctions du Projet 7 - fct_modelisation
###########################################################################################################################

###########################################################################################################################
## 1. Importation des librairies :

import numpy as np
import warnings

# Scikit-learn : 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, make_scorer, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

# MLFlow : 
import mlflow

# SMOTE : 
from imblearn.over_sampling import SMOTE

# HyperOpt : 
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval

# functools : 
from functools import partial

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

###########################################################################################################################
# 1. Prespocessing de data : 

def preprocess_data(data, reduce_fraction=None):
    """
    Prétraite les données d'entrée pour l'apprentissage automatique.

    Args:
        data (pd.DataFrame): Le DataFrame d'entrée contenant les caractéristiques et la variable cible ('TARGET').
        reduce_fraction (float, facultatif): Fraction de lignes à échantillonner pour la réduction de dimensions.
            Si None, aucune réduction de dimensions n'est effectuée. Par défaut, None.

    Returns:
        tuple: Un tuple contenant quatre éléments - X_train, X_test, y_train, y_test.

    Remarque :  Si `reduce_fraction` est spécifié, la fonction effectue une réduction de dimensions en
    échantillonnant de manière aléatoire une fraction des lignes de l'ensemble d'entraînement avant la standardisation.
    L'ensemble de test est standardisé en utilisant les mêmes paramètres de mise à l'échelle que l'ensemble d'entraînement.
    La fonction utilise SMOTE pour gérer le déséquilibre de classe avant toute réduction de dimensions.
    """
    
    # Séparation des données et de la Target : 
    X = data.drop('TARGET', axis=1)
    y = data['TARGET'].copy()
    
    # Séparation en ensemble de train et test : 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    # Gestion du déséquilibre de classe avec SMOTE : 
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Réduction de dimensions si spécifié : 
    if reduce_fraction is not None:
        # Calcul du nombre de lignes à échantillonner :
        num_samples = int(reduce_fraction * len(X_train))
        
        # Sélection aléatoire des indices à échantillonner : 
        sample_indices = np.random.choice(X_train.index, size=num_samples, replace=False)
        
        # Application de la réduction de dimensions : 
        X_train_reduce = X_train.loc[sample_indices]
        y_train_reduce = y_train.loc[sample_indices]
        
        # Standardisation du test et du train :     
        scaler = StandardScaler()
        scaler.fit(X_train)
    
        X_train_reduce = scaler.transform(X_train_reduce)
        X_test = scaler.transform(X_test)

        return X_train_reduce, X_test, y_train_reduce, y_test
    
    # Standardisation du test et du train :     
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


###########################################################################################################################
# 2. Optimisation des hyperparamètres Hyperopt et Trackage avec MLFlow : 

def search_best_model(data, dict_model, run_name=None, reduction_percentage=None, max_evals=10):
    """
    Optimise les hyperparamètres d'un modèle d'apprentissage automatique avec Hyperopt
    et enregistre les résultats avec MLflow.

    Args:
    - data (pandas.DataFrame): Le jeu de données d'entraînement.
    - dict_model (dict): Un dictionnaire contenant des informations sur le modèle d'apprentissage automatique à optimiser.
        - 'model' (object): Une instance du modèle scikit-learn à optimiser.
        - 'space' (dict): L'espace de recherche des hyperparamètres défini avec Hyperopt.
    - run_name (str, optional): Le nom à associer à l'exécution MLflow. Par défaut, None.
    - reduction_percentage (float, optional): Le pourcentage de données à utiliser pour l'optimisation. Par défaut, None.
    - max_evals (int, optional): Le nombre maximal d'évaluations pour l'optimisation des hyperparamètres. Par défaut, 10.

    """
    
    # Pre_processiong de data : 
    X_train, X_test, y_train, y_test = preprocess_data(
        data=data,
        reduce_fraction=reduction_percentage
        )
        
    # Création de trials pour récupérer l'ensemble des résultats d'optimisation : 
    trials = Trials()
            
    # Création de loss_fn incluant la fonction objective : 
    loss_fn = partial(objective,
                      X=X_train,
                      y=y_train,
                      model=dict_model['model'],
                      )
            
    # Recherche des hyperparamètres avec hyperopt : 
    best = fmin(
        fn=loss_fn,
        space=dict_model['space'],
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=2,
        )
            
    # Récupération des meilleurs résultats de validation : 
    best_validation_results= trials.best_trial['result']
            
    # Récupération des meilleurs hyperparamètres :
    best_params = space_eval(dict_model['space'], best)
    del best_params['threshold']

    # Enregistrement des résultats avec MLFlow : 
    with mlflow.start_run(run_name=run_name):
                
        # Enregistrement des meilleurs hyperparamètres : 
        mlflow.log_params(best_params)
        
        # Enregistrement des métrics sur la validation : 
        mlflow.log_metrics({'score_metier_val': best_validation_results['loss']})
        mlflow.log_metrics({'accuracy_val': best_validation_results['accuracy_score']})
        mlflow.log_metrics({'recall_val': best_validation_results['recall_score']})
        mlflow.log_metrics({'precision_val': best_validation_results['precision_score']})
        mlflow.log_metrics({'f1_val': best_validation_results['f1_score']})
        mlflow.log_metrics({'auc_val': best_validation_results['auc_score']})
        mlflow.log_metrics({'threshold': best_validation_results['threshold']})
         
        # Entrainement du meilleur modèle : 
        best_model = dict_model['model'].set_params(**best_params)
        best_model.fit(X_train, y_train)
                
        # Calcul et enregistrement des métriques sur le test : 
        y_proba = best_model.predict_proba(X_test)[:,1]
        y_pred = [1 if proba > best_validation_results['threshold'] else 0 for proba in y_proba]

        mlflow.log_metrics({'accuracy_test': accuracy_score(y_test, y_pred)})
        mlflow.log_metrics({'recall_test': recall_score(y_test, y_pred)})
        mlflow.log_metrics({'precision_test': precision_score(y_test, y_pred)})
        mlflow.log_metrics({'f1_test': f1_score(y_test, y_pred)})
        mlflow.log_metrics({'auc_test': roc_auc_score(y_test, y_pred)})
        mlflow.log_metrics({'score_metier_test': score_metier(y_test, y_proba, best_validation_results['threshold'])})

        # Enregistrement du meilleur modèle : 
        mlflow.sklearn.log_model(best_model, "best_model")
        

def objective(params, X, y, model):
    """ 
    Fonction objectif pour l'optimisation des hyperparamètres d'un modèle d'apprentissage automatique.

    Args:
        params (dict): Dictionnaire contenant les hyperparamètres à optimiser, y compris le seuil de décision.
        X (array-like): Les caractéristiques d'entraînement.
        y (array-like): Les étiquettes d'entraînement.
        model (object): Instance du modèle d'apprentissage automatique.

    Returns:
        dict: Dictionnaire contenant les résultats de la fonction objectif.
            - 'loss' (float): Score métier à minimiser.
            - 'status' (str): Statut de l'optimisation (STATUS_OK si réussi).
            - 'accuracy_score' (float): Score de précision moyen.
            - 'recall_score' (float): Score de rappel moyen.
            - 'precision_score' (float): Score de précision moyen.
            - 'f1_score' (float): Score F1 moyen.
            - 'auc_score' (float): Score AUC moyen.
            - 'threshold' (float): Seuil de décision optimal.
    """
    
    # Séparation des hyperparamètres du modèle et du seuil de décision : 
    model_params = {key: value for key, value in params.items() if key != 'threshold'}
    threshold = params['threshold']
    
    # Initialisation du modèle : 
    model = model.set_params(**model_params)
    
    # Entrainnnement croisé :
    scores = cross_validate(model, X, y, scoring=custom_scoring(threshold), cv=5, n_jobs=-1)   

    results = {
        'loss': np.mean(scores['test_score_metier']),  # Score métier à minimiser
        'status': STATUS_OK,
        'accuracy_score': np.mean(scores['test_accuracy']),
        'recall_score': np.mean(scores['test_recall']),
        'precision_score': np.mean(scores['test_precision']),
        'f1_score': np.mean(scores['test_f1']),
        'auc_score': np.mean(scores['test_auc']),
        'threshold': threshold,
    }
    
    return results


###########################################################################################################################
# 3. Score métier : 

def score_metier(y_true, y_prob, threshold):
    """
     Calcule le score métier en fonction des prédictions probabilistes d'un modèle.

    Args:
        y_true (array-like): Les vraies étiquettes.
        y_prob (array-like): Les probabilités prédites par le modèle.
        threshold (float): Le seuil de décision pour la classification binaire.

    Returns:
        float: Le coût total basé sur les prédictions du modèle.
    """
    #cost_fp = 1
    #cost_fn = 10
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    else:
        y_prob = y_prob
        
    y_pred = (y_prob > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    cost = (10 * cm[1, 0] + cm[0, 1]) / (10 * cm[1, 0] + cm[0, 1] + cm[1, 1] + cm[0, 0])

    #total_cost = cm[0, 1] * cost_fp + cm[1, 0] * cost_fn
    
    return cost


def custom_scoring(threshold):
    """
    Crée un dictionnaire de métriques de performance personnalisées pour l'évaluation croisée.

    Args:
        threshold (float): Le seuil de décision pour la classification binaire.

    Returns:
        dict: Dictionnaire contenant des métriques de performance.
            - 'score_metier': Score métier calculé avec le seuil donné.
            - 'accuracy': Score de précision.
            - 'recall': Score de rappel.
            - 'precision': Score de précision.
            - 'f1': Score F1.
            - 'auc': Aire sous la courbe ROC.
    """
    return {
        'score_metier': make_scorer(score_metier, threshold=threshold, needs_proba=True),
        'accuracy': 'accuracy',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'auc': 'roc_auc',
    }


###########################################################################################################################
# 4. Evaluation RMSE : 

def evaluate_model_rmse(model, X_train, X_test, y_train, y_test):
    """
     Évalue un modèle en calculant la racine carrée de l'erreur quadratique moyenne (RMSE) sur les ensembles d'entraînement et de test.

    Parameters:
    - model (object): Le modèle à évaluer.
    - X_train (array-like or pd.DataFrame): Les données d'entraînement.
    - X_test (array-like or pd.DataFrame): Les données de test.
    - y_train (array-like): Les étiquettes d'entraînement.
    - y_test (array-like): Les étiquettes de test.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    
    print(f"RMSE sur le train : {rmse_train}")
    print(f"RMSE sur le test : {rmse_test}")