###########################################################################################################################
# Fichier d'automatisation du pré-traitement Projet 7 
###########################################################################################################################
""" 
Ce fichier permet l'automatisation du regroupement des datasets : 
- application_train/test, ici appelé application globalement.
- bureau et bureau_balance
- previous_appplication
- POS_cash_balance
- installments_payments
- credit_card_balance

Il filtre sur les colonnes, pour ne garder que les features sélectionnés dans la branche d'analyse exploratoire (features importance réalisé avec LightGBM)
Il réalise également l'imputation des valeurs manquantes par le mode (définit comme étant la meilleure imputation dans la branche modélisation), 

Renvoie au final deux fichiers data_train et data_test prêt à l'utilisation par le modèle. 
"""
###########################################################################################################################
## 1. Importation des libraires :

import os 
import pickle
import chardet
import warnings
import numpy as np
import pandas as pd 

from sklearn.impute import SimpleImputer

# Ignorer les warnings de type FutureWarning : 
warnings.filterwarnings("ignore", category=FutureWarning)

###########################################################################################################################
## Importation du DataFrame : 

def importation_df (dossier, fichier):
    """
   Importe un fichier CSV en utilisant l'encodage détecté automatiquement.

    Args:
        dossier (str): Chemin vers le dossier contenant le fichier CSV.
        fichier (str): Nom du fichier CSV.

    Returns:
        pandas.DataFrame or None: Le DataFrame contenant les données du fichier CSV,
        ou None en cas d'erreur lors de l'importation.
    """
    
    # Création du chemin : 
    chemin = os.path.join(dossier, fichier)

    # Récupération de l'encodage du fichier : 
    with open(chemin, 'rb') as file:
        detection = chardet.detect(file.read())
    
    encodage = detection['encoding']
    
    # Lecture et ouverture du fichier :      
    try: 
        df = pd.read_csv(chemin, encoding=encodage)
        return df
    
    except FileNotFoundError:
        print(f"Le fichier {chemin} n'a pas été trouvé.")
        return None
    
    except UnicodeDecodeError as e:
        print(f"Erreur de décodage lors de la lecture du fichier {chemin} : {e}")
        return None

    except Exception as e:
        print(f"Erreur inattendue lors de l'importation du fichier {chemin} : {e}")
        return None
    
###########################################################################################################################
## Enregistrement et chargement Pickle : 

def chargement_pickle(name, chemin): 
    
    path = chemin + '/' + name + '.pickle'
    
    with open(path, 'rb') as f:
        fichier = pickle.load(f)

    return fichier


def enregistrement_pickle(name, chemin, fichier):
    
    path = chemin + '/' + name + '.pickle'

    with open(path, 'wb') as f:
        pickle.dump(fichier, f)

###########################################################################################################################
## OneHotEncoding :  

def one_hot_encoder(df, nan_as_category=True):
    """
    Effectue l'encodage one-hot des colonnes catégorielles d'un DataFrame.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données à encoder.
        nan_as_category (bool, optional): Indique si les valeurs NaN doivent être traitées comme une catégorie. Par défaut, True.

    Returns:
        pandas.DataFrame: Le DataFrame avec les colonnes catégorielles encodées en one-hot.
        list: La liste des nouvelles colonnes créées lors de l'encodage.
    """
    
    # 1. Récupération des colonnes du DataFrame : 
    original_columns = list(df.columns)
    
    # 2. Récupération des colonnes catégorielles : 
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    # 3. Gestion des caractères spéciaux : 
    for col in categorical_columns:
        df[col] = df[col].str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_').str.replace(',', '_').str.replace(':', '_')

    # 4. Création du OneHotEncoding :
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    
    # 5. Récupération des colonnes encodées : 
    new_columns = [c for c in df.columns if c not in original_columns]
    
    return df, new_columns

###########################################################################################################################
## Application train / test :

def application(dossier, nan_as_category=False):
    """
    Applique des transformations sur les données du fichier 'application.csv'.

    Args:
        dossier (str): Chemin du dossier contenant le fichier 'application.csv'.
        nan_as_category (bool, optional): Si True, considère les valeurs NaN comme des catégories.
            Defaults to False.

    Returns:
        pandas.DataFrame: Le DataFrame transformé avec les modifications spécifiées.
    """
    
    # Importation du fichier : 
    train = importation_df(
        fichier='application_train.csv',
        dossier=dossier,
    )
    
    test = importation_df(
        fichier='application_test.csv',
        dossier=dossier,
    )
    
    # Concaténation : 
    df = pd.concat([train, test], ignore_index=True)
    
    #  Filtre des XNA sur la variable CODE_GENDER : 
    df = df.loc[df['CODE_GENDER'] != 'XNA']
    
    # Gestion des colonnes avec variables Binaires : 
    for feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[feature], _ = pd.factorize(df[feature])
        
    # Encodage des variables catégorielles : 
    df, _ = one_hot_encoder(
        df=df, 
        nan_as_category=nan_as_category,
        )

    # Remplacement des valeurs de DAYS_EMPLOYED: 365.243 par nan : 
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        
    # Création de nouvelles features (en pourcentage) : 
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    return df


###########################################################################################################################
## Bureau et bureau_balance :

def bureau_and_bureau_balance(dossier, nan_as_category=True):
    """
    Effectue des opérations de prétraitement sur les fichiers 'bureu.csv' et 'bureau_balance.csv'
    et agrège les informations pour créer un DataFrame consolidé.

    Args:
        dossier (str): Chemin du dossier contenant les fichiers 'bureu.csv' et 'bureau_balance.csv'.
        nan_as_category (bool, optional): Si True, considère les valeurs NaN comme des catégories.
            Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame agrégé avec les informations consolidées des fichiers 'bureu.csv' et 'bureau_balance.csv'.
    """
    
    # Importation des fichiers : 
    bureau = importation_df(
        fichier='bureau.csv',
        dossier=dossier,
    )
    
    bureau_balance = importation_df(
        fichier='bureau_balance.csv',
        dossier=dossier,
    )
    
    # Encodage des variables catégorielles de bureau_balance et bureau : 
    bureau_balance, bb_cat = one_hot_encoder(
        df=bureau_balance, 
        nan_as_category=nan_as_category,
        )

    bureau, bureau_cat = one_hot_encoder(
        df=bureau, 
        nan_as_category=nan_as_category,
        )

    # Aggrégation de bureau_balance en fonction de la variable 'SK_ID_BUREAU' :
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']} # Création du dictionnaire d'aggrégation

    for col in bb_cat:
        bb_aggregations[col] = ['mean'] # Complétion du dictionnaire pour les colonnes catégorielles
    
    bureau_balance = bureau_balance.groupby('SK_ID_BUREAU').agg(bb_aggregations) # Aggrégation
    bureau_balance.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_balance.columns.tolist()]) # Applatissement des noms de colonnes

    # Ajout des variables de bureau_balance à bureau : 
    bureau = bureau.join(bureau_balance, how='left', on='SK_ID_BUREAU')
    bureau = bureau.drop(['SK_ID_BUREAU'], axis=1)

    # Création d'un dictionnaire d'aggrégation pour les variables numériques de bureau : 
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }

    # Création d'un dictionnaire d'aggrégation pour les variables catgorielles de bureau : 
    cat_aggregations = {}

    for cat in bureau_cat: # Pour les variables initales de bureau
        cat_aggregations[cat] = ['mean']
        
    for cat in bb_cat: # Pour les variables provenant de bureau_balance
        cat_aggregations[cat + "_MEAN"] = ['mean']

    # Aggrégation de bureau en fonction de la variable 'SK_ID_CURR' :
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    # Création de variables spécifiques pour les crédits Actifs (utilisation uniquement des variables numériques)
    active = bureau.loc[bureau['CREDIT_ACTIVE_Active'] == 1] # Récupération des crédits actifs 
    active = active.groupby('SK_ID_CURR').agg(num_aggregations) # Aggrégation spécifiques sur ces crédits
    active.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active.columns.tolist()])
    bureau_agg = bureau_agg.join(active, how='left', on='SK_ID_CURR') # Ajout de ces variables à bureau_agg 

    # Création de variables spécifiques pour les crédits clôturés (utilisation uniquement des variables numériques)
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed.columns.tolist()])
    bureau_agg = bureau_agg.join(closed, how='left', on='SK_ID_CURR')
    
    return bureau_agg

###########################################################################################################################
## Previous application : 

def previous_application(dossier, nan_as_category=True): 
    """
    Effectue des opérations de prétraitement sur le fichier 'previous_application.csv' 
    et agrège les informations pour créer un DataFrame consolidé.

    Args:
        dossier (str): Chemin du dossier contenant le fichier 'previous_application.csv'.
        nan_as_category (bool, optional): Si True, considère les valeurs NaN comme des catégories.
            Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame agrégé avec les informations consolidées du fichier 'previous_application.csv'.
    """
    
    # Importation du fichier : 
    previous = importation_df(
        fichier='previous_application.csv',
        dossier=dossier,
    )

    # Encodage des variables catégorielles : 
    previous, cat_col = one_hot_encoder(
        df=previous, 
        nan_as_category=nan_as_category,
    )

    # Gestion des jours égals à 365243 : (mise à valeur nan)
    previous['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    previous['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    previous['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    previous['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    previous['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    # Création d'une nouvelle variable : (pourcentage - valeur demandé / valeur reçue)
    previous['APP_CREDIT_PERC'] = previous['AMT_APPLICATION'] / previous['AMT_CREDIT']

    # Création d'un dictionnaire d'aggrégation pour les variables numériques de previous : 
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        }

    # Création d'un dictionnaire d'aggrégation pour les variables catgorielles de previous : 
    cat_aggregations = {}

    for cat in cat_col:
        cat_aggregations[cat] = ['mean']
            
    # Aggrégation de previous en fonction de la variable 'SK_ID_CURR' : 
    prev_agg = previous.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])


    # Création de variables spécifiques pour les candidatures approuvées (utilisation uniquement des variables numériques)
    approved = previous.loc[previous['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved.columns.tolist()])

    prev_agg = prev_agg.join(approved, how='left', on='SK_ID_CURR')


    # Création de variables spécifiques pour les candidatures refusées (utilisation uniquement des variables numériques)
    refused = previous.loc[previous['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused.columns.tolist()])
    prev_agg = prev_agg.join(refused, how='left', on='SK_ID_CURR')

    return prev_agg

###########################################################################################################################
## POS_cash_balance : 

def pos_cash(dossier, nan_as_category=True):
    """
    Effectue des opérations de prétraitement sur le fichier 'POS_CASH_balance.csv' 
    et agrège les informations pour créer un DataFrame consolidé.

    Args:
        dossier (str): Chemin du dossier contenant le fichier 'POS_CASH_balance.csv'.
        nan_as_category (bool, optional): Si True, considère les valeurs NaN comme des catégories.
            Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame agrégé avec les informations consolidées du fichier 'POS_CASH_balance.csv'.
    """
    
    # Importation du fichier :
    pos_cash = importation_df(
        fichier='POS_CASH_balance.csv',
        dossier=dossier,
    )

    # Encodage des variables catégorielles de pos_cash: 
    pos_cash, cat_col = one_hot_encoder(
        df=pos_cash, 
        nan_as_category=nan_as_category,
    )
    
    # Création d'un dictionnaire d'aggrégation pour pos_cash : 
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
        }
    
    for cat in cat_col:
        aggregations[cat] = ['mean']

    # Aggrégation de pos_cash en fonction de la variable 'SK_ID_CURR' : 
    pos_agg = pos_cash.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Création de la variables 'POS_COUNT' (nombre de dépôt effectué par le demandeur)
    pos_agg['POS_COUNT'] = pos_cash.groupby('SK_ID_CURR').size()
    
    return pos_agg


###########################################################################################################################
## Installments payments: 

def installments_payments(dossier, nan_as_category=True):
    """
    Effectue des opérations de prétraitement sur le fichier 'installments_payments.csv' 
    et agrège les informations pour créer un DataFrame consolidé.

    Args:
        dossier (str): Chemin du dossier contenant le fichier 'installments_payments.csv'.
        nan_as_category (bool, optional): Si True, considère les valeurs NaN comme des catégories.
            Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame agrégé avec les informations consolidées du fichier 'installments_payments.csv'.
    """

    # Importation du fichier : 
    installments = importation_df(
        fichier='installments_payments.csv',
        dossier=dossier,
    )

    # Encodage des variables catégorielles de pos_cash: 
    installments, cat_col = one_hot_encoder(
        df=installments, 
        nan_as_category=nan_as_category,
    )

    # Création de nouvelles variables : (Pourcentage et différence payés à chaque versement) 
    installments['PAYMENT_PERC'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
    installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']

    # Créaton des variables DPD et DBD : 
    """
    DPD (days past due) : nombre de jours de retard dans le paiement. 
    DBD (days before due) : nombre de jours restants avant la date prévue de paiement. 

    ----> SUpression des valeurs négatives (mise à O).
    """
        
    installments['DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['DBD'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']

    installments['DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
    installments['DBD'] = installments['DBD'].apply(lambda x: x if x > 0 else 0)

    # Création d'un dictionnaire d'aggrégation pour installments : 
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
    
    for cat in cat_col:
        aggregations[cat] = ['mean']

    # Aggrégation de installments en fonction de la variable 'SK_ID_CURR' : 
    installments_agg = installments.groupby('SK_ID_CURR').agg(aggregations)
    installments_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in installments_agg.columns.tolist()])

    # Création de la variables 'INSTAL_COUNT' (nombre de versemant effectué par le demandeur)
    installments_agg['INSTAL_COUNT'] = installments.groupby('SK_ID_CURR').size()

    return installments_agg


###########################################################################################################################
## Credit card balance : 

def credit_card_balance(dossier, nan_as_category=True):
    """ 
    Effectue des opérations de prétraitement sur le fichier 'credit_card_balance.csv' 
    et agrège les informations pour créer un DataFrame consolidé.

    Args:
        dossier (str): Chemin du dossier contenant le fichier 'credit_card_balance.csv'.
        nan_as_category (bool, optional): Si True, considère les valeurs NaN comme des catégories.
            Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame agrégé avec les informations consolidées du fichier 'credit_card_balance.csv'.
    """

    # Importation du fichier : 
    credit_card = importation_df(
        fichier='credit_card_balance.csv',
        dossier=dossier,
    )

    # Encodage des variables catégorielles de credit_card: 
    credit_card, cat_col = one_hot_encoder(
        df=credit_card, 
        nan_as_category=nan_as_category,
    )

    # Suppression de 'SK_ID_PREV' : 
    credit_card = credit_card.drop('SK_ID_PREV', axis=1)

    # Récupération des variables numériques : 
    num_col = [col for col in credit_card.columns if col not in cat_col]

    # Création d'un dictionnaire d'aggrégation : 
    aggregations = {}

    for col in num_col: 
        aggregations[col] = ['min', 'max', 'mean', 'sum', 'var']

    for col in cat_col: 
        aggregations[col] = ['mean', 'sum', 'var']   

    # Aggrégation de credit_card :
    credit_card_agg = credit_card.groupby('SK_ID_CURR').agg(aggregations)
    credit_card_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in credit_card_agg.columns.tolist()])

    # Création de la variables 'CC_COUNT' (nombre de carte de crédit par demandeur)
    credit_card_agg['CC_COUNT'] = credit_card.groupby('SK_ID_CURR').size()

    return credit_card_agg


###########################################################################################################################
## Imputation et réduction des type de données :

def imputation_reduction(df):
    """
    Prétraitement des données d'entraînement et de test.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données d'entraînement et de test.
    
    Returns:
        pandas.DataFrame:  Un DataFrame avec les données prétraitées,
    """
    
    # Création de l'imputer : 
    imputer = SimpleImputer(strategy='most_frequent')
    
    # Séparation en train et test : 
    df_train = df.loc[df['TARGET'].notna()]
    df_test = df.loc[df['TARGET'].isna()]

    # Récupération de la colonne TARGET : 
    target = df_train['TARGET'].copy()
    target = target.astype('int8')
    
    df_train = df_train.drop('TARGET', axis=1)
    df_test = df_test.drop('TARGET', axis=1)
    
    # Gestion des valeurs infinies : 
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
   
    # Imputation des valeurs manquantes : 
    df_train_imputed = pd.DataFrame(imputer.fit_transform(df_train), columns=df_train.columns, index=df_train.index)
    df_test_imputed = pd.DataFrame(imputer.transform(df_test), columns=df_test.columns, index=df_test.index)
   
    # Conversion des colonnes booléennes en int8 :
    bool_columns = df_train.select_dtypes(include=['bool']).columns
   
    df_train_imputed[bool_columns] = df_train_imputed[bool_columns].astype('int8')
    df_test_imputed[bool_columns] = df_test_imputed[bool_columns].astype('int8')
   
    # Conversion des colonnes float64 en float32 :
    float_columns = df_train.select_dtypes(include=['float64']).columns
   
    df_train_imputed[float_columns] = df_train_imputed[float_columns].astype('float32')
    df_test_imputed[float_columns] = df_test_imputed[float_columns].astype('float32')
   
    # Conversion des colonnes int64 en int8 :
    int_columns = df_train.select_dtypes(include=['int64']).columns
   
    df_train_imputed[int_columns] = df_train_imputed[int_columns].astype('int8')
    df_test_imputed[int_columns] = df_test_imputed[int_columns].astype('int8')
   
    # Ajout de la colonne TARGET à df_train : 
    df_train_imputed['TARGET'] = target
    
    return df_train_imputed, df_test_imputed

###########################################################################################################################
## Fonction principale :

def main(dossier_donnees, dossier_sauvegarde, fichier_col): 
    """
    Fonction principale pour le traitement des fichiers et la création du DataFrame consolidé 'data'.

    Args:
        dossier (str): Chemin du dossier contenant les fichiers nécessaires.
        fichier_col (str): Nom du fichier contenant la liste des colonnes à conserver.
    """
    
    # Création de data à partir du fichier application : 
    data = application(dossier=dossier_donnees)
    print("Traitement de application.csv terminé.")
    
    # Création de bureau et aggrégation dans data : 
    bureau = bureau_and_bureau_balance(dossier=dossier_donnees)
    data = data.join(bureau, how='left', on='SK_ID_CURR')
    print("Traitement de bureau.csv et bureau_balance.csv terminé.")
    
    # Création de previous et aggrégation dans data : 
    previous = previous_application(dossier=dossier_donnees)
    data = data.join(previous, how='left', on='SK_ID_CURR')
    print("Traitement de previous_application.csv terminé")
    
    # Création de pos et aggrégation dans data : 
    pos = pos_cash(dossier=dossier_donnees)
    data = data.join(pos, how='left', on='SK_ID_CURR')
    print("Traitement de POS_CASH_balance.csv terminé.")
       
    # Création de installments et aggrégation dans data : 
    installments = installments_payments(dossier=dossier_donnees)
    data = data.join(installments, how='left', on='SK_ID_CURR')
    print("Traitement de installments_payments.csv terminé.")
    
    # Création de credit_card et aggrégation dans data :
    credit_card = credit_card_balance(dossier=dossier_donnees)
    data = data.join(credit_card, how='left', on='SK_ID_CURR')
    print("Traitement de credit_card_balance.csv terminé.")
    
    # Mise de la variable SK_ID_CURR en index : 
    data = data.set_index('SK_ID_CURR')
    
    # Filtre des colonnes :
    ls_col = chargement_pickle(
        name=fichier_col,
        chemin=dossier_sauvegarde,
    )
    
    data = data.loc[:,ls_col]
    
    # Imputation des valeurs manquantes par le mode : 
    data_train, data_test = imputation_reduction(
        df=data,
    )
    
    # Enregistrement des data_train et data_test : 
    enregistrement_pickle(
        name='data_final_train',
        chemin=dossier_sauvegarde,
        fichier=data_train,
    )
    
    enregistrement_pickle(
        name='data_final_test',
        chemin=dossier_sauvegarde,
        fichier=data_test,
    )
    
    print("Traitement des fichiers et création de data_train et data_test terminé")


###########################################################################################################################
## Script : 
if __name__ == '__main__':
    
    # Dossier de données : 
    dossier_donnees = '1. Données'
    
    # Dossier de sauvegarde : 
    dossier_sauvegarde = '2. Sauvegardes'
    
    # Liste des colonnes à filtrer : 
    fichier_col = 'Liste_features'

    # Lancemeent du script : 
    print("Lancement du script 'data_preparation.py'.")
    
    main(
        dossier_donnees=dossier_donnees,
        dossier_sauvegarde=dossier_sauvegarde,
        fichier_col=fichier_col,
        )
