###########################################################################################################################
# Fichier de fonctions du Projet 7 - fct_data
###########################################################################################################################

###########################################################################################################################
## 1. Importation des libraires :
 
from sys import displayhook
import os 
import pickle
import chardet
import warnings

# Data 
import numpy as np
import pandas as pd 

# Graphique
import matplotlib.pyplot as plt 
import seaborn as sns
from phik import phik_matrix

# Scikit-learn : 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

# LightGBM : 
from lightgbm import LGBMClassifier

# Ignorer les warnings de type FutureWarning : 
warnings.filterwarnings("ignore", category=FutureWarning)

###########################################################################################################################
# 2. Enregistrement Pickle : 

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
# 3. Information sur les DataFrames

def info_variables(df): 
    """
    Cette fonction analyse un DataFrame et fournit des informations sur ses caractéristiques, telles que les types de variables,
    la dimension, le nombre de lignes dupliquées et la répartition des types de variables sous forme d'un graphique en secteurs.

    Args:
        df (pandas.DataFrame): Le DataFrame à analyser.
    """
    
     # 1. Création d'une liste de dictionnaires pour stocker les informations : 
    data_info = []
    
    # 2. Récupération des éléments pour chaque colonnes du DataFrame :
    for col in df.columns:
        col_name = col
        col_type = str(df[col].dtype)

         # Ajout des éléments à la liste : 
        data_info.append({'Nom_de_colonne': col_name,
                          'Data_type': col_type})
        
    # 3. Création du DataFrame à partir de la liste de dictionnaires : 
    column_info = pd.DataFrame(data_info)
    
    # 4. Définition de l'index de column_info :
    column_info = column_info.set_index('Nom_de_colonne')
    
    # 5. Création de df_graph : 
    grouped = column_info.groupby('Data_type')
    nbre_par_type = grouped['Data_type'].count()
    pourcentage = (nbre_par_type / len(data_info)) * 100
    df_graph = pd.DataFrame({'Nombre par type de variable': nbre_par_type, '% des types de variable': pourcentage})
    df_graph = df_graph.reset_index()

    # 6. Affichage du résultats : 
    ########################################################
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    
    # 6.1 Dimmension :
    print(f"Dimmension du DataFrame : {df.shape}")
    print('--' * 50)
    
    # 6.2 Valeurs dupliquées : 
    print(f"Nombre de lignes dupliquées : {df.duplicated().sum()}")
    print('--' * 50)
    
    # 6.3 Affichage des 3 premières lignes : 
    df
    
    # 6.3 Affichage du DataFrame avec des variables : 
    print("Information sur les types de variables :")
    displayhook(column_info)
    print('--' * 50)
    
    # Affichage du DataFrame de la répartition des type de variable et tracé du grapique : 
    print("Répartition des types de variables :")
    displayhook(df_graph)
    
    
    plt.figure()
    labels = df_graph['Data_type']
    sizes = df_graph['% des types de variable']
    
    plt.pie(
        x=sizes,
        labels=labels,
        autopct='%1.2f%%',
        startangle=90,
        )
    plt.show()
    
    print('--' * 50)
    
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    

def describe_dataframe(df):
    """
     Cette fonction génère des statistiques descriptives pour un DataFrame, en affichant les informations
    sur les colonnes numériques et catégorielles.

    Args:
        df (pandas.DataFrame): Le DataFrame à analyser.
    """
   
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    
    # 1. Colonnes numériques : 
    numeric_columns = df.select_dtypes(include='number')
    if not numeric_columns.empty:
        # Info sur la répartition des variables numériques :
        print(f"Statistiques descriptives des colonnes numériques : \n")
        displayhook(numeric_columns.describe())
    
    # 2. Colonnes catégorielles : 
    categorical_columns = df.select_dtypes(include='object')
    if not categorical_columns.empty:
        # Info sur la répartition des variables catégorielles :
        print(f"Statistiques descriptives des colonnes catégorielles : \n")
        displayhook(categorical_columns.describe())
        
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    

def valeurs_manquantes(df, titre_graph): 
    """
    Cette fonction analyse un DataFrame pour identifier les valeurs manquantes et génère un graphique
    montrant le pourcentage de valeurs manquantes pour chaque variable.

    Args:
        df (pandas.DataFrame): Le DataFrame à analyser.
        titre_graph (str): Le titre du graphique à afficher.
    """
    
    # 1. Création des donées : 
    nbr_val_manq = df.isnull().sum().sum()
    nbr_val_tot = df.shape[0]*df.shape[1]
    pourcentage_val_manq = nbr_val_manq / nbr_val_tot * 100
    
    df_val_manq = df.isnull().sum().to_frame('nb_val_manq')
    df_val_manq['%'] = df_val_manq['nb_val_manq'] / len(df) * 100
    df_val_manq = df_val_manq[df_val_manq['%'] > 0].sort_values('%', ascending=False)
    
    # 2. Affichage des résultats : 
    print(f"Nomnbrde valeurs manquantes : {nbr_val_manq} | Nbre de données : {nbr_val_tot} | -----> {round(pourcentage_val_manq,2)} % de valeurs manquantes ")
    print('--'*50)
    
    print(f"Nombre de variables avec des valeurs manquantes : {len(df_val_manq)}/{len(df.columns)}")
    
    if len(df_val_manq) == 0: 
        return

    plt.figure(figsize=(20,8))

    sns.barplot(
        data=df_val_manq,
        x=df_val_manq.index,
        y='%'
    )
    
    plt.title(f"Pourcentage des valeurs manquantes dans {titre_graph}")
    plt.xlabel('Variable')
    plt.xticks(rotation=90)

    plt.show()
        

def detection_encodage_fichier(fichier):
    """
    Cette fonction détecte l'encodage d'un fichier en utilisant la bibliothèque chardet.

    Args:
        fichier (str): Le chemin vers le fichier que vous souhaitez analyser.

    Returns:
        str: L'encodage détecté du fichier.
    """
    with open(fichier, 'rb') as file:
        detection = chardet.detect(file.read())
    return detection['encoding']


def importation_DataFrame(dossier): 
    """
    Cette fonction importe des fichiers CSV depuis un dossier donné, détecte automatiquement l'encodage de chaque fichier, et les stocke dans un dictionnaire de DataFrames.

    Args:
        dossier (str): Le chemin du dossier contenant les fichiers CSV à importer.

    Returns:
        dict: Un dictionnaire contenant les DataFrames importés. Les clés sont les noms de fichiers sans l'extension .csv.
    """

    # 1. Création des données : 
    ls_fichier = [fichier for fichier in os.listdir(dossier) if fichier.endswith('.csv')]
    dataframes = {}

    # 2. Chargement des DataFrames dans le dictionnaire : 
    for nom in ls_fichier:
        chemin = os.path.join(dossier, nom)
        encodage = detection_encodage_fichier(chemin)
            
        try: 
            df = pd.read_csv(chemin, encoding=encodage)
            dataframes[nom[:-4]] = df
        except Exception as e:
            print(f"Erreur lors de l'importation du fichier {nom}: {e}")
        
    # 4. Affichage des résultats d'importation :    
    print("Dataframes importés dans le dictionnaire dataframes :\n")
    for nom in dataframes.keys():
        print(f"- {nom}")

    return dataframes


###########################################################################################################################
# 4. Matrices de corrélation 

def plot_phik_matrix(data, categorical_columns, figsize=(20,20), mask_upper=True, tight_layout=True, linewidth=0.1, fontsize=10, cmap='Blues', show_target_top_corr=True, target_top_columns=10):
    """
    Tracer une matrice de corrélation Phi-K pour les variables catégorielles.

    Args:
        data (DataFrame): Le DataFrame à partir duquel construire la matrice de corrélation.
        categorical_columns (list): Liste des colonnes catégorielles dont les valeurs Phi-K doivent être tracées.
        figsize (tuple, optional): Taille de la figure du graphique. Par défaut, (20, 20).
        mask_upper (bool, optional): Indique si l'on doit afficher uniquement la partie supérieure du graphique. Par défaut, True.
        tight_layout (bool, optional): Indique si le tracé doit utiliser une disposition ajustée. Par défaut, True.
        linewidth (float, optional): Épaisseur de la ligne pour le tracé. Par défaut, 0.1.
        fontsize (int, optional): Taille de police pour les étiquettes des axes X et Y. Par défaut, 10.
        cmap (str, optional): La colormap à utiliser pour le graphique. Par défaut, 'Blues'.
        show_target_top_corr (bool, optional): Indique si l'on doit afficher les variables catégorielles les plus corrélées avec la variable cible. Par défaut, True.
        target_top_columns (int, optional): Le nombre de variables les plus corrélées avec la variable cible à afficher. Par défaut, 10.
    """
    
    # 1. Récupération des colonnes de variables catégorielles : 
    
    data_for_phik = data[categorical_columns].astype('object')
    phik_matrix = data_for_phik.phik_matrix()
    
    print('-'*50)
    
    # 2. Choix de l'affichage supérieur ou inférieur : 
    if mask_upper:
        mask_array = np.ones(phik_matrix.shape)
        mask_array = np.triu(mask_array)
    else:
        mask_array = np.zeros(phik_matrix.shape)
      
    # 3. Tracé de la matrice Phi-K : 
      
    plt.figure(figsize=figsize, tight_layout=tight_layout)
    
    sns.heatmap(phik_matrix, annot=False, mask=mask_array, linewidth=linewidth, cmap=cmap)
    
    plt.title("Matrice de corrélation Phi-K sur les variables catégorielles")
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    
    plt.show()
    
    print("-"*50)
    
    # 4. Affichage des premières colonnes possèdant la corrélation la plus élevée avec la variable cible :         
    if show_target_top_corr:
        
        print("Les catégories possédant les valeurs de corrélation Phi-K les plus élevées avec la variable cible sont : ")
        
        phik_df = pd.DataFrame({'Variables' : phik_matrix.TARGET.index[1:], 'Phik-Correlation' : phik_matrix.TARGET.values[1:]})
        phik_df = phik_df.sort_values(by = 'Phik-Correlation', ascending = False)
        displayhook(phik_df.head(target_top_columns))
        
        print("-"*50)
        
        
def correlation_matrix(data, var_num, figsize=(10,8)):
    """
    Affiche une heatmap de la matrice de corrélation entre les colonnes numériques du DataFrame.

    Args:
        data (DataFrame): Le DataFrame contenant les données.
        var_num (list): Liste des noms des colonnes numériques à inclure dans la matrice de corrélation.
        figsize (tuple, optional): Taille de la figure pour l'affichage. Par défaut, (10, 8).

    """

    # 1. Calcul de la matrice de corrélation : 
    correlation_matrix = data[var_num].corr()

    # 2. Création d'un masque pour masquer la partie supérieure du triangle : 
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # 3. Affichage de la heatmap :
    plt.figure(figsize=figsize)

    sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',
        mask=mask,
        )

    plt.title("Matrice de Corrélation des variables numériques")

    plt.show()      
        
        
def top_correlation_var_num_target(data, var_num, target, n_top=10):
    """ 
    Affiche les principales corrélations de la variable cible avec les variables numériques.

    Args:
        data (DataFrame): Le DataFrame contenant les données.
        var_num (list): Liste des noms des colonnes numériques à inclure dans l'analyse de corrélation.
        target (str): Le nom de la variable cible (catégorielle).
        n_top (int, optional): Le nombre de meilleures corrélations à afficher. Par défaut, 10.

    """
    
    # 1. Sélection des colonnes numériques et de la colonne de la variable cible : 
    numeric_df = data[var_num + [target]]

    # 2. Calcul de la matrice de corrélation Phi-K : 
    corr_matrix = numeric_df.phik_matrix()

    # 3. Récupération de la colonne de corrélation de TARGET :
    corr_target = corr_matrix[target].drop(target)

    # 4. Création d'un DataFrame :
    corr_df = pd.DataFrame({'Variable': corr_target.index, 'Phik-Correlation': corr_target.values}).sort_values('Phik-Correlation', ascending=False).head(n_top)

    # 5. Affichage :
    print("Les variables numériques possèdant les valeurs de corrélation Phik les plus élevées avec la variable cible sont : ")
    displayhook(corr_df)
    
    
###########################################################################################################################
# 5. Analyse univariée   

def type_donnees_uniques(data, col):
    """
    Affiche des informations sur les données uniques dans une colonne spécifique.

    Cette fonction permet d'obtenir des informations sur les données uniques dans une colonne donnée d'un DataFrame. Elle affiche
    le nombre de valeurs uniques, le pourcentage de valeurs manquantes (NaN), et la liste des valeurs uniques dans la colonne.

    Args:
        data (DataFrame): Le DataFrame contenant les données.
        col (str): Le nom de la colonne à analyser.
    """
    
    # 1. Récupération de la colonne : 
    serie = data[col].unique()
    
    # 2. Calcul du pourcentage de nan : 
    pourcentage_nan = data[col].isnull().sum() / len(data[col]) * 100 
    
    # 2. Affichage : 
    print('--'*50)
    print(f"Nombre de valeur unique de la variable {col} : {len(serie)}")
    print('--'*50)
    print(f"Pourcentage de valeurs manquantes : {round(pourcentage_nan, 2)} %")
    print('--'*50)
    print("Valeurs uniques : \n")
    
    for i in serie: 
        print(f"\t-{i}") 
   
    
def repartition_var_categorielle(data, var_categorielle, figsize=(18, 6)):
    """
    Affiche la répartition d'une variable catégorielle et le pourcentage de défaillants dans chaque catégorie.

    Cette fonction génère deux graphiques barplot pour analyser la répartition d'une variable catégorielle dans un DataFrame.
    Le premier graphique affiche le nombre d'occurrences de chaque catégorie, tandis que le deuxième graphique affiche le
    pourcentage de défaillants (TARGET = 1) dans chaque catégorie.

    Args:
        data (DataFrame): Le DataFrame contenant les données.
        var_categorielle (str): Le nom de la variable catégorielle à analyser.
        figsize (tuple, optional): La taille de la figure pour afficher les graphiques. Par défaut, (18, 6).

    """
    
    # 1. Calcul du nombre d'occurrences total: 
    data_to_plot = data[var_categorielle].value_counts().sort_values(ascending=False).to_frame('Nombre').reset_index()
    
    # 2. Calcul du % de defaillants dans chaque catégories : 
    pourcentage_defaillants_par_categories = data.loc[data['TARGET'] == 1, var_categorielle].value_counts() / data[var_categorielle].value_counts()  * 100
    pourcentage_defaillants_par_categories = pourcentage_defaillants_par_categories.sort_values(ascending=False).to_frame('%').reset_index()
    
    # 3. Création des figures : 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 4. Graphique_1 : 
    sns.barplot(
        data=data_to_plot, 
        x=var_categorielle,
        y='Nombre',
        palette='hls',
        ax=ax1,
    )
    
    for p in ax1.patches: 
        ax1.annotate(
            text=f"{round(p.get_height() / data_to_plot['Nombre'].sum() * 100, 2)}%",
            xy=(p.get_x(), p.get_height()),
            xytext=(-p.get_width()/2, 4),
            textcoords='offset points',
            fontsize='x-small',
        )
    
    ax1.set_title('Toutes TARGET')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    
    # 5. Graphique_2 : 
    
    sns.barplot(
        data=pourcentage_defaillants_par_categories,
        x=var_categorielle,
        y='%',
        palette='hls',
        ax=ax2
        )

    for p in ax2.patches: 
        ax2.annotate(
            text=f"{round(p.get_height(), 2)}%",
            xy=(p.get_x(), p.get_height()),
            xytext=(-p.get_width()/2, 4),
            textcoords='offset points',
            fontsize='x-small',
        )

    ax2.set_title('TARGET = 1 --> Défaillants')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

    # 6. Affichage des graphique
    plt.suptitle(f'Répartition de {var_categorielle}', fontsize=20)
    plt.show() 


def graph_variable_continue(data, column_name, plots=['kde', 'box', 'violin'], valeurs_limites=None, figsize=(20, 9), log_scale=False, palette=['SteelBlue', 'Crimson']):
    """
    Affiche des graphiques pour explorer une variable continue.

    Cette fonction génère plusieurs types de graphiques pour explorer une variable continue dans un DataFrame.
    Elle permet de visualiser la distribution de probabilité (kdeplot), les boîtes à moustaches (boxplot) et les graphiques en violon (violinplot)
    pour les défaillants (TARGET = 1) et les non-défaillants (TARGET = 0).
    Il est également possible de spécifier des limites pour filtrer les données si nécessaire.

    Args:
        data (DataFrame): Le DataFrame contenant les données.
        column_name (str): Le nom de la variable continue à explorer.
        plots (list, optional): Les types de graphiques à générer parmi ['kde', 'box', 'violin']. Par défaut, les trois types
            de graphiques sont générés. 
        valeurs_limites (tuple, optional): Les limites pour filtrer les données de la variable continue. Par défaut, None (pas de filtrage).
        figsize (tuple, optional): La taille de la figure pour afficher les graphiques. Par défaut, (20, 9).
        log_scale (bool, optional): Si True, utilise une échelle logarithmique pour l'axe des x (pour kdeplot) et/ou l'axe des y
            (pour boxplot et violinplot). Par défaut, False.
        palette (list, optional): La palette de couleurs à utiliser pour les graphiques. Par défaut, ['SteelBlue', 'Crimson'].

    """
    
    # 1. Copie du DataFrame : 
    data_to_plot = data.copy()
    
    # 2. Filtres des données en cas de valeurs aberrantes : 
    if valeurs_limites:
        mask_1 = data_to_plot[column_name] > valeurs_limites[0]
        mask_2 = data_to_plot[column_name] < valeurs_limites[1]
        
        data_to_plot = data_to_plot.loc[mask_1 & mask_2]
    
    # 3. Création de la figure des graphiques :     
    number_of_subplots = len(plots)
    plt.figure(figsize=figsize)

    # 4. Création des graphiques en boucle : 
    
    for i, ele in enumerate(plots):
        
        plt.subplot(1, number_of_subplots, i + 1)
        plt.subplots_adjust(wspace=0.25)

        if ele == 'kde':
            
            sns.kdeplot(
                data=data_to_plot[column_name][data['TARGET'] == 0].dropna(),
                label='Non Défaillants',
                color=palette[0],
                )
             
            sns.kdeplot(
                data=data_to_plot[column_name][data['TARGET'] == 1].dropna(),
                label='Défaillants',
                color=palette[1],
                )
             
            
            plt.xlabel(column_name)
            plt.ylabel('Distribution de probabilité')            
            plt.title(f"Distribution de Probabilité de {column_name}")
            plt.legend() 
            
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)')

        if ele == 'violin':
            
            sns.violinplot(
                x='TARGET',
                y=column_name,
                data=data_to_plot,
                palette=palette,
                )
            
            plt.title(f"Violin-Plot de {column_name}")
            
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

        if ele == 'box':
            
            sns.boxplot(
                x='TARGET',
                y=column_name,
                data=data_to_plot,
                palette=palette,
                )
            
            plt.title(f"Box-Plot de {column_name}")
            
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')
        
    # 5. Affichage des graphiques :         
    plt.suptitle(f'Répartition de {column_name}', fontsize=20)
    plt.show()

      
###########################################################################################################################
# 6. Regroupement des DataFrames 

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
# 7. Features importance 

def features_importance_lightgbm(df, num_folds, class_weight=None): 
    
    # 1. Séparation de df en semalbe test / train en fonction de la cible : 
    train = df.loc[df['TARGET'].notna()]
    test = df.loc[df['TARGET'].isna()]
    
    print(f"Dimension de train : {train.shape}")
    print(f"Dimension de test : {test.shape}")
    
    del df 
    
    # 2. Création de la validation croisée : 
    folds = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=42,
        )
    
    # 3. Création de X et y à partir de train :
    features = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    X = train[features]
    y = train['TARGET']

    # 4. Création des données de staockage : 
    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()

    # 4. Boucle sur les différents folds : 
     
    for n_fold, (train_index, validation_index) in enumerate(folds.split(X, y)): 
         
        # 4.1 Création des données d'entrainement et de validation : 
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_val = X.iloc[validation_index]
        y_val = y.iloc[validation_index]
        
        # 4.2 Création du modèle LightGBM (paramètres fournis par Kernel Kaggle, trouvé par optimisation bayésienne)
        model = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            verbose=-1,
            class_weight=class_weight, # Ajout de la gestion du déséquilibre des classes
            )
        
        # 4.3 Entrainement du modèle : 
        model.fit(
            X_train,
            y_train, 
            eval_set=(X_val, y_val), 
            eval_metric='auc',
            early_stopping_rounds=200,
            verbose=200
        )
        
        # 4.4 Prédiction des probabilités de l'appartenance à la classe 1 sur la validation : 
        oof_preds[validation_index] = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
        
        # 4.5 Prédiction des probabilités de l'appartenance à la classe 1 sur le test (divisé par le nombre de folds pour obtenir la moyenne finale sur l'ensemble de folds): 
        sub_preds += model.predict_proba(test[features], num_iteration=model.best_iteration_)[:, 1] / folds.n_splits
        
        # 4.6 Complétion de feature_importance_df : 
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        # 4.7 Affichage du résultat sur le fold en cours : 
        print(f"Fold {int(n_fold + 1)}, score AUC : {round(roc_auc_score(y_val, oof_preds[validation_index]),6)}")
        
    # 5. Affichage du résultats total : 
    print(f"AUC score total : {round(roc_auc_score(y, oof_preds),6)}")

    return feature_importance_df
      
      
###########################################################################################################################
# 8. Pré-traitement DataFrame     

def pre_process_dataframe(df, fillna_strategy):
    """
    Prétraitement des données d'entraînement et de test.

    Args:
        df (pandas.DataFrame): Le DataFrame contenant les données d'entraînement et de test.
        fillna_strategy (str): La stratégie d'imputation pour les valeurs manquantes.
            Doit être l'un des suivants : 'median', 'mean', 'mode'.

    Raises:
        ValueError: Si fillna_strategy n'est pas 'median', 'mean' ou 'mode'.

    Returns:
        pandas.DataFrame, pandas.DataFrame: Deux DataFrames, l'un pour les données d'entraînement prétraitées,
        l'autre pour les données de test prétraitées.
    """
    
    # 1. Vérification de fillna_strategy : 
    if fillna_strategy not in ['median', 'mean', 'mode']:
        raise ValueError("fillna_strategy doit être égal à 'median', 'mean' ou 'mode'.")
    
    # 2. Création de l'imputer : 
    if fillna_strategy == 'mode':
        imputer = SimpleImputer(strategy='most_frequent')
    else:
        imputer = SimpleImputer(strategy=fillna_strategy)
        
    # 3. Séparation en train et test : 
    df_train = df.loc[df['TARGET'].notna()].reset_index(drop=True)
    df_test = df.loc[df['TARGET'].isna()].reset_index(drop=True)
    
    # 4. Récupération de la colonne TARGET : 
    target = df_train['TARGET'].copy()
    target = target.astype('int8')
    
    df_train = df_train.drop('TARGET', axis=1)
    df_test = df_test.drop('TARGET', axis=1)
   
    # 5. Gestion des valeurs infinies : 
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
   
    # 6. Imputation des valeurs manquantes : 
    df_train_imputed = pd.DataFrame(imputer.fit_transform(df_train), columns=df_train.columns)
    df_test_imputed = pd.DataFrame(imputer.transform(df_test), columns=df_test.columns)
   
    # 7. Conversion des colonnes booléennes en int8 :
    bool_columns = df_train.select_dtypes(include=['bool']).columns
   
    df_train_imputed[bool_columns] = df_train_imputed[bool_columns].astype('int8')
    df_test_imputed[bool_columns] = df_test_imputed[bool_columns].astype('int8')
   
    # 8. Conversion des colonnes float64 en float32 :
    float_columns = df_train.select_dtypes(include=['float64']).columns
   
    df_train_imputed[float_columns] = df_train_imputed[float_columns].astype('float32')
    df_test_imputed[float_columns] = df_test_imputed[float_columns].astype('float32')
   
    # 9. Conversion des colonnes int64 en int8 :
    int_columns = df_train.select_dtypes(include=['int64']).columns
   
    df_train_imputed[int_columns] = df_train_imputed[int_columns].astype('int8')
    df_test_imputed[int_columns] = df_test_imputed[int_columns].astype('int8')
   
    # 10. Ajout de la colonne TARGET à df_train : 
    df_train_imputed['TARGET'] = target
    
    # 11. Affichage : 
    print('**'*50)
    print(f"\nTRAIN\n"
          f"Nombre de colonnes avec au moins une valeur manquante : {df_train_imputed.isna().any().sum()}"
          f"\ntype de données : {df_train_imputed.dtypes.unique().tolist()}\n"
          )
    print('**'*50)
    
    print('**'*50)
    print(f"\nTEST\n"
          f"Nombre de colonnes avec au moins une valeur manquante : {df_test_imputed.isna().any().sum()}"
          f"\ntype de données : {df_test_imputed.dtypes.unique().tolist()}\n"
          )
    print('**'*50)
    
    return df_train_imputed, df_test_imputed

