
import streamlit as st
import pandas as pd
from supabase import create_client, Client
import numpy as np
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
from pandas import Timestamp
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, roc_auc_score

def soustraire_dates(df, colonne1, colonne2, nouvelle_colonne):
    try:
        # Vérifier si les colonnes contiennent des valeurs NaN
        masque = ~(df[colonne1].isna() | df[colonne2].isna())

        # Soustraire colonne2 de colonne1 pour les lignes sans NaN et convertir en jours
        df.loc[masque, nouvelle_colonne] = ((df.loc[masque, colonne1] - df.loc[masque, colonne2]) / np.timedelta64(1, 'D')).astype(int)

    except TypeError:
        print("non")

    return df

from dotenv import load_dotenv
load_dotenv() 

supabase_key = os.getenv("SUPABASE_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase: Client = create_client(supabase_url,supabase_key)

def fetch_data_from_supabase(table_name):
    """
    Récupère les données d'une table Supabase et les convertit en DataFrame.

    Args:
    - table_name (str): Le nom de la table à récupérer.

    Returns:
    - pd.DataFrame: Les données de la table sous forme de DataFrame.
    - None: Si aucune donnée n'est trouvée ou en cas d'erreur.
    """
    try:
        response = supabase.table(table_name).select('*').order("id", desc=True).limit(1000).execute()
        if response.data:
            return pd.DataFrame(response.data)
        else:
            st.write(f"Aucune donnée trouvée dans la table '{table_name}'.")
            return None
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la récupération des données de la table '{table_name}' : {e}")
        return None

def label_to_classes(due_days):
   
    if due_days <0:
        return 0
    elif due_days == 0 :
        return 1
    elif due_days >= 1 and due_days <= 60:
        return 2
    else :
        return 3


    
def retard_facture(row):
    # Exemple de logique de prédiction: remplacer par votre propre modèle
    if pd.isnull(row['type_retard']):
        return ''
    elif row['type_retard']== 0:
        return 'En avance'
    elif row['type_retard'] == 1:
        return 'à temps'
    else:
        return 'En retard'    
    
def map_month_to_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'
    
def compute_historique_retards(df_with_date_op, df_without_date_op):
    # Triez le DataFrame avec les dates d'opération
    df_with_date_op = df_with_date_op.sort_values(by=['codetiers', 'date_echeance'])
    df_with_date_op['Historique_retards'] = 0
    df_without_date_op['Historique_retards'] = 0
    historique = {}
    
    # Calcul de l'historique des retards pour les lignes avec des dates d'opération
    for idx, row in df_with_date_op.iterrows():
        client = row['codetiers']
        if client not in historique:
            historique[client] = []
        
        # Calculer le nombre de retards passés avant cette facture
        historique_retards = sum(historique[client])
        df_with_date_op.at[idx, 'Historique_retards'] = historique_retards
        
        # Ajouter le statut de retard actuel à l'historique
        historique[client].append(row['type_retard'])

    # Copier la colonne d'historique des retards pour les lignes sans dates d'opération
    df_without_date_op['Historique_retards'] = df_without_date_op.groupby('codetiers')['Historique_retards'].transform('first')
    
    return df_without_date_op

def prepross(df):
    df['date_echeance']=  pd.to_datetime(df['date_echeance'])
    # Appliquez la fonction pour créer la caractéristique "Season" (Saison)
    df['Season'] = df['date_echeance'].dt.month.apply(map_month_to_season)
    # Créez une instance de OneHotEncoder
    encoder = OneHotEncoder()

    # Ajustez l'encodeur aux valeurs uniques de la colonne "Season"
    encoder.fit(df['Season'].values.reshape(-1, 1))
    # Transformez la colonne "Season" en one-hot encoding
    encoded_season = encoder.transform(df['Season'].values.reshape(-1, 1))
    # Créez un DataFrame à partir des valeurs encodées
    df_encoded = pd.DataFrame(encoded_season.toarray(), columns=encoder.get_feature_names_out(['Season']))
    # Réinitialisez l'index du DataFrame d'origine
    df.reset_index(drop=True, inplace=True)
    # Concaténez le DataFrame encodé avec le DataFrame d'origine
    df = pd.concat([df, df_encoded], axis=1)
    # Supprimer la colonne "Season" d'origine
    df.drop(columns=['Season'], inplace=True)
    #date_reference = Timestamp(datetime.now().date())
    date_reference = pd.to_datetime('2023-12-31')
    df['Duree_avant_echeance'] = (df['date_echeance'] - date_reference).dt.days
    return df 

def load_model(cluster_num):
    try:
        model_path = f'random_forest_cluster_{cluster_num}.pkl'
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"Model file not found for cluster {cluster_num}")
        return None

def make_predictions(df):
    predictions = []
    
    # Renommer les colonnes si nécessaire
    df.rename(columns={
        'montant_fac': 'Montant fac',
        'idevt': 'Idevt',
        'unpaidinvoicecount': 'UnpaidInvoiceCount'
    }, inplace=True, errors='raise')
    
    for index, row in df.iterrows():
        
        cluster = row['Cluster']
        model = load_model(cluster)
        if not pd.isna(cluster):
            if model is not None:
                try:
                    # Assurez-vous que les noms des colonnes sont exactement les mêmes que ceux utilisés lors de l'entraînement
                    feature_names = ['Cluster', 'Montant fac', 'Idevt', 'litige', 'UnpaidInvoiceCount', 'Duree_avant_echeance', 'Historique_retards', 'Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter']
                    features = pd.DataFrame([row[feature_names]], columns=feature_names)
                    
                    prediction = model.predict(features)
                    predictions.append(prediction[0])
                
                except Exception as e:
                    st.write(f"Error predicting for cluster {cluster}: {e}")
                    predictions.append(None)
            else:
                predictions.append(None)
        else:
            predictions.append(None)
    return predictions

    

def color_pred(val):
    color = ''
    if val == 3:
        color = 'red'
    elif val == 2:
        color = 'orange'
    elif val == 0:
        color = 'blue'
    elif val == 1:
        color = 'green'
    return f'background-color: {color}'

def save_predictions_to_supabase(predictions_df):
    # Convertir les colonnes 'id' et 'idevt' en type int
        if 'id' in predictions_df.columns:
            predictions_df['id'] = predictions_df['id'].astype(int)
        if 'idevt' in predictions_df.columns:
            predictions_df['idevt'] = predictions_df['idevt'].astype(int)

        # Récupérer le dernier 'id' inséré dans la table
        last_id = 0
        try:
            response = supabase.table('predictions').select('id').order('id', desc=True).limit(1).execute()
            if response.data:
                last_id = response.data[0]['id']
            else:
                last_id = 0
        except:
            non ='boo'
        # Générer de nouveaux IDs uniques pour les nouvelles lignes
        if 'id' in predictions_df.columns:
            predictions_df['id'] = range(last_id + 1, last_id + 1 + len(predictions_df))

        # Vérifier l'existence de chaque ID et générer de nouveaux IDs uniques si nécessaire
        existing_ids = set()
        for _, row in predictions_df.iterrows():
            id_value = row['id']
            while id_value in existing_ids:
                id_value += 1
            existing_ids.add(id_value)
            row['id'] = id_value

        colonnes_necessaires = ["id","num_fac","prediction"]
        data_to_insert = predictions_df[colonnes_necessaires].to_dict(orient='records')

        # Insertion des données dans la table appropriée
        try:
            response = supabase.table('prediction').insert(data_to_insert).execute()
        except:
            non ="je pleure"

    

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de confusion')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

# Courbe ROC
def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    st.pyplot()

# Courbe de précision-rappel
def plot_precision_recall_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue')
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title('Courbe de précision-rappel')
    st.pyplot()

# Taux d'exactitude
def calculate_accuracy(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    st.write(f'Taux d\'exactitude : {accuracy}')

# Distribution des prédictions
def plot_prediction_distribution( y_pred):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_pred)
    plt.xlabel('Prédictions')
    plt.ylabel('Nombre d\'occurrences')
    plt.title('Distribution des prédictions')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)