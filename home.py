import streamlit as st
import pandas as pd
from supabase import create_client, Client
import numpy as np
import os
import pickle
from back import *

st.title("Tableau de bord")

# Bouton de rafraîchissement pour obtenir les dernières données de la base de données
if st.button("Refresh"):
    st.experimental_rerun()

facture = fetch_data_from_supabase('facture')
virement = fetch_data_from_supabase('virement')


if facture is not None and virement is not None:
    # Concaténer les DataFrames sur les colonnes communes
    df_merge = pd.merge(facture, virement, on=['id', 'codetiers', 'idevt'], how='outer')
    df_merge = df_merge[df_merge['montant_fac'].notna()]
else:
    st.error("Erreur dans la récupération des données.")

if df_merge is not None:
    # Filtrer les lignes avec des valeurs non nulles dans 'date_op'
    df_with_date_op = df_merge[df_merge['date_op'].notna()]
    # Filtrer les lignes avec des valeurs nulles dans 'date_op'
    df_without_date_op = df_merge[df_merge['date_op'].isna()]

    # Appliquer la logique pour les lignes avec des valeurs non nulles dans 'date_op'
    if not df_with_date_op.empty:
        df_with_date_op['date_op'] = pd.to_datetime(df_with_date_op['date_op'])
        df_with_date_op['date_echeance'] = pd.to_datetime(df_with_date_op['date_echeance'])
        df_with_date_op = soustraire_dates(df_with_date_op, "date_op", "date_echeance", "jours_retard")
        df_with_date_op["jours_retard"] = df_with_date_op["jours_retard"].astype(str).str.extract('(-?\d+)').astype(float)
        df_with_date_op["type_retard"] = df_with_date_op["jours_retard"].apply(lambda x: label_to_classes(x))
        df_with_date_op['Etat'] = df_with_date_op.apply(retard_facture, axis=1)
        df_with_date_op['prediction']= ''
        df_final = df_with_date_op

    # Appliquer la logique pour les lignes avec des valeurs nulles dans 'date_op'
    if not df_without_date_op.empty:
        df_pred = compute_historique_retards(df_with_date_op, df_without_date_op)
        df_pred = prepross(df_pred)
        df_clusters = pd.read_csv('cluster.csv') 
        df_clusters.rename(columns={'CodeTiers':'codetiers'}, inplace=True, errors='raise')
        df_clusters['codetiers'] = df_clusters['codetiers'].astype('int')
        df_clusters['codetiers'] = df_clusters['codetiers'].astype(str)
        df_pred_cluster = pd.merge(df_pred, df_clusters, on='codetiers', how='left')
        df_pred_cluster['unpaidinvoicecount'] = df_pred_cluster['unpaidinvoicecount'].fillna(0)
        df_pred_cluster['prediction'] = make_predictions(df_pred_cluster)
        df_merge['date_op'] = pd.to_datetime(df_merge['date_op'])
        df_merge['date_echeance'] = pd.to_datetime(df_merge['date_echeance'])
        df_merge = soustraire_dates(df_merge, "date_op", "date_echeance", "jours_retard")
        df_merge["jours_retard"] = df_merge["jours_retard"].astype(str).str.extract('(-?\d+)').astype(float)
        df_merge["type_retard"] = df_merge["jours_retard"].apply(lambda x: label_to_classes(x))
        df_merge['Etat'] = df_merge.apply(retard_facture, axis=1)
        df_final = pd.merge(df_merge, df_pred_cluster[['id', 'codetiers', 'prediction']], on=['id', 'codetiers'], how='left')
        df_final['prediction'] = df_final.apply(lambda row: row['prediction'] if pd.isna(row['date_op']) else '', axis=1)
        predictions_df = df_final[["id","num_fac","prediction"]]
        save_predictions_to_supabase(predictions_df)

df_final['litige'] = df_final['litige'].fillna(0)
df_final['litige'] = df_final['litige'].astype(int)
df_final['montant_fac'] = df_final['montant_fac'].round(2)
df_final['montant_fac'] = df_final['montant_fac'].map('{:.1f}'.format)

df_affichage = df_final[["codetiers","num_fac","montant_fac","date_echeance","litige","Etat","prediction"]]
df_affichage.rename(columns={'codetiers':'Code Tiers','num_fac': 'Numéro de Facture','montant_fac': 'Montant de la Facture','date_echeance': 'Date écheance'},inplace=True, errors='raise')
styled_df = df_affichage.style.applymap(color_pred, subset=['prediction'])


# Ajouter une sélection de code tiers dans la barre latérale
options = ['All'] + df_affichage['Code Tiers'].unique().tolist()
add_selectbox = st.sidebar.selectbox('Tiers', options)

# Filtrer le DataFrame en fonction de la sélection de l'utilisateur
if add_selectbox:
    if add_selectbox == 'All':
        filtered_df = df_affichage
    else:
        filtered_df = df_affichage[df_affichage['Code Tiers'].str.contains(add_selectbox, case=False, na=False)]
        
# Appliquer le style uniquement à l'affichage
st.dataframe(filtered_df.tail(40).style.applymap(color_pred, subset=['prediction']).set_properties(**{
        'text-align': 'center',
        'font-size': '14px'
}).set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#add8e6'), ('color', '#fff')]
}]))




