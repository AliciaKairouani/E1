import os
import pandas as pd
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer les variables d'environnement
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_API_KEY")

# Créer un client Supabase
supabase: Client = create_client(supabase_url, supabase_key)

# Configuration de l'interface de téléversement de fichiers
st.title("Upload your CSV file")

uploaded_file = st.file_uploader("Télécharger un fichier CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Afficher le DataFrame
    st.write("l'upload a réussi")

    # Définir les colonnes nécessaires pour chaque table
    colonnes_facture = ['id', 'codetiers','idevt','num_fac','montant_fac','litige','date_echeance']  # Remplacez par les noms réels des colonnes pour la table 'facture'
    colonnes_virement = ['idevt', 'id','montant_vir','date_op','codetiers']  # Remplacez par les noms réels des colonnes pour la table 'virement'

    # Vérifier si les colonnes nécessaires sont présentes
    if all(col in df.columns for col in colonnes_facture):
        table = 'facture'
        colonnes_necessaires = colonnes_facture
    elif all(col in df.columns for col in colonnes_virement):
        table = 'virement'
        colonnes_necessaires = colonnes_virement
    else:
        st.error("Les colonnes nécessaires ne sont pas présentes dans le fichier CSV.")
        table = None

    # Si les colonnes nécessaires sont présentes, préparer les données pour l'insertion
    if table:
        # Convertir les colonnes 'id' et 'idevt' en type int
        if 'id' in df.columns:
            df['id'] = df['id'].astype(int)
        if 'idevt' in df.columns:
            df['idevt'] = df['idevt'].astype(int)

        # Récupérer le dernier 'id' inséré dans la table
        last_id = 0
        try:
            response = supabase.table(table).select('id').order('id', desc=True).limit(1).execute()
            if response.data:
                last_id = response.data[0]['id']
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la récupération du dernier 'id' : {e}")

        # Générer de nouveaux IDs uniques pour les nouvelles lignes
        if 'id' in df.columns:
            df['id'] = range(last_id + 1, last_id + 1 + len(df))

        # Vérifier l'existence de chaque ID et générer de nouveaux IDs uniques si nécessaire
        existing_ids = set()
        for _, row in df.iterrows():
            id_value = row['id']
            while id_value in existing_ids:
                id_value += 1
            existing_ids.add(id_value)
            row['id'] = id_value

        data_to_insert = df[colonnes_necessaires].to_dict(orient='records')

        # Insertion des données dans la table appropriée
    try:
        response = supabase.table(table).insert(data_to_insert).execute()
        st.success(f"Les données ont été insérées avec succès dans la table {table}.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de l'insertion des données : {e}")


