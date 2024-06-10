import streamlit as st
import pandas as pd
from streamlit_searchbox import st_searchbox
import plotly.express as px
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import os
from supabase import create_client, Client

st.markdown("<h1 style='text-align: center; padding: 40px;'>Détail Clients </h1>", unsafe_allow_html=True)

def soustraire_dates(df, colonne1, colonne2, nouvelle_colonne):
    try:
        # Vérifier si les colonnes contiennent des valeurs NaN
        masque = ~(df[colonne1].isna() | df[colonne2].isna())

        # Soustraire colonne2 de colonne1 pour les lignes sans NaN et convertir en jours
        df.loc[masque, nouvelle_colonne] = ((df.loc[masque, colonne1] - df.loc[masque, colonne2]) / np.timedelta64(1, 'D')).astype(int)

    except TypeError:
        print("non")

    return df


supabase_key = os.getenv("SUPABASE_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")

supabase: Client = create_client(supabase_url,supabase_key)
try:
    # Exécuter une requête pour récupérer des données de la table 'facture'
    response = supabase.table('facture').select('*').execute()
    response2 = supabase.table('virement').select('*').execute()
    response3 = supabase.table('litige').select('*').execute()
    # Convertir les données en DataFrame
    if response.data:
        facture = pd.DataFrame(response.data)
    else:
        st.write("Aucune donnée trouvée dans la table 'facture'.")
    if response2.data:
        virement = pd.DataFrame(response2.data)
    else:
        st.write("Aucune donnée trouvée dans la table 'virement'.")
    if response3.data:
        litige = pd.DataFrame(response3.data)
    else:
        st.write("Aucune donnée trouvée dans la table 'virement'.")
except Exception as e:
    st.error(f"Une erreur est survenue lors de la récupération des données : {e}")


df_merge = pd.merge(facture,virement, on=["id", "codetiers","idevt"])
df_merge['date_op'] = pd.to_datetime(df_merge['date_op'])
df_merge['date_echeance'] = pd.to_datetime(df_merge['date_echeance'])


df_merge  = soustraire_dates(df_merge, "date_op", "date_echeance", "jours_retard")
df_merge["jours_retard"] = df_merge["jours_retard"].astype(str).str.extract('(-?\d+)').astype(float)

add_selectbox = st.sidebar.selectbox('Tiers', df_merge['codetiers'].unique().tolist())


if add_selectbox:
    # Filtrer le DataFrame en fonction de la valeur saisie
    filtered_df = df_merge[df_merge['codetiers'].str.contains(add_selectbox, case=False, na=False)]
    filtered_litige = litige[litige['codetiers'].str.contains(add_selectbox, case=False, na=False)]
   # Vérifier si le DataFrame filtré est vide
    if filtered_df.empty:
        st.error("Aucun client trouvé")
    else:
        # quelque KPI
        total_factures = filtered_df['num_fac'].nunique()

        # Calculer le nombre de factures payées et non payées
        filtered_df['facture_payee'] = np.where(filtered_df['montant_vir'].notnull(), 'Payée', 'Non payée')
        factures_payees = filtered_df[filtered_df['facture_payee'] == 'Payée']['num_fac'].nunique()
        factures_non_payees = filtered_df[filtered_df['facture_payee'] == 'Non payée']['num_fac'].nunique()

        # Calculer le nombre de factures en retard et le retard moyen
        factures_en_retard = filtered_df[filtered_df['jours_retard'] > 0]['num_fac'].nunique()
        retard_moyen = filtered_df[filtered_df['jours_retard'] > 0]['jours_retard'].mean()
        retard_moyen_arrondi = round(retard_moyen, 1)
        # Afficher les KPI dans l'application Streamlit
        st.subheader("KPI")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Factures totales", total_factures)
        with col2:
            st.metric("Factures payées", factures_payees)
        with col3:
            st.metric("Factures non payées", factures_non_payees)
        with col4:
            st.metric("Factures en retard", factures_en_retard)
        st.metric("Retard moyen (jours)", retard_moyen_arrondi)

        # Créer les figures
        #Fig 1
        fig1 = px.histogram(filtered_df, x="montant_fac", color="codetiers", hover_data=["montant_fac"])
        fig1.update_layout(title_text="Montants de facture", xaxis_title="Montant des factures", yaxis_title="")


        # Fig 3
        df_monthly = filtered_df[filtered_df['montant_vir'] > 0].groupby(pd.Grouper(key='date_op', freq='M')).sum('montant_vir').reset_index()
        fig3 = alt.Chart(df_monthly).mark_line().encode(
        x='date_op:T',
        y='montant_vir:Q',
        tooltip=['date_op:T', 'montant_vir:Q']
        )

        # fig 4 
        fig4 = px.histogram(filtered_df, x='jours_retard', color='codetiers')
        fig4.update_layout(
        title='Répartition des factures en fonction du nombre de jours de retard',
        xaxis_title='Nombre de jours de retard',
        yaxis_title='Nombre de factures'
        )
        st.plotly_chart(fig1)
        # Afficher les figures en 2x2
        st.markdown("Montant des virement en fonction des mois")
        st.altair_chart(fig3, use_container_width=True)
        
        st.plotly_chart(fig4)

        # Calculer la solution la plus utilisée
        
        solution_counts = filtered_litige[filtered_litige['protocollevelname'].notna()]['protocollevelname'].value_counts()
        solution_most_used = solution_counts.index[0] if len(solution_counts) > 0 else 'Aucune solution préférée'
       
        # Afficher le KPI
        st.markdown("#### Solution de recouvrement la plus utilisée")
        st.write(f"**{solution_most_used}**")

     
        # Supprimer les valeurs NaN et calculer le mail le plus utilisé
        dunningsenttoemail_counts = filtered_litige['dunningsenttoemail'].dropna().value_counts()
        mail_most_used = dunningsenttoemail_counts.index[0] if len(dunningsenttoemail_counts) > 0 else 'Aucun mail préféré'

        # Afficher le KPI
        st.markdown("#### Mail le plus utilisé")
        st.write(f"**{mail_most_used}**")
else:
    # quelque KPI
        total_factures = df_merge['num_fac'].nunique()

        # Calculer le nombre de factures payées et non payées
        df_merge['facture_payee'] = np.where(df_merge['montant_vir'].notnull(), 'Payée', 'Non payée')
        factures_payees = df_merge[df_merge['facture_payee'] == 'Payée']['num_fac'].nunique()
        factures_non_payees = df_merge[df_merge['facture_payee'] == 'Non payée']['num_fac'].nunique()

        # Calculer le nombre de factures en retard et le retard moyen
        factures_en_retard = df_merge[df_merge['jours_retard'] > 0]['num_fac'].nunique()
        retard_moyen = df_merge[df_merge['jours_retard'] > 0]['jours_retard'].mean()
        retard_moyen_arrondi = round(retard_moyen, 1)
        # Afficher les KPI dans l'application Streamlit
        st.subheader("KPI")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Factures totales", total_factures)
        with col2:
            st.metric("Factures payées", factures_payees)
        with col3:
            st.metric("Factures non payées", factures_non_payees)
        with col4:
            st.metric("Factures en retard", factures_en_retard)
        st.metric("Retard moyen (jours)", retard_moyen_arrondi)

        # Créer les figures
        #Fig 1
        fig1 = px.histogram(df_merge, x="montant_fac", y="codetiers", color="codetiers", hover_data=["montant_fac"], barmode="group")
        fig1.update_layout(title_text="Montants de facture", xaxis_title="Montant des factures", yaxis_title="")
        
        #Fig 2
        df_merge['jours_retard'] = pd.to_numeric(df_merge['jours_retard'])
        
        df_monthly = df_merge.groupby(pd.Grouper(key='date_op', freq='M'))['jours_retard'].mean().reset_index()
        df_monthly['date_op'] = df_merge['date_op'].dt.strftime('%Y-%m')
        fig2 = px.line(df_monthly, x="date_op", y="jours_retard", title="Tendance des jours de retard")
        fig2.update_layout(xaxis_title="Mois", yaxis_title="Moyenne des jours de retard")

        # Fig 3
        df_monthly = df_merge[df_merge['montant_vir'] > 0].groupby(pd.Grouper(key='date_op', freq='M')).sum('montant_vir').reset_index()
        fig3 = alt.Chart(df_monthly).mark_line().encode(
        x='date_op:T',
        y='montant_vir:Q',
        tooltip=['date_op:T', 'montant_vir:Q']
        )

        # fig 4 
        fig4 = px.histogram(df_merge, x='jours_retard', color='codetiers', barmode='group')
        fig4.update_layout(
        title='Répartition des factures en fonction du nombre de jours de retard',
        xaxis_title='Nombre de jours de retard',
        yaxis_title='Nombre de factures'
        )

        # Afficher les figures en 2x2
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1)
            st.altair_chart(fig3, use_container_width=True)
        with col2:
            st.plotly_chart(fig2)
            st.plotly_chart(fig4)

        # Calculer le top 3 des solutions les plus utilisées
        solution_counts = litige['protocollevelname'].value_counts()[:3]
        top_3_solutions = ', '.join(solution_counts.index.tolist()) if len(solution_counts) > 0 else 'Aucune solution trouvée'

        # Calculer le top 3 des protocoles les plus utilisés
        protocol_counts = litige['protocolname'].value_counts()[:3]
        top_3_protocols = ', '.join(protocol_counts.index.tolist()) if len(protocol_counts) > 0 else 'Aucun protocole trouvé'

        # Afficher les KPI
        st.markdown("#### Top 3 des solutions et protocoles les plus utilisés")
        st.write(f"**Top 3 des solutions :** {top_3_solutions} | **Top 3 des protocoles :** {top_3_protocols}")

        