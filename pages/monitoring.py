import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import os 
from back import *

st.title('Monitoring des performances du modèle')

facture = fetch_data_from_supabase('facture')
virement = fetch_data_from_supabase('virement')
prediction = fetch_data_from_supabase('prediction')

df_merge = pd.merge(facture, virement, on=['id', 'codetiers', 'idevt'], how='outer')
df_merge = df_merge[df_merge['montant_fac'].notna()]
df_merge['date_op'] = pd.to_datetime(df_merge['date_op'])
df_merge['date_echeance'] = pd.to_datetime(df_merge['date_echeance'])
df_merge = soustraire_dates(df_merge, "date_op", "date_echeance", "jours_retard")
df_merge["jours_retard"] = df_merge["jours_retard"].astype(str).str.extract('(-?\d+)').astype(float)
df_merge["type_retard"] = df_merge["jours_retard"].apply(lambda x: label_to_classes(x))

df_t = df_merge[["num_fac","type_retard"]]
df = pd.merge(df_t, prediction, on=['num_fac'], how='outer')
df = df[df['num_fac'].notna()]
y_true = df['type_retard']
y_pred = df['prediction']

st.subheader('Matrice de confusion')
plot_confusion_matrix(y_true, y_pred)




st.subheader('Taux d\'exactitude')
calculate_accuracy(y_true, y_pred)

st.subheader('Distribution des prédictions')
plot_prediction_distribution(y_pred)

st.set_option('deprecation.showPyplotGlobalUse', False)