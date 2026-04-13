import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('model_depression.pkl')
encoders = joblib.load('encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

st.title("Prédicteur de Santé Mentale Étudiante")
st.write("Répondez aux questions ci-dessous pour une évaluation rapide.")

# Interfaces de saisie
sleep = st.selectbox("Qualité de votre sommeil ?", encoders['Sleep Pattern?'].classes_)
lonely = st.selectbox("Vous sentez-vous isolé(e) ?", encoders["Do you feel isolated or lonely even when you're surrounded by people"].classes_)

if st.button("Prédire"):
    # Transformation des entrées
    sleep_encoded = encoders['Sleep Pattern?'].transform([sleep])[0]
    lonely_encoded = encoders["Do you feel isolated or lonely even when you're surrounded by people"].transform([lonely])[0]
    
    # Prédiction
    prediction = model.predict([[sleep_encoded, lonely_encoded]])
    resultat = target_encoder.inverse_transform(prediction)[0]
    
    if resultat == "Yes":
        st.error("Résultat : Risque de dépression détecté. Il est conseillé de consulter un professionnel.")
    else:
        st.success("Résultat : Pas de signe majeur de dépression détecté selon ces facteurs.")