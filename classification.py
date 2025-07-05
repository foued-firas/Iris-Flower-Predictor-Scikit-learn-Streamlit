import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# --------- Chargement des données ---------
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# --------- Entraînement du modèle ---------
model = RandomForestClassifier()
model.fit(df.iloc[:, :-1], df['species'])

# --------- Personnalisation de la page ---------
st.set_page_config(page_title="Iris Flower Classifier ", layout="centered")
st.title("🌸 Iris Flower Classification")
st.markdown("Utilisez les curseurs dans la **barre latérale** pour prédire l'espèce d'une fleur d'Iris.")

# --------- Interface utilisateur (sidebar) ---------
st.sidebar.header(" Paramètres d'entrée")

sepal_length = st.sidebar.slider(
    "Longueur du sépale (cm)", 
    float(df['sepal length (cm)'].min()), 
    float(df['sepal length (cm)'].max()), 
    float(df['sepal length (cm)'].mean())
)

sepal_width = st.sidebar.slider(
    "Largeur du sépale (cm)", 
    float(df['sepal width (cm)'].min()), 
    float(df['sepal width (cm)'].max()), 
    float(df['sepal width (cm)'].mean())
)

petal_length = st.sidebar.slider(
    "Longueur du pétale (cm)", 
    float(df['petal length (cm)'].min()), 
    float(df['petal length (cm)'].max()), 
    float(df['petal length (cm)'].mean())
)

petal_width = st.sidebar.slider(
    "Largeur du pétale (cm)", 
    float(df['petal width (cm)'].min()), 
    float(df['petal width (cm)'].max()), 
    float(df['petal width (cm)'].mean())
)

# --------- Prédiction ---------
input_data = [sepal_length, sepal_width, petal_length, petal_width]
prediction = model.predict([input_data])
predicted_species = target_names[prediction[0]]

# --------- Affichage des résultats ---------
st.subheader("Caractéristiques fournies")
features_df = pd.DataFrame(
    {
        "Mesure": [
            "Longueur du sépale", 
            "Largeur du sépale", 
            "Longueur du pétale", 
            "Largeur du pétale"
        ],
        "Valeur (cm)": input_data
    }
)
st.table(features_df)

st.subheader(" Résultat de la prédiction")
st.success(f" Espèce prédite : **{predicted_species.capitalize()}**")

# --------- Footer ---------
st.markdown("---")
st.markdown(
    "<small>Démo d'une application Streamlit - Classification avec Random Forest 🌲</small>",
    unsafe_allow_html=True
)
