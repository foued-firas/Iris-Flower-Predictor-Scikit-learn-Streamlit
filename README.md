# 🌸 Iris Flower Classifier – Streamlit + Random Forest

Cette application interactive permet de prédire l'espèce d'une fleur d'Iris selon ses mesures morphologiques. Le modèle utilise un Random Forest de scikit-learn intégré via Streamlit.

---

##  Démo locale

### Installation

```bash
git clone https://github.com/foued-firas/iris-flower-classifier-streamlit.git
cd iris-flower-classifier-streamlit
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
streamlit run classification.py
