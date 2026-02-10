import os
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.title("Heart Disease Prediction – ML Models")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("CSV must contain 'target' column")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        MODEL_FILES = {
            "Logistic Regression": "logistic_regression.pkl",
            "Decision Tree": "decision_tree.pkl",
            "KNN": "KNN.pkl",
            "Naive Bayes": "naive_bayes.pkl",
            "Random Forest": "random_forest.pkl",
            "XGBoost": "XGBoost.pkl"
        }

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_file = MODEL_FILES[model_name]
        model_path = os.path.join(BASE_DIR, "model", model_file)        
        model = joblib.load(model_path)

        # 🔴 THIS IS WHERE THE CHANGE GOES
        #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        #model_path = os.path.join(BASE_DIR, "model", f"{model_name}.pkl")
        #model = joblib.load(model_path)
        
        y_pred = model.predict(X)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)
