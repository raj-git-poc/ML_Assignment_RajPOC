import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heart Disease Prediction - ML Models")

uploaded_file = st.file_uploader("heart.csv", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X = data.drop("target", axis=1)
    y = data["target"]

    model = joblib.load(f"model/{model_name}.pkl")
    y_pred = model.predict(X)

    st.subheader("Evaluation Metrics")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

