import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Heart Disease ML App")

st.title("❤️ Heart Disease Prediction – ML Models")

uploaded_file = st.file_uploader(
    "heart.csv",
    type=["csv"]
)

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
        st.error("CSV must contain a 'target' column")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        model = joblib.load(f"model/{model_name}.pkl")
        y_pred = model.predict(X)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
