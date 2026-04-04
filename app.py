import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Page config
st.set_page_config(page_title="AutoML App", layout="wide")

# Title
st.markdown(
    """
    <h1 style='text-align: center;'>AutoML Web App</h1>
    <p style='text-align: center;'>Made by <b>Upain</b></p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Upload file
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    # Dataset Overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Dataset Shape")
        st.write(df.shape)

        st.write("Duplicate Rows")
        st.write(df.duplicated().sum())

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

    # Head & Tail
    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Last 5 Rows")
    st.dataframe(df.tail())

    # Dataset Info
    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # Sidebar ML Settings
    st.sidebar.title("ML Settings")

    target = st.sidebar.selectbox("Select Target Column", df.columns)
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

    scaler_option = st.sidebar.selectbox(
        "Scaling Method",
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]

    # Scaling
    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif scaler_option == "RobustScaler":
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = []

    st.subheader("Model Results")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append([name, acc, prec, rec, f1])

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

    st.subheader("Model Comparison")
    st.dataframe(results_df)

    # Confusion Matrix for Random Forest
    st.subheader("Confusion Matrix (Random Forest)")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Download model
    joblib.dump(model, "model.pkl")
    with open("model.pkl", "rb") as f:
        st.download_button("Download Trained Model", f, file_name="model.pkl")

# Footer
st.markdown("---")
st.markdown(
    "<center>Made by Upain | AutoML Streamlit App</center>",
    unsafe_allow_html=True
)
