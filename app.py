import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# Page Config
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

# Upload File
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Dataset Preview
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

    # Head and Tail
    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Last 5 Rows")
    st.dataframe(df.tail())

    # Dataset Info
    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # Sidebar Settings
    st.sidebar.title("ML Settings")

    target = st.sidebar.selectbox("Select Target Column", df.columns)

    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

    scaler_option = st.sidebar.selectbox(
        "Scaling Method",
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    model_choice = st.sidebar.selectbox(
        "Select Algorithm",
        ["Logistic Regression", "SVM", "Random Forest", "Decision Tree", "KNN"]
    )

    # Prepare Data
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

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier()
    }

    # Train Selected Model
    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.subheader("Model Performance")
    st.write("Algorithm:", model_choice)
    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    # Model Comparison Button
    if st.button("Compare All Models"):
        results = []

        for name, m in models.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)

            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, average='weighted')
            rec = recall_score(y_test, pred, average='weighted')
            f1 = f1_score(y_test, pred, average='weighted')

            results.append([name, acc, prec, rec, f1])

        results_df = pd.DataFrame(
            results,
            columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
        )

        st.subheader("Model Comparison")
        st.dataframe(results_df)

