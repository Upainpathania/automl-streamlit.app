import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, r2_score
)
import streamlit as st

st.title("AutoML Web App")
st.markdown("### Made by Upain")

st.title("Auto ML Streamlit App")

# Upload dataset
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("Dataset Preview")
    st.dataframe(df)
 # Dataset Overview / EDA
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

st.subheader("First 5 Rows")
st.dataframe(df.head())

st.subheader("Last 5 Rows")
st.dataframe(df.tail())

# Dataset Info
st.subheader("Dataset Info")

import io
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)   

    # Sidebar options
    st.sidebar.title("ML Settings")

    target = st.sidebar.selectbox("Select Target Column", df.columns)
    problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

    scaler_option = st.sidebar.selectbox(
        "Scaling",
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    # EDA Section
    st.subheader("EDA - Data Visualization")

    if st.checkbox("Show Correlation Heatmap"):
        plt.figure()
        sns.heatmap(df.corr(), annot=True)
        st.pyplot(plt)

    if st.checkbox("Show Histogram"):
        column = st.selectbox("Select Column", df.columns)
        plt.figure()
        sns.histplot(df[column])
        st.pyplot(plt)

    if st.checkbox("Show Boxplot"):
        column = st.selectbox("Select Boxplot Column", df.columns)
        plt.figure()
        sns.boxplot(x=df[column])
        st.pyplot(plt)

    # Prepare Data
    X = df.drop(target, axis=1)
    y = df[target]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scaling
    if scaler_option == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_option == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Model Selection
    if problem_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC(probability=True),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier()
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "SVR": SVR(),
            "KNN Regressor": KNeighborsRegressor()
        }

    if st.button("Train Models"):
        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            if problem_type == "Classification":
                acc = accuracy_score(y_test, pred)
                prec = precision_score(y_test, pred, average='weighted')
                rec = recall_score(y_test, pred, average='weighted')
                f1 = f1_score(y_test, pred, average='weighted')

                results.append([name, acc, prec, rec, f1])

                st.subheader(name)
                st.write("Accuracy:", acc)
                st.write("Precision:", prec)
                st.write("Recall:", rec)
                st.write("F1 Score:", f1)

                cm = confusion_matrix(y_test, pred)
                plt.figure()
                sns.heatmap(cm, annot=True)
                st.pyplot(plt)

                st.text(classification_report(y_test, pred))

            else:
                mse = mean_squared_error(y_test, pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, pred)

                results.append([name, mse, rmse, r2])

                st.subheader(name)
                st.write("MSE:", mse)
                st.write("RMSE:", rmse)
                st.write("R2 Score:", r2)

        # Results Table
        st.subheader("Model Comparison")
        if problem_type == "Classification":
            results_df = pd.DataFrame(
                results,
                columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
            )
        else:
            results_df = pd.DataFrame(
                results,
                columns=["Model", "MSE", "RMSE", "R2"]
            )

        st.dataframe(results_df)
