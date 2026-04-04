import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

st.title("AutoML Web App")

file = st.file_uploader("Upload Dataset")

if file is not None:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    st.write("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if df[target].dtype == 'object' or df[target].nunique() < 20:
        problem_type = "classification"
    else:
        problem_type = "regression"

    st.write("Problem Type:", problem_type)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("Train Model"):
        if problem_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier()
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor()
            }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.write("Model:", name)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
                st.write("Accuracy:", score)
            else:
                score = r2_score(y_test, preds)
                st.write("R2 Score:", score)
