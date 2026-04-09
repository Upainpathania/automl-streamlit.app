# AutoML Streamlit App (Full Code)

```python
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Metrics
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="AutoML App", layout="wide")

st.title("AutoML Streamlit App")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.sidebar.header("ML Settings")

    # Learning Type
    learning_type = st.sidebar.selectbox(
        "Learning Type",
        ["Supervised", "Unsupervised"]
    )

    # Scaling
    scaling = st.sidebar.selectbox(
        "Scaling",
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    # Outlier Handling
    outlier_method = st.sidebar.selectbox(
        "Outlier Handling",
        ["None", "Remove", "Cap"]
    )

    # Target Column
    target = st.sidebar.selectbox("Target Column", df.columns)

    # Test Size
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

    # Split X y
    X = df.drop(columns=[target])
    y = df[target]

    # Detect Problem Type
    if y.dtype == 'object' or y.nunique() < 15:
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    st.sidebar.write("Detected:", problem_type)

    # Handle categorical
    X = pd.get_dummies(X)

    # Outlier Handling
    if outlier_method != "None":

        df_temp = pd.concat([X, y], axis=1)

        for col in X.columns:

            Q1 = df_temp[col].quantile(0.25)
            Q3 = df_temp[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            if outlier_method == "Remove":
                df_temp = df_temp[(df_temp[col] >= lower) & (df_temp[col] <= upper)]

            elif outlier_method == "Cap":
                df_temp[col] = np.where(df_temp[col] > upper, upper, df_temp[col])
                df_temp[col] = np.where(df_temp[col] < lower, lower, df_temp[col])

        X = df_temp.drop(columns=[target])
        y = df_temp[target]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scaling
    if scaling == "StandardScaler":
        scaler = StandardScaler()
    elif scaling == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Compare Models Button
    if st.button("Compare All Models"):

        results = {}

        # Classification Models
        if problem_type == "Classification":

            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(),
                "KNN": KNeighborsClassifier()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                results[name] = acc

        # Regression Models
        else:

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor()
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                score = r2_score(y_test, pred)
                results[name] = score

        results_df = pd.DataFrame(results.items(), columns=["Model", "Score"])
        results_df = results_df.sort_values("Score", ascending=False)

        st.subheader("Model Comparison")
        st.dataframe(results_df)

        best_model_name = results_df.iloc[0,0]
        st.success(f"Best Model: {best_model_name}")

        # Train Best Model
        best_model = models[best_model_name]
        best_model.fit(X_train, y_train)

        st.subheader("Prediction")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col)

        input_df = pd.DataFrame([input_data])

        if scaler:
            input_df = scaler.transform(input_df)

        if st.button("Predict"):
            prediction = best_model.predict(input_df)
            st.success(f"Prediction: {prediction[0]}")









