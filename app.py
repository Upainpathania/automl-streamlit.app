

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Supervised Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Unsupervised Models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)

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

# Load Data
def load_data(file):
    try:
        return pd.read_csv(file, sep=None, engine='python')
    except:
        try:
            for sep in [';', '|', '\t']:
                try:
                    return pd.read_csv(file, sep=sep)
                except:
                    continue
            return pd.read_excel(file)
        except:
            st.error("Unsupported file format ❌")
            return None


# Upload File
file = st.file_uploader("Upload File", type=["csv", "txt", "xlsx"])

if file:

    df = load_data(file)

    # ================= EDA =================
    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Shape:", df.shape)
        st.write("Duplicates:", df.duplicated().sum())

    with col2:
        st.write("Missing Values")
        st.write(df.isnull().sum())

    st.subheader("Dataset Info")

    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # ================= Visualization =================

    st.subheader("Visualization")

    graph_type = st.selectbox(
        "Select Graph",
        ["Histogram", "Scatter", "Boxplot"]
    )

    numeric_cols = df.select_dtypes(include=np.number).columns

    if graph_type == "Histogram":
        col = st.selectbox("Column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], ax=ax)
        st.pyplot(fig)

    elif graph_type == "Scatter":
        x = st.selectbox("X", numeric_cols)
        y = st.selectbox("Y", numeric_cols)

        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x], y=df[y], ax=ax)
        st.pyplot(fig)

    elif graph_type == "Boxplot":
        col = st.selectbox("Column", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    # ================= Sidebar =================

    st.sidebar.title("ML Settings")

    learning_type = st.sidebar.selectbox(
        "Learning Type",
        ["Supervised", "Unsupervised"]
    )

    scaler_option = st.sidebar.selectbox(
        "Scaling",
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    outlier_method = st.sidebar.selectbox(
        "Outlier Handling",
        ["None", "Remove", "Cap"]
    )

    # ================= Supervised =================

    if learning_type == "Supervised":

        target = st.sidebar.selectbox("Target Column", df.columns)

        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

        X = df.drop(columns=[target])
        y = df[target]

        # Encode categorical
        X = pd.get_dummies(X, drop_first=True)

        # Auto detect
        if y.dtype == 'object' or y.nunique() < 15:
            problem_type = "Classification"
        else:
            problem_type = "Regression"

        st.sidebar.write("Detected:", problem_type)

        # Outliers
        if outlier_method != "None":
            for col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                if outlier_method == "Remove":
                    X = X[(X[col] >= lower) & (X[col] <= upper)]

                elif outlier_method == "Cap":
                    X[col] = np.where(X[col] > upper, upper, X[col])
                    X[col] = np.where(X[col] < lower, lower, X[col])

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

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Models
        if problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVM": SVC(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier()
            }

        else:
            models = {
                "Linear Regression": LinearRegression(),
                "SVR": SVR(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor()
            }

        model_choice = st.sidebar.selectbox(
            "Algorithm",
            list(models.keys())
        )

        model = models[model_choice]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("Performance")

        # Classification
        if problem_type == "Classification":

            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
            st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
            st.write("F1:", f1_score(y_test, y_pred, average='weighted'))

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots()

            sns.heatmap(
                confusion_matrix(y_test, y_pred),
                annot=True,
                fmt='d',
                ax=ax
            )

            st.pyplot(fig)

        # Regression
        else:

            st.write("MAE:", mean_absolute_error(y_test, y_pred))
            st.write("MSE:", mean_squared_error(y_test, y_pred))
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
            st.write("R2:", r2_score(y_test, y_pred))

        # ================= Compare All Models =================

        if st.button("Compare All Models"):

            results = []

            for name, m in models.items():

                m.fit(X_train, y_train)

                pred = m.predict(X_test)

                if problem_type == "Classification":

                    acc = accuracy_score(y_test, pred)
                    f1 = f1_score(y_test, pred, average='weighted')

                    results.append([name, acc, f1])

                else:

                    r2 = r2_score(y_test, pred)
                    rmse = np.sqrt(mean_squared_error(y_test, pred))

                    results.append([name, r2, rmse])

            if problem_type == "Classification":

                results_df = pd.DataFrame(
                    results,
                    columns=["Model", "Accuracy", "F1"]
                )

            else:

                results_df = pd.DataFrame(
                    results,
                    columns=["Model", "R2", "RMSE"]
                )

            st.subheader("Model Comparison")

            st.dataframe(results_df)

    # ================= Unsupervised =================

    else:

        X = pd.get_dummies(df, drop_first=True)

        unsup = st.sidebar.selectbox(
            "Algorithm",
            ["KMeans", "DBSCAN", "PCA"]
        )

        if unsup == "KMeans":

            k = st.sidebar.slider("Clusters", 2, 10, 3)

            model = KMeans(n_clusters=k)

            labels = model.fit_predict(X)

            st.write("Silhouette:", silhouette_score(X, labels))

            st.bar_chart(pd.Series(labels).value_counts())

        elif unsup == "DBSCAN":

            model = DBSCAN()

            labels = model.fit_predict(X)

            st.bar_chart(pd.Series(labels).value_counts())

        elif unsup == "PCA":

            pca = PCA(n_components=2)

            comp = pca.fit_transform(X)

            fig, ax = plt.subplots()

            ax.scatter(comp[:, 0], comp[:, 1])

            st.pyplot(fig)







