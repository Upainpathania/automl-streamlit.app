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


# File Upload
file = st.file_uploader("Upload File", type=["csv", "txt", "xlsx"])

if file:
    df = load_data(file)

    # ================= EDA =================
    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Duplicate Rows:", df.duplicated().sum())

    with col2:
        st.write("Missing Values:")
        st.write(df.isnull().sum())

    st.subheader("Head")
    st.dataframe(df.head())

    st.subheader("Tail")
    st.dataframe(df.tail())

    st.subheader("Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # ================= Visualization =================
    st.subheader("Data Visualization")

    graph_type = st.selectbox(
        "Select Graph Type",
        ["Histogram", "Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"]
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    if graph_type == "Histogram":
        col = st.selectbox("Select Column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col], ax=ax)
        st.pyplot(fig)

    elif graph_type == "Bar Chart":
        col = st.selectbox("Select Column", all_cols)
        df[col].value_counts().plot(kind='bar')
        st.pyplot(plt)

    elif graph_type == "Line Chart":
        col = st.selectbox("Select Column", numeric_cols)
        st.line_chart(df[col])

    elif graph_type == "Scatter Plot":
        x_col = st.selectbox("X Axis", numeric_cols)
        y_col = st.selectbox("Y Axis", numeric_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
        st.pyplot(fig)

    elif graph_type == "Box Plot":
        col = st.selectbox("Select Column", numeric_cols)
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
        "Scaling Method",
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    # ================= Supervised =================

    if learning_type == "Supervised":

        problem_type = st.sidebar.selectbox(
            "Problem Type",
            ["Classification", "Regression"]
        )

        target = st.sidebar.selectbox("Select Target Column", df.columns)

        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

        # Metrics
        if problem_type == "Classification":
            metric_options = ["Accuracy", "Precision", "Recall", "F1 Score"]
        else:
            metric_options = ["MAE", "MSE", "RMSE", "R2 Score", "Adjusted R2"]

        selected_metrics = st.sidebar.multiselect("Select Metrics", metric_options)

        # Prepare Data
        X = df.drop(columns=[target])
        y = df[target]

        # Fix categorical
        X = pd.get_dummies(X, drop_first=True)

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
        if problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(),
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

        model_choice = st.sidebar.selectbox("Select Algorithm", list(models.keys()))

        model = models[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Performance")

        # Classification
        if problem_type == "Classification":

            if "Accuracy" in selected_metrics:
                st.write("Accuracy:", accuracy_score(y_test, y_pred))

            if "Precision" in selected_metrics:
                st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))

            if "Recall" in selected_metrics:
                st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))

            if "F1 Score" in selected_metrics:
                st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

        # Regression
        else:

            n = X_test.shape[0]
            p = X_test.shape[1]

            if "MAE" in selected_metrics:
                st.write("MAE:", mean_absolute_error(y_test, y_pred))

            if "MSE" in selected_metrics:
                st.write("MSE:", mean_squared_error(y_test, y_pred))

            if "RMSE" in selected_metrics:
                st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

            if "R2 Score" in selected_metrics:
                r2 = r2_score(y_test, y_pred)
                st.write("R2 Score:", r2)

            if "Adjusted R2" in selected_metrics:
                r2 = r2_score(y_test, y_pred)
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                st.write("Adjusted R2:", adj_r2)

    # ================= Unsupervised =================

    else:

        st.subheader("Unsupervised Learning")

        X = pd.get_dummies(df, drop_first=True)

        if scaler_option == "StandardScaler":
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif scaler_option == "MinMaxScaler":
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif scaler_option == "RobustScaler":
            scaler = RobustScaler()
            X = scaler.fit_transform(X)

        unsup_algo = st.sidebar.selectbox(
            "Select Algorithm",
            ["KMeans", "DBSCAN", "PCA"]
        )

        if unsup_algo == "KMeans":
            k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            model = KMeans(n_clusters=k)
            labels = model.fit_predict(X)

            st.write("Silhouette Score:", silhouette_score(X, labels))

            st.subheader("Cluster Distribution")
            st.bar_chart(pd.Series(labels).value_counts())

        elif unsup_algo == "DBSCAN":
            model = DBSCAN()
            labels = model.fit_predict(X)
            st.subheader("Cluster Distribution")
            st.bar_chart(pd.Series(labels).value_counts())

        elif unsup_algo == "PCA":
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)

            st.subheader("PCA Visualization")
            fig, ax = plt.subplots()
            ax.scatter(components[:, 0], components[:, 1])
            st.pyplot(fig)
```




