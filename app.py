import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Supervised Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Unsupervised Models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
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
        sep = csv.Sniffer().sniff(
            file.read(5000).decode()
        ).delimiter
        
        file.seek(0)
        df = pd.read_csv(file, sep=sep)

    except:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=None, engine='python')
        except:
            try:
                file.seek(0)
                df = pd.read_excel(file)
            except:
                st.error("Unsupported file format ❌")
                return None

    # Fix broken single column
    if len(df.columns) == 1:
        try:
            df = df.iloc[:,0].str.split(";", expand=True)
        except:
            pass

    return df


file = st.file_uploader("Upload File", type=["csv", "txt", "xlsx"])

if file:

    df = load_data(file)

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

        X = pd.get_dummies(X, drop_first=True)

        if y.dtype == 'object' or y.nunique() < 15:
            problem_type = "Classification"
        else:
            problem_type = "Regression"

        st.sidebar.write("Detected:", problem_type)

        if outlier_method != "None":

            df_temp = pd.concat([X, y], axis=1)

            for col in X.columns:

                Q1 = df_temp[col].quantile(0.25)
                Q3 = df_temp[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                if outlier_method == "Remove":
                    df_temp = df_temp[
                        (df_temp[col] >= lower) &
                        (df_temp[col] <= upper)
                    ]

                elif outlier_method == "Cap":
                    df_temp[col] = np.where(
                        df_temp[col] > upper,
                        upper,
                        df_temp[col]
                    )

                    df_temp[col] = np.where(
                        df_temp[col] < lower,
                        lower,
                        df_temp[col]
                    )

            X = df_temp.drop(columns=[target])
            y = df_temp[target]

        scaler = None

        if scaler_option == "StandardScaler":
            scaler = StandardScaler()

        elif scaler_option == "MinMaxScaler":
            scaler = MinMaxScaler()

        elif scaler_option == "RobustScaler":
            scaler = RobustScaler()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Updated Models
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

        if problem_type == "Classification":

            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
            st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
            st.write("F1:", f1_score(y_test, y_pred, average='weighted'))

            cm = confusion_matrix(y_test, y_pred)

            st.subheader("Confusion Matrix")

            cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {i}" for i in np.unique(y_test)],
            columns=[f"Predicted {i}" for i in np.unique(y_test)]
            )

            st.dataframe(cm_df)

        else:

            st.write("MAE:", mean_absolute_error(y_test, y_pred))
            st.write("MSE:", mean_squared_error(y_test, y_pred))
            st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
            st.write("R2:", r2_score(y_test, y_pred))

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
            ["Random Forest", "Decision Tree", "KMeans", "DBSCAN", "PCA"]
        )

        if unsup == "Random Forest":

            model = IsolationForest()
            labels = model.fit_predict(X)
            st.bar_chart(pd.Series(labels).value_counts())

        elif unsup == "Decision Tree":

            model = DecisionTreeClassifier()
            labels = model.fit_predict(X)
            st.bar_chart(pd.Series(labels).value_counts())

        elif unsup == "KMeans":

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








