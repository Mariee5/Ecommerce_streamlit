import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Set page config ---
st.set_page_config(page_title="🧵 Ethnic Wear Review Analyzer", layout="wide")

# --- Custom App Title ---
st.markdown("<h1 style='text-align: center; color: teal;'>👗 Ethnic Wear Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("##### Analyze synthetic customer reviews for Indian clothing collections 💬")

# --- Sidebar selection ---
st.sidebar.title("🔍 Choose a Task")
task = st.sidebar.radio("Select Analysis Type", ["📄 View Dataset", "🧹 Clean Data", "📊 Visualize", "📈 Regression", "📌 Clustering", "✔️ Classification"])

# --- Spark Session ---
def create_spark_session():
    return SparkSession.builder.appName("EthnicWearReviewApp").getOrCreate()

# --- Synthetic Data Generation ---
def generate_data(rows=10000):
    np.random.seed(42)
    data = {
        "Customer ID": np.random.randint(1000, 1100, size=rows),
        "Age": np.random.randint(18, 65, size=rows),
        "Title": np.random.choice(["Loved it!", "Could be better", "Superb", "Not worth it", "Elegant design"], size=rows),
        "Review": np.random.choice([
            "Fabric was rich and elegant.", 
            "Too tight at the waist.", 
            "Color didn’t match the picture.", 
            "Perfect for the festive season.", 
            "Got compliments all day!"
        ], size=rows),
        "Rating": np.random.randint(1, 6, size=rows),
        "Recommended": np.random.choice([0, 1], size=rows),
        "Helpful Votes": np.random.poisson(3, size=rows),
        "Division": np.random.choice(["Ethnic", "Casual", "Fusion"], size=rows),
        "Department": np.random.choice(["Sarees", "Kurtis", "Lehengas", "Tops"], size=rows),
        "Category": np.random.choice(["Wedding", "Office", "College", "Casual"], size=rows)
    }
    return pd.DataFrame(data)

# --- Plotting Functions ---
def plot_reg(df):
    fig, ax = plt.subplots()
    sns.regplot(x=df["Age"], y=df["Rating"], scatter_kws={"color": "blue"}, line_kws={"color": "red"}, ax=ax)
    ax.set_title("Age vs Rating Regression")
    st.pyplot(fig)

def plot_cluster(df, preds):
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["Age"], y=df["Helpful Votes"], hue=preds.squeeze(), palette="Set2", ax=ax)
    ax.set_title("K-Means Cluster by Age and Helpfulness")
    st.pyplot(fig)

# --- Data Load ---
df_pandas = generate_data()
spark = create_spark_session()
df = spark.createDataFrame(df_pandas)

# --- Task Execution ---
if task == "📄 View Dataset":
    st.subheader("Sample Review Data (Synthetic)")
    st.dataframe(df_pandas.head(50))

elif task == "🧹 Clean Data":
    st.subheader("🧽 Cleaned Dataset Preview")
    df_pandas.dropna(inplace=True)
    df = spark.createDataFrame(df_pandas)
    st.dataframe(df_pandas.head(30))
    st.success("✅ Missing values handled successfully!")

elif task == "📊 Visualize":
    st.subheader("🧠 Exploratory Data Analysis")

    with st.expander("📍 Rating Distribution"):
        fig, ax = plt.subplots()
        sns.histplot(df_pandas['Rating'], bins=5, kde=True, ax=ax)
        ax.set_title("Product Rating Distribution")
        st.pyplot(fig)

    with st.expander("📊 Helpfulness vs Rating"):
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Rating', y='Helpful Votes', data=df_pandas, ax=ax2)
        st.pyplot(fig2)

    st.subheader("📋 Summary Statistics")
    st.dataframe(df.describe().toPandas())

elif task == "📈 Regression":
    st.subheader("Linear Regression: Predict Rating from Age")
    assembler = VectorAssembler(inputCols=["Age"], outputCol="features")
    model_df = assembler.transform(df).select('features', col("Rating").alias("label"))
    model = LinearRegression().fit(model_df)
    st.write("📉 Coefficient:", model.coefficients)
    st.write("📏 Intercept:", model.intercept)
    plot_reg(df_pandas)

elif task == "📌 Clustering":
    st.subheader("K-Means Clustering on Age & Helpfulness")
    assembler = VectorAssembler(inputCols=["Age", "Helpful Votes"], outputCol="features")
    model_df = assembler.transform(df)
    kmeans = KMeans(k=3).fit(model_df)
    preds = kmeans.transform(model_df).select("prediction").toPandas()
    st.write("📍 Cluster Centers:", kmeans.clusterCenters())
    plot_cluster(df_pandas, preds)

elif task == "✔️ Classification":
    st.subheader("Logistic Regression to Predict Recommendation")
    assembler = VectorAssembler(inputCols=["Age", "Rating"], outputCol="features")
    model_df = assembler.transform(df).select("features", col("Recommended").alias("label"))
    model = LogisticRegression().fit(model_df)
    st.write("🧮 Coefficients:", model.coefficients)
    st.write("Intercept:", model.intercept)

    # Test example
    test = spark.createDataFrame([(22, 5), (45, 2)], ["Age", "Rating"])
    test_model = assembler.transform(test).select("features")
    prediction = model.transform(test_model).select("prediction").toPandas()
    st.write("🔎 Test Predictions (Recommended or Not):")
    st.dataframe(prediction)

# Footer
st.markdown("---")
st.caption("💡 Created with 💖 using PySpark, Streamlit, and Matplotlib")
