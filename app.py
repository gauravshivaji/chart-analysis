import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Buy Probability Evaluation", layout="wide")

st.title("📊 Buy Probability Model Accuracy Dashboard")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Detect date columns
    date_cols = [c for c in df.columns if "2026" in str(c)]

    if len(date_cols) < 3:
        st.error("Date columns not detected correctly.")
        st.stop()

    start_price = date_cols[0]
    end_price1 = date_cols[1]
    end_price2 = date_cols[2]

    name_col = "name"
    prob_col = "pb"

    st.write("### Detected Columns")
    st.write("Start Price:", start_price)
    st.write("Comparison 1:", end_price1)
    st.write("Comparison 2:", end_price2)

    # Clean probability column
    df[prob_col] = (
        df[prob_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    if df[prob_col].max() > 1:
        df[prob_col] = df[prob_col] / 100

    # Clean price columns
    df[start_price] = pd.to_numeric(df[start_price], errors="coerce")
    df[end_price1] = pd.to_numeric(df[end_price1], errors="coerce")
    df[end_price2] = pd.to_numeric(df[end_price2], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=[start_price, end_price1, end_price2])

    # Avoid divide by zero
    df = df[df[start_price] > 1]

    # Prediction
    df["prediction"] = (df[prob_col] > 0.5).astype(int)

    # -------------------
    # Period 1
    # -------------------

    df["return_1"] = (df[end_price1] - df[start_price]) / df[start_price]
    df["actual_1"] = (df["return_1"] > 0).astype(int)

    # -------------------
    # Period 2
    # -------------------

    df["return_2"] = (df[end_price2] - df[start_price]) / df[start_price]
    df["actual_2"] = (df["return_2"] > 0).astype(int)

    # Remove unrealistic returns
    df = df[(df["return_1"] < 1) & (df["return_1"] > -1)]
    df = df[(df["return_2"] < 1) & (df["return_2"] > -1)]

    acc1 = accuracy_score(df["actual_1"], df["prediction"])
    acc2 = accuracy_score(df["actual_2"], df["prediction"])

    # -------------------
    # Accuracy Metrics
    # -------------------

    st.subheader("Model Accuracy")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy (Jan → Mar)", f"{acc1*100:.2f}%")

    with col2:
        st.metric("Accuracy (Jan → Feb)", f"{acc2*100:.2f}%")

    # -------------------
    # Confusion Matrix
    # -------------------

    st.subheader("Confusion Matrix (Jan → Mar)")

    cm1 = confusion_matrix(df["actual_1"], df["prediction"])

    fig1, ax1 = plt.subplots()
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax1)

    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    st.pyplot(fig1)

    st.subheader("Confusion Matrix (Jan → Feb)")

    cm2 = confusion_matrix(df["actual_2"], df["prediction"])

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=ax2)

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    st.pyplot(fig2)

    # -------------------
    # Accuracy Chart
    # -------------------

    st.subheader("Accuracy Comparison")

    fig3, ax3 = plt.subplots()

    periods = ["Jan→Mar", "Jan→Feb"]
    values = [acc1*100, acc2*100]

    ax3.bar(periods, values)
    ax3.set_ylabel("Accuracy %")

    st.pyplot(fig3)

    # ==========================================================
    # MODEL INSIGHT ANALYSIS
    # ==========================================================

    st.header("🔎 Model Insight Analysis")

    # Probability distribution

    st.subheader("Probability Distribution")

    fig4, ax4 = plt.subplots()

    ax4.hist(df[prob_col], bins=20)

    ax4.set_title("Distribution of Model Probabilities")
    ax4.set_xlabel("pb")
    ax4.set_ylabel("Number of Stocks")

    st.pyplot(fig4)

    # ==========================================================
    # PB RANGE INVESTMENT ANALYSIS
    # ==========================================================

    st.header("📊 PB Range Investment Analysis")

    investment = 100000

    bins = [0,0.2,0.4,0.6,0.8,1]

    labels = [
        "0–20%",
        "20–40%",
        "40–60%",
        "60–80%",
        "80–100%"
    ]

    df["pb_range"] = pd.cut(df[prob_col], bins=bins, labels=labels)

    pb_table = df.groupby("pb_range").agg(
        stocks=("pb_range","count"),
        median_return_feb=("return_2","median"),
        median_return_mar=("return_1","median")
    ).reset_index()

    pb_table["value_feb"] = investment * (1 + pb_table["median_return_feb"])
    pb_table["value_mar"] = investment * (1 + pb_table["median_return_mar"])

    pb_table["median_return_feb"] = pb_table["median_return_feb"] * 100
    pb_table["median_return_mar"] = pb_table["median_return_mar"] * 100

    pb_table = pb_table.round(2)

    st.dataframe(pb_table)

    # Chart

    st.subheader("Return vs Model Probability")

    fig5, ax5 = plt.subplots()

    ax5.bar(pb_table["pb_range"], pb_table["median_return_mar"])

    ax5.set_xlabel("PB Range")
    ax5.set_ylabel("Median Return %")

    st.pyplot(fig5)

    # -------------------
    # Top Predictions
    # -------------------

    st.subheader("Top Confidence Predictions")

    top_pb = df.sort_values(prob_col, ascending=False).head(10)

    st.dataframe(top_pb[[name_col, prob_col, "return_1"]])

    # -------------------
    # Extreme Return Debug
    # -------------------

    st.subheader("Extreme Return Check")

    extreme = df[df["return_1"] > 0.5]

    st.dataframe(extreme[[name_col, start_price, end_price1, "return_1"]])

    # -------------------
    # Detailed Results
    # -------------------

    st.subheader("Detailed Predictions")

    df["correct_mar"] = df["actual_1"] == df["prediction"]
    df["correct_feb"] = df["actual_2"] == df["prediction"]

    st.dataframe(
        df[
            [
                name_col,
                prob_col,
                start_price,
                end_price1,
                end_price2,
                "prediction",
                "actual_1",
                "actual_2",
                "correct_mar",
                "correct_feb",
            ]
        ]
    )

else:
    st.info("Upload your Excel file to begin analysis.")
