import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Buy Probability Evaluation", layout="wide")

st.title("📊 Buy Probability Model Accuracy Dashboard")

st.write("Upload your Excel file to evaluate prediction accuracy.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error("Error reading Excel file.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Expected column names
    # ---------------------------

    start_price = "01-01-2026"
    end_price1 = "12-03-2026"
    end_price2 = "27/2.26"
    prob_col = "pb"

    required_cols = [start_price, end_price1, end_price2, prob_col]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in Excel file.")
            st.stop()

    # ---------------------------
    # Clean probability column
    # ---------------------------

    df[prob_col] = (
        df[prob_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float) / 100
    )

    # ---------------------------
    # Prediction from probability
    # ---------------------------

    df["prediction"] = (df[prob_col] > 0.5).astype(int)

    # ---------------------------
    # Period 1 (Jan → Mar)
    # ---------------------------

    df["return_1"] = (df[end_price1] - df[start_price]) / df[start_price]

    df["actual_1"] = (df["return_1"] > 0).astype(int)

    acc1 = accuracy_score(df["actual_1"], df["prediction"])

    # ---------------------------
    # Period 2 (Jan → Feb)
    # ---------------------------

    df["return_2"] = (df[end_price2] - df[start_price]) / df[start_price]

    df["actual_2"] = (df["return_2"] > 0).astype(int)

    acc2 = accuracy_score(df["actual_2"], df["prediction"])

    # ---------------------------
    # Accuracy Display
    # ---------------------------

    st.subheader("Model Accuracy")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy (01 Jan → 12 Mar)", f"{acc1*100:.2f}%")

    with col2:
        st.metric("Accuracy (01 Jan → 27 Feb)", f"{acc2*100:.2f}%")

    # ---------------------------
    # Confusion Matrix (Jan → Mar)
    # ---------------------------

    st.subheader("Confusion Matrix (Jan → Mar)")

    cm1 = confusion_matrix(df["actual_1"], df["prediction"])

    fig1, ax1 = plt.subplots()

    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax1)

    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    st.pyplot(fig1)

    # ---------------------------
    # Confusion Matrix (Jan → Feb)
    # ---------------------------

    st.subheader("Confusion Matrix (Jan → Feb)")

    cm2 = confusion_matrix(df["actual_2"], df["prediction"])

    fig2, ax2 = plt.subplots()

    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=ax2)

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    st.pyplot(fig2)

    # ---------------------------
    # Accuracy Comparison Chart
    # ---------------------------

    st.subheader("Accuracy Comparison")

    fig3, ax3 = plt.subplots()

    periods = ["Jan → Mar", "Jan → Feb"]
    accuracy_vals = [acc1*100, acc2*100]

    ax3.bar(periods, accuracy_vals)

    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Model Performance by Time Period")

    st.pyplot(fig3)

    # ---------------------------
    # Detailed Table
    # ---------------------------

    st.subheader("Detailed Predictions")

    df["correct_mar"] = df["actual_1"] == df["prediction"]
    df["correct_feb"] = df["actual_2"] == df["prediction"]

    st.dataframe(
        df[
            [
                "name",
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
    st.info("Please upload an Excel file to start analysis.")
