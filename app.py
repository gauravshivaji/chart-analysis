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

    # Detect date columns automatically
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

    # Clean probability
    df[prob_col] = (
        df[prob_col]
        .astype(str)
        .str.replace("%","",regex=False)
        .astype(float)
    )

    if df[prob_col].max() > 1:
        df[prob_col] = df[prob_col] / 100

    # Prediction
    df["prediction"] = (df[prob_col] > 0.5).astype(int)

    # -------------------
    # Period 1
    # -------------------

    df["return_1"] = (df[end_price1] - df[start_price]) / df[start_price]

    df["actual_1"] = (df["return_1"] > 0).astype(int)

    acc1 = accuracy_score(df["actual_1"], df["prediction"])

    # -------------------
    # Period 2
    # -------------------

    df["return_2"] = (df[end_price2] - df[start_price]) / df[start_price]

    df["actual_2"] = (df["return_2"] > 0).astype(int)

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
    # Confusion Matrix 1
    # -------------------

    st.subheader("Confusion Matrix (Jan → Mar)")

    cm1 = confusion_matrix(df["actual_1"], df["prediction"])

    fig1, ax1 = plt.subplots()

    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax1)

    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    st.pyplot(fig1)

    # -------------------
    # Confusion Matrix 2
    # -------------------

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

    # -------------------
    # Detailed Results
    # -------------------

    st.subheader("Detailed Predictions")

    df["correct_mar"] = df["actual_1"] == df["prediction"]
    df["correct_feb"] = df["actual_2"] == df["prediction"]

    st.dataframe(
        df[
            [
                "name",
                "pb",
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
