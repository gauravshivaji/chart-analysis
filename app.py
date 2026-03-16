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
    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Select Columns")

    columns = df.columns.tolist()

    start_price = st.selectbox("Select Start Date Price Column", columns)
    end_price1 = st.selectbox("Select End Date Column (Comparison 1)", columns)
    end_price2 = st.selectbox("Select End Date Column (Comparison 2)", columns)
    prob_col = st.selectbox("Select Probability Column", columns)
    name_col = st.selectbox("Select Stock Name Column", columns)

    # Clean probability column
    df[prob_col] = (
        df[prob_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    if df[prob_col].max() > 1:
        df[prob_col] = df[prob_col] / 100

    # Prediction
    df["prediction"] = (df[prob_col] > 0.5).astype(int)

    # Period 1
    df["return_1"] = (df[end_price1] - df[start_price]) / df[start_price]
    df["actual_1"] = (df["return_1"] > 0).astype(int)

    acc1 = accuracy_score(df["actual_1"], df["prediction"])

    # Period 2
    df["return_2"] = (df[end_price2] - df[start_price]) / df[start_price]
    df["actual_2"] = (df["return_2"] > 0).astype(int)

    acc2 = accuracy_score(df["actual_2"], df["prediction"])

    st.subheader("Model Accuracy")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy Comparison 1", f"{acc1*100:.2f}%")

    with col2:
        st.metric("Accuracy Comparison 2", f"{acc2*100:.2f}%")

    # Confusion Matrix 1
    st.subheader("Confusion Matrix (Comparison 1)")

    cm1 = confusion_matrix(df["actual_1"], df["prediction"])

    fig1, ax1 = plt.subplots()
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax1)

    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    st.pyplot(fig1)

    # Confusion Matrix 2
    st.subheader("Confusion Matrix (Comparison 2)")

    cm2 = confusion_matrix(df["actual_2"], df["prediction"])

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=ax2)

    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    st.pyplot(fig2)

    # Accuracy comparison chart
    st.subheader("Accuracy Comparison Chart")

    fig3, ax3 = plt.subplots()

    periods = ["Comparison 1", "Comparison 2"]
    accuracy_vals = [acc1 * 100, acc2 * 100]

    ax3.bar(periods, accuracy_vals)

    ax3.set_ylabel("Accuracy %")
    ax3.set_title("Model Performance")

    st.pyplot(fig3)

    # Detailed results
    st.subheader("Detailed Prediction Table")

    df["correct_comp1"] = df["actual_1"] == df["prediction"]
    df["correct_comp2"] = df["actual_2"] == df["prediction"]

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
                "correct_comp1",
                "correct_comp2",
            ]
        ]
    )

else:
    st.info("Upload an Excel file to start analysis.")
