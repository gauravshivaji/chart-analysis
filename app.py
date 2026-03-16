import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("📊 Probability Based Stock Prediction Evaluation")

uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "csv"])

if uploaded_file:

    # Load file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.strip()

    st.subheader("Dataset Columns")
    st.write(df.columns)

    # --- Identify important columns ---
    price_columns = []

    for col in df.columns:
        if "2026" in str(col) or "2025" in str(col):
            price_columns.append(col)

    if len(price_columns) < 2:
        st.error("Need two price columns (start and end date)")
        st.stop()

    start_price_col = price_columns[0]
    end_price_col = price_columns[-1]

    prob_col = "pb"

    # Rename columns
    df = df.rename(columns={
        start_price_col: "start_price",
        end_price_col: "end_price"
    })

    # Calculate return
    df["return_%"] = ((df["end_price"] - df["start_price"]) / df["start_price"]) * 100

    # Probability buckets
    bins = [0,20,40,60,80,100]
    labels = ["0-20","20-40","40-60","60-80","80-100"]

    df["prob_bucket"] = pd.cut(df[prob_col], bins=bins, labels=labels)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    investment = st.number_input("Total Investment", value=100000)

    results = []

    for bucket in labels:

        group = df[df["prob_bucket"] == bucket]

        if len(group) == 0:
            continue

        money_per_stock = investment / len(group)

        final_values = money_per_stock * (group["end_price"] / group["start_price"])

        total_initial = money_per_stock * len(group)
        total_final = final_values.sum()

        results.append({
            "Probability Range": bucket,
            "Stocks": len(group),
            "Initial Investment": total_initial,
            "Final Value": total_final,
            "Return %": ((total_final-total_initial)/total_initial)*100
        })

    result_df = pd.DataFrame(results)

    st.subheader("📊 Investment Performance by Probability Bucket")
    st.dataframe(result_df)

    # Return chart
    fig = px.bar(
        result_df,
        x="Probability Range",
        y="Return %",
        title="Return % by Probability Bucket"
    )

    st.plotly_chart(fig)

    # Investment comparison
    fig2 = px.bar(
        result_df,
        x="Probability Range",
        y=["Initial Investment","Final Value"],
        barmode="group",
        title="Initial vs Final Investment"
    )

    st.plotly_chart(fig2)

    # Average stock return
    avg_returns = df.groupby("prob_bucket")["return_%"].mean().reset_index()

    fig3 = px.line(
        avg_returns,
        x="prob_bucket",
        y="return_%",
        markers=True,
        title="Average Return vs Probability"
    )

    st.plotly_chart(fig3)

    st.success("Analysis Completed Successfully ✅")
