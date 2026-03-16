import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Stock Probability Prediction Evaluation")

uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx","csv"])

if uploaded_file:

    # Load dataset
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Remove duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename important columns safely
    column_map = {}

    for col in df.columns:
        c = col.lower()

        if "start" in c:
            column_map[col] = "start_price"
        elif "end" in c:
            column_map[col] = "end_price"
        elif "pb" in c:
            column_map[col] = "pb"
        elif "name" in c:
            column_map[col] = "stock"

    df = df.rename(columns=column_map)

    # Ensure required columns exist
    required = ["start_price","end_price","pb"]

    for r in required:
        if r not in df.columns:
            st.error(f"Column {r} not found in dataset")
            st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Calculate returns
    df["return_%"] = ((df["end_price"] - df["start_price"]) / df["start_price"]) * 100

    # Probability buckets
    bins = [0,0.2,0.4,0.6,0.8,1.0]
    labels = ["0-20","20-40","40-60","60-80","80-100"]

    df["prob_bucket"] = pd.cut(df["pb"], bins=bins, labels=labels)

    investment = st.number_input("Total Investment", value=100000)

    results = []

    for bucket in labels:

        group = df[df["prob_bucket"] == bucket]

        if len(group) == 0:
            continue

        money_per_stock = investment / len(group)

        final_value = money_per_stock * (group["end_price"] / group["start_price"])

        total_initial = money_per_stock * len(group)
        total_final = final_value.sum()

        results.append({
            "Probability Range": bucket,
            "Stocks": len(group),
            "Initial Investment": round(total_initial,2),
            "Final Value": round(total_final,2),
            "Return %": round(((total_final-total_initial)/total_initial)*100,2)
        })

    result_df = pd.DataFrame(results)

    st.subheader("📊 Investment Performance by Probability")

    st.dataframe(result_df)

    # Return chart
    fig1 = px.bar(
        result_df,
        x="Probability Range",
        y="Return %",
        title="Return by Probability Bucket"
    )

    st.plotly_chart(fig1)

    # Investment comparison
    fig2 = px.bar(
        result_df,
        x="Probability Range",
        y=["Initial Investment","Final Value"],
        barmode="group",
        title="Initial vs Final Investment"
    )

    st.plotly_chart(fig2)

    # Average return
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
