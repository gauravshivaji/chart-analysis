import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Probability Based Stock Prediction Evaluation")

uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx","csv"])

if uploaded_file:

    # -----------------------------
    # Load dataset
    # -----------------------------
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    st.subheader("Original Columns")
    st.write(df.columns)

    # -----------------------------
    # Select correct columns
    # -----------------------------
    price_cols = [c for c in df.columns if "2026" in c or "2025" in c]

    if len(price_cols) < 2:
        st.error("Price columns not detected correctly")
        st.stop()

    start_col = price_cols[0]
    end_col = price_cols[-1]

    df = df.rename(columns={
        start_col:"start_price",
        end_col:"end_price"
    })

    # -----------------------------
    # Clean numeric values
    # -----------------------------
    df["start_price"] = pd.to_numeric(df["start_price"], errors="coerce")
    df["end_price"] = pd.to_numeric(df["end_price"], errors="coerce")
    df["pb"] = pd.to_numeric(df["pb"], errors="coerce")

    df = df.dropna(subset=["start_price","end_price","pb"])

    # remove invalid rows
    df = df[df["start_price"] > 0]

    # -----------------------------
    # Calculate returns
    # -----------------------------
    df["return_%"] = ((df["end_price"] - df["start_price"]) / df["start_price"]) * 100

    # Remove unrealistic extreme values
    df = df[df["return_%"].abs() < 200]

    st.subheader("Clean Dataset")
    st.dataframe(df.head())

    # -----------------------------
    # Probability buckets
    # -----------------------------
    bins = [0,0.2,0.4,0.6,0.8,1]
    labels = ["0-20","20-40","40-60","60-80","80-100"]

    df["prob_bucket"] = pd.cut(df["pb"], bins=bins, labels=labels)

    investment = st.number_input("Total Investment", value=100000)

    results = []

    for bucket in labels:

        group = df[df["prob_bucket"] == bucket]

        if len(group) == 0:
            continue

        weight = investment / len(group)

        portfolio = weight * (group["end_price"] / group["start_price"])

        total_final = portfolio.sum()

        results.append({
            "Probability Range": bucket,
            "Stocks": len(group),
            "Initial Investment": investment,
            "Final Value": round(total_final,2),
            "Return %": round(((total_final-investment)/investment)*100,2)
        })

    result_df = pd.DataFrame(results)

    st.subheader("📊 Investment Performance by Probability")
    st.dataframe(result_df)

    # -----------------------------
    # Charts
    # -----------------------------
    fig = px.bar(
        result_df,
        x="Probability Range",
        y="Return %",
        title="Return by Probability Bucket"
    )
    st.plotly_chart(fig)

    avg = df.groupby("prob_bucket")["return_%"].mean().reset_index()

    fig2 = px.line(
        avg,
        x="prob_bucket",
        y="return_%",
        markers=True,
        title="Average Stock Return vs Probability"
    )

    st.plotly_chart(fig2)

    # -----------------------------
    # Debug extreme returns
    # -----------------------------
    st.subheader("Top 10 Highest Returns (Debug)")
    st.dataframe(df.sort_values("return_%",ascending=False).head(10))
