import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Stock Evaluation", layout="wide")

st.title("📊 Probability Based Stock Prediction Evaluation")

uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "csv"])

if uploaded_file is not None:
    # -----------------------------
    # Load dataset
    # -----------------------------
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Select columns
    # -----------------------------
    col_alpha, col_beta, col_gamma = st.columns(3)
    
    with col_alpha:
        start_col = st.selectbox("START PRICE column", df.columns)
    with col_beta:
        end_col = st.selectbox("END PRICE column", df.columns)
    with col_gamma:
        prob_col = st.selectbox("PROBABILITY column", df.columns)

    # Convert to numeric and drop NaNs
    df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
    df[end_col] = pd.to_numeric(df[end_col], errors="coerce")
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")

    df = df.dropna(subset=[start_col, end_col, prob_col])
    
    # Filter valid data
    df = df[df[start_col] > 0]
    df["return_%"] = ((df[end_col] - df[start_col]) / df[start_col]) * 100
    df = df[df["return_%"].abs() < 200] # Remove outliers

    # -----------------------------
    # Analysis Logic
    # -----------------------------
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    df["prob_bucket"] = pd.cut(df[prob_col], bins=bins, labels=labels)

    investment = st.number_input("Investment per Bucket", value=100000)

    results = []
    for bucket in labels:
        group = df[df["prob_bucket"] == bucket]
        if len(group) == 0:
            continue
        
        # Calculate performance
        weight_per_stock = investment / len(group)
        final_vals = weight_per_stock * (group[end_col] / group[start_col])
        total_final = final_vals.sum()

        results.append({
            "Probability Range": bucket,
            "Stocks": len(group),
            "Initial Investment": investment,
            "Final Value": round(total_final, 2),
            "Return %": round(((total_final - investment) / investment) * 100, 2)
        })

    if results:
        result_df = pd.DataFrame(results)
        st.subheader("📈 Performance by Probability Bucket")
        st.table(result_df)

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(result_df, x="Probability Range", y="Return %", color="Return %",
                          title="Return % per Bucket")
            st.plotly_chart(fig1, use_container_width=True)
        
        with c2:
            avg_returns = df.groupby("prob_bucket", observed=True)["return_%"].mean().reset_index()
            fig2 = px.line(avg_returns, x="prob_bucket", y="return_%", markers=True,
                           title="Avg Stock Return vs Probability")
            st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # Top Portfolio Simulation
    # -----------------------------
    st.divider()
    st.subheader("🚀 Top Probability Portfolio Simulation")
    
    top_n = st.slider("Number of top stocks", 5, 50, 10)
    top_stocks = df.sort_values(prob_col, ascending=False).head(top_n).copy()
    
    if not top_stocks.empty:
        money_per_stock = investment / top_n
        top_stocks["final_value"] = money_per_stock * (top_stocks[end_col] / top_stocks[start_col])
        
        p_initial = investment
        p_final = top_stocks["final_value"].sum()
        p_return = ((p_final - p_initial) / p_initial) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Initial", f"₹{p_initial:,.2f}")
        m2.metric("Final", f"₹{p_final:,.2f}")
        m3.metric("Return", f"{p_return:.2f}%")
        
        st.dataframe(top_stocks[[prob_col, start_col, end_col, "final_value"]])
    
else:
    st.info("Please upload a CSV or Excel file to begin.")
