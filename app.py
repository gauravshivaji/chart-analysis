import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Probability Based Stock Prediction Evaluation")

uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx","csv"])

if uploaded_file is not None:

```
# -----------------------------
# Load dataset
# -----------------------------
if uploaded_file.name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_csv(uploaded_file)

# Clean column names
df.columns = df.columns.astype(str).str.strip()

st.subheader("Dataset Columns")
st.write(df.columns)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Select columns
# -----------------------------
start_col = st.selectbox("Select START PRICE column", df.columns)
end_col = st.selectbox("Select END PRICE column", df.columns)
prob_col = st.selectbox("Select PROBABILITY column", df.columns)

# Convert numeric
df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
df[end_col] = pd.to_numeric(df[end_col], errors="coerce")
df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")

df = df.dropna(subset=[start_col, end_col, prob_col])

# Remove invalid prices
df = df[df[start_col] > 0]

# -----------------------------
# Calculate return
# -----------------------------
df["return_%"] = ((df[end_col] - df[start_col]) / df[start_col]) * 100

# Remove extreme outliers
df = df[df["return_%"].abs() < 200]

st.subheader("Clean Data Preview")
st.dataframe(df.head())

# -----------------------------
# Probability buckets
# -----------------------------
bins = [0,0.2,0.4,0.6,0.8,1]
labels = ["0-20","20-40","40-60","60-80","80-100"]

df["prob_bucket"] = pd.cut(df[prob_col], bins=bins, labels=labels)

investment = st.number_input("Total Investment", value=100000)

results = []

for bucket in labels:

    group = df[df["prob_bucket"] == bucket]

    if len(group) == 0:
        continue

    weight = investment / len(group)

    portfolio_value = weight * (group[end_col] / group[start_col])

    total_final = portfolio_value.sum()

    results.append({
        "Probability Range": bucket,
        "Stocks": len(group),
        "Initial Investment": investment,
        "Final Value": round(total_final,2),
        "Return %": round(((total_final-investment)/investment)*100,2)
    })

result_df = pd.DataFrame(results)

st.subheader("📊 Investment Performance by Probability Bucket")
st.dataframe(result_df)

# -----------------------------
# Top Probability Portfolio
# -----------------------------
st.subheader("🚀 Top Probability Portfolio Simulation")

top_n = st.slider(
    "Select number of top probability stocks",
    min_value=5,
    max_value=50,
    value=10
)

top_stocks = df.sort_values(prob_col, ascending=False).head(top_n).copy()

money_per_stock = investment / top_n

top_stocks["investment"] = money_per_stock
top_stocks["final_value"] = money_per_stock * (top_stocks[end_col] / top_stocks[start_col])

portfolio_initial = top_stocks["investment"].sum()
portfolio_final = top_stocks["final_value"].sum()

portfolio_return = ((portfolio_final - portfolio_initial) / portfolio_initial) * 100

st.write("### Top Probability Stocks")
st.dataframe(top_stocks[[prob_col, start_col, end_col, "final_value"]])

col1, col2, col3 = st.columns(3)

col1.metric("Initial Investment", f"₹{portfolio_initial:,.2f}")
col2.metric("Final Portfolio Value", f"₹{portfolio_final:,.2f}")
col3.metric("Return %", f"{portfolio_return:.2f}%")

# -----------------------------
# Charts
# -----------------------------
fig1 = px.bar(
    result_df,
    x="Probability Range",
    y="Return %",
    title="Return by Probability Bucket"
)

st.plotly_chart(fig1)

avg_returns = df.groupby("prob_bucket")["return_%"].mean().reset_index()

fig2 = px.line(
    avg_returns,
    x="prob_bucket",
    y="return_%",
    markers=True,
    title="Average Stock Return vs Probability"
)

st.plotly_chart(fig2)

# Debug extreme returns
st.subheader("Top 10 Highest Returns (Debug)")
st.dataframe(df.sort_values("return_%", ascending=False).head(10))
```
