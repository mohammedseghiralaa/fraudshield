from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
DEMO_PATH = Path(__file__).parent / "demo_transactions.csv"

st.set_page_config(page_title="FraudShield", page_icon="🛡️")
st.title("🛡️ FraudShield")
st.write("Pick a transaction, send it to the API, and see the model's decision.")


@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_PATH)


demo = load_demo()

filter_choice = st.radio(
    "Show transactions",
    ["All", "Real frauds only", "Real normal only"],
    horizontal=True,
)
if filter_choice == "Real frauds only":
    options = demo.index[demo["true_label"] == 1]
elif filter_choice == "Real normal only":
    options = demo.index[demo["true_label"] == 0]
else:
    options = demo.index

idx = st.selectbox(
    "Transaction",
    options,
    format_func=lambda i: f"#{i} · amount {demo.loc[i, 'Amount']:.2f}",
)
row = demo.loc[idx]

with st.expander("Show the transaction features"):
    st.dataframe(row.drop("true_label").to_frame("value"))

if st.button("Score this transaction", type="primary"):
    payload = row.drop("true_label").astype(float).to_dict()
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Start it with: uvicorn fraudshield.api:app --reload")
        st.stop()
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        st.stop()

    result = r.json()

    col1, col2 = st.columns(2)
    col1.metric("Fraud probability", f"{result['fraud_probability']:.1%}")
    col2.metric("Decision threshold", f"{result['threshold']:.2f}")

    if result["is_fraud"]:
        st.error("🚨 Flagged as FRAUD")
    else:
        st.success("✅ Looks normal")

    real = "fraud" if int(row["true_label"]) == 1 else "normal"
    st.caption(f"Real label in the dataset: **{real}**")
