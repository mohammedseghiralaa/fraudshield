import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_ROWS = 500

st.set_page_config(page_title="Batch scoring", page_icon="🛡️")
st.title("Batch scoring")
st.write("Upload a CSV of transactions. Every row is scored through the API.")

uploaded = st.file_uploader("CSV file", type="csv")
if uploaded is None:
    st.info("Try `app/demo_transactions.csv`.")
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

if len(df) > MAX_ROWS:
    st.warning(f"Only the first {MAX_ROWS} rows are scored.")
    df = df.head(MAX_ROWS)

st.write(f"{len(df)} transactions loaded.")

if st.button("Score all", type="primary"):
    probs, flags = [], []
    bar = st.progress(0)
    try:
        for n, (_, row) in enumerate(df[FEATURES].iterrows(), start=1):
            r = requests.post(API_URL, json=row.astype(float).to_dict(), timeout=10)
            r.raise_for_status()
            res = r.json()
            probs.append(res["fraud_probability"])
            flags.append(res["is_fraud"])
            bar.progress(n / len(df))
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}. Is the API running?")
        st.stop()

    out = df.copy()
    out["fraud_probability"] = probs
    out["flagged"] = flags
    st.session_state["batch"] = (uploaded.name, out)

saved = st.session_state.get("batch")
if saved and saved[0] == uploaded.name:
    out = saved[1]
    flagged = int(out["flagged"].sum())

    c1, c2 = st.columns(2)
    c1.metric("Transactions", len(out))
    c2.metric("Flagged as fraud", flagged)

    if "true_label" in out.columns:
        real = out["true_label"] == 1
        caught = int((out["flagged"] & real).sum())
        missed = int((~out["flagged"] & real).sum())
        false_alarms = int((out["flagged"] & ~real).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Frauds caught", f"{caught} / {caught + missed}")
        c2.metric("Frauds missed", missed)
        c3.metric("False alarms", false_alarms)

    shown = ["fraud_probability", "flagged", "Amount"]
    if "true_label" in out.columns:
        shown.append("true_label")
    st.dataframe(out[shown].sort_values("fraud_probability", ascending=False))

    st.download_button("Download results (CSV)", out.to_csv(index=False), "fraudshield_results.csv")
    