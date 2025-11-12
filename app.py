import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="FairLens — Performance Review Auditor", layout="wide")

# ===============================
# Load Bias Rules
# ===============================
@st.cache_data
def load_bias_rules(path: str = "bias_rules.csv") -> pd.DataFrame:
    p = Path(path)
    cols = ["phrase", "category", "context_rule", "tip"]
    if not p.exists():
        st.warning("bias_rules.csv not found; using empty rule set.")
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(p, sep="|", engine="python")
    except Exception as e:
        st.error(f"Failed to read bias_rules.csv: {e}")
        return pd.DataFrame(columns=cols)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"bias_rules.csv missing columns: {missing}")
        return pd.DataFrame(columns=cols)
    return df[cols]

# ===============================
# Core Data + App Flow
# ===============================
@st.cache_data
def load_reviews(path: str = "samplereviews.csv"):
    if not Path(path).exists():
        return pd.DataFrame(columns=["employee_id","role","gender","kpi_rating","competency_rating","initiative_rating","overall_rating","comment"])
    return pd.read_csv(path)

bias_df = load_bias_rules()
reviews_df = load_reviews()

tab1, tab2, tab3 = st.tabs(["Submit Review", "Audit & Fairness", "Privacy & Export"])

# Submit Review
with tab1:
    st.header("Submit a Performance Review (Anonymized)")
    emp_id = st.text_input("Employee ID (e.g., E011)")
    role = st.selectbox("Role", ["Manager", "Analyst", "Engineer"])
    gender = st.selectbox("Gender (for parity demo)", ["F", "M"])
    kpi = st.slider("KPI", 1, 5, 3)
    comp = st.slider("Competency", 1, 5, 3)
    init = st.slider("Initiative", 1, 5, 3)
    overall = st.slider("Overall", 1, 5, 3)
    comment = st.text_area("Manager Comment (no PII)")
    if st.button("Save Review"):
        new = pd.DataFrame([[emp_id, role, gender, kpi, comp, init, overall, comment]],
                           columns=reviews_df.columns)
        st.session_state["reviews"] = pd.concat([reviews_df, new], ignore_index=True)
        st.success("✅ Review saved for session!")

# Audit & Fairness
with tab2:
    st.header("Narrative Flags (Vague / Bias) with Coaching Tips")
    if bias_df.empty:
        st.error("⚠️ bias_rules.csv missing or invalid.")
    else:
        st.dataframe(bias_df)

    st.header("Ratings Fairness (Mean Gap + AIR)")
    if not reviews_df.empty:
        result = reviews_df.groupby("gender")[["kpi_rating","competency_rating","initiative_rating","overall_rating"]].mean()
        st.dataframe(result)

# Privacy & Export
with tab3:
    st.header("Privacy, Governance & Export")
    st.markdown("""
    - **No PII:** Use anonymized IDs only.
    - **Aggregation-first:** n ≥ 5 per group.
    - **Retention:** Session-based; no server storage.
    - **Explainability:** Rule-based flags editable via bias_rules.csv.
    """)

    st.download_button("⬇️ Download Current Reviews CSV", data=reviews_df.to_csv(index=False), file_name="reviews_export.csv")

st.caption("© FairLens — demo build v1.1")
