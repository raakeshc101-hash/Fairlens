# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="FairLens — Performance Review Auditor", layout="wide")

# =========================================================
# Utilities
# =========================================================

def _safe_read_pipe_csv(path: str, expected_cols: list[str]) -> pd.DataFrame:
    """
    Robustly read a pipe-delimited CSV. Falls back to manual split if needed.
    Guarantees expected columns exist in return value.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=expected_cols)

    try:
        df = pd.read_csv(p, sep="|", engine="python")
        # If the file collapsed into a single column (e.g., quoted header),
        # recover by raw read and manual split.
        if list(df.columns) == ["phrase|category|context_rule|tip"] or df.shape[1] == 1:
            raw = pd.read_csv(p, header=None, engine="python", dtype=str)
            df = raw.iloc[:, 0].str.split("|", expand=True)
            df.columns = expected_cols
    except Exception:
        # Try a last-chance manual split
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f.readlines()]
        parts = [ln.split("|") for ln in lines if ln.strip()]
        if not parts:
            return pd.DataFrame(columns=expected_cols)
        header = parts[0]
        rows = parts[1:]
        if len(header) == len(expected_cols):
            df = pd.DataFrame(rows, columns=header)
        else:
            return pd.DataFrame(columns=expected_cols)

    # Keep only expected columns; add missing as blanks
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[expected_cols].fillna("")
    return df


# =========================================================
# Data Loading
# =========================================================

@st.cache_data
def load_bias_rules(path: str = "bias_rules.csv") -> pd.DataFrame:
    cols = ["phrase", "category", "context_rule", "tip"]
    return _safe_read_pipe_csv(path, cols)

@st.cache_data
def seed_reviews() -> pd.DataFrame:
    # Small starter dataset so the fairness tab has something to show
    data = [
        ["E001", "Manager", "F", 4, 4, 4, 4, "Strong potential; team player."],
        ["E002", "Manager", "M", 3, 4, 3, 3, "Good attitude; average execution."],
        ["E003", "Manager", "F", 4, 3, 3, 3, "Works well under pressure; sometimes too energetic."],
        ["E004", "Manager", "M", 3, 3, 3, 3, "Not a good cultural fit. Hard worker though."],
        ["E005", "Manager", "F", 4, 4, 4, 4, "Great attitude; on-time delivery."],
        ["E009", "Analyst", "F", 3, 3, 3, 3, "Positive without evidence."]
    ]
    cols = ["employee_id","role","gender","kpi_rating","competency_rating","initiative_rating","overall_rating","comment"]
    return pd.DataFrame(data, columns=cols)

def get_reviews_df() -> pd.DataFrame:
    if "reviews_df" not in st.session_state:
        st.session_state["reviews_df"] = seed_reviews().copy()
    return st.session_state["reviews_df"]

def save_review_row(row: dict):
    df = get_reviews_df()
    st.session_state["reviews_df"] = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# =========================================================
# Text flagging helpers
# =========================================================

POSITIVE_WORDS = set(["good","great","improved","improving","excellent","amazing","nice","pleasant","awesome"])
BEHAVIOR_VERBS = set(["completed","delivered","reduced","increased","launched","led","designed","documented","resolved",
                      "trained","implemented","created","shipped","debugged","wrote","analyzed","interviewed"])

def is_positive_without_evidence(text: str) -> bool:
    """
    Simple heuristic: contains a positive adjective but none of the behavior verbs.
    Used to flag 'positive-without-evidence' as a vague statement.
    """
    t = (text or "").lower()
    if any(w in t for w in POSITIVE_WORDS) and not any(v in t for v in BEHAVIOR_VERBS):
        return True
    return False

def apply_bias_rules_to_comment(text: str, rules_df: pd.DataFrame) -> list[dict]:
    """
    Returns a list of matches of the form:
    {"phrase":..., "category":..., "tip":...}
    - 'context_rule' column controls special logic:
        - 'pattern' => treat 'phrase' as regex substring (case-insensitive)
        - 'if_gender_female' => caller should filter by gender before calling
        - 'always' (default) => simple 'phrase' substring match (case-insensitive)
    Special pseudo-rule: 'positive-without-evidence'
    """
    t = (text or "").lower()
    out = []

    # learned rule: positive-without-evidence (phrase stored in lexicon)
    pwe_rows = rules_df[rules_df["phrase"].str.lower() == "positive without evidence"]
    if not pwe_rows.empty and is_positive_without_evidence(t):
        r = pwe_rows.iloc[0]
        out.append({"phrase": "positive-without-evidence", "category": r["category"], "tip": r["tip"]})

    # literal / regex matches from lexicon
    for _, r in rules_df.iterrows():
        phrase = str(r["phrase"]).strip()
        cat = str(r["category"]).strip()
        ctx = (str(r["context_rule"]).strip() or "always").lower()
        tip = str(r["tip"]).strip()
        if phrase.lower() == "positive without evidence":
            continue  # already handled above
        if not phrase:
            continue

        try:
            if ctx == "pattern":
                import re
                if re.search(phrase, t, flags=re.IGNORECASE):
                    out.append({"phrase": phrase, "category": cat, "tip": tip})
            else:
                # default: case-insensitive substring
                if phrase.lower() in t:
                    out.append({"phrase": phrase, "category": cat, "tip": tip})
        except Exception:
            # ignore a bad regex row rather than crash the app
            pass

    return out

# =========================================================
# UI
# =========================================================

st.title("FairLens — Performance Review Auditor (v1.1)")
st.caption("Detect vague / biased phrases, coach rewrites, and check group fairness on ratings. No PII. n ≥ 5 to aggregate.")

bias_rules = load_bias_rules()
reviews_df = get_reviews_df()

tab_submit, tab_audit, tab_privacy = st.tabs(["Submit Review", "Audit & Fairness", "Privacy & Export"])

# --------------------------
# Submit Review
# --------------------------
with tab_submit:
    st.subheader("Submit a Performance Review (Anonymized)")

    with st.form("review_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            emp_id = st.text_input("Employee ID (e.g., E011)")
        with c2:
            role = st.selectbox("Role", ["Manager", "Analyst", "Engineer"])
        with c3:
            gender = st.selectbox("Gender (for parity demo)", ["F", "M"])

        c4, c5, c6, c7 = st.columns(4)
        with c4:
            kpi = st.slider("KPI", 1, 5, 3)
        with c5:
            comp = st.slider("Competency", 1, 5, 3)
        with c6:
            init = st.slider("Initiative", 1, 5, 3)
        with c7:
            overall = st.slider("Overall", 1, 5, 3)

        comment = st.text_area("Manager Comment (no PII)")

        submitted = st.form_submit_button("Save Review")
        if submitted:
            if not emp_id.strip():
                st.warning("Please enter an anonymized Employee ID.")
            else:
                save_review_row({
                    "employee_id": emp_id.strip(),
                    "role": role, "gender": gender,
                    "kpi_rating": kpi, "competency_rating": comp,
                    "initiative_rating": init, "overall_rating": overall,
                    "comment": comment.strip()
                })
                st.success("✅ Review saved in session.")

    st.divider()
    st.write("**Current (Anonymized) Reviews in Session**")
    st.dataframe(get_reviews_df(), use_container_width=True)

# --------------------------
# Audit & Fairness
# --------------------------
with tab_audit:
    st.subheader("Narrative Flags (Vague / Bias) with Coaching Tips")

    if bias_rules.empty:
        st.error("`bias_rules.csv` missing or invalid. Expect columns: 'phrase|category|context_rule|tip' (pipe-separated).")
    else:
        # Build flag rows for each comment
        rows = []
        for _, r in get_reviews_df().iterrows():
            matches = apply_bias_rules_to_comment(r["comment"], bias_rules)
            for m in matches:
                rows.append({
                    "employee_id": r["employee_id"],
                    "role": r["role"],
                    "gender": r["gender"],
                    "phrase": m["phrase"],
                    "category": m["category"],
                    "tip": m["tip"]
                })
        flags_df = pd.DataFrame(rows, columns=["employee_id","role","gender","phrase","category","tip"])
        st.dataframe(flags_df if not flags_df.empty else pd.DataFrame(columns=["employee_id","role","gender","phrase","category","tip"]),
                     use_container_width=True)

    st.divider()
    st.subheader("Ratings Fairness (Mean Gap + AIR)")

    if len(get_reviews_df()) < 5:
        st.info("Need at least 5 rows to inspect group fairness meaningfully.")
    else:
        by = st.selectbox("Compare by group", ["gender", "role"])
        agg = get_reviews_df().groupby(by)[["kpi_rating","competency_rating","initiative_rating","overall_rating"]].agg(["mean","count"])
        # flatten columns for display
        agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns]
        st.dataframe(agg, use_container_width=True)

        # Mean overall gap (simple signal)
        means = get_reviews_df().groupby(by)["overall_rating"].mean()
        if len(means) >= 2:
            max_mean = means.max()
            min_mean = means.min()
            gap = round(max_mean - min_mean, 2)
            st.info(f"Mean Overall Rating Gap = {gap} (on 1–5 scale). ≥ 0.30 may warrant a calibration review.")
        else:
            st.info("Not enough groups for mean gap.")

        st.subheader("Meets/Exceeds Parity (AIR proxy)")
        thr = st.slider("Meets/Exceeds threshold (Overall ≥)", 1, 5, 3)
        rates = (get_reviews_df()["overall_rating"] >= thr).groupby(get_reviews_df()[by]).mean().rename("rate").reset_index()
        st.dataframe(rates, use_container_width=True)
        if len(rates) >= 2:
            air = rates["rate"].min() / max(rates["rate"].max(), 1e-9)
            st.success(f"AIR (min/max) = {air:.2f} — rule-of-thumb ≥ 0.80.")
        else:
            st.info("Not enough groups for AIR.")

# --------------------------
# Privacy & Export
# --------------------------
with tab_privacy:
    st.subheader("Privacy, Governance & Export")
    st.markdown("""
- **No PII:** Use anonymized IDs only.
- **Aggregation-first:** Group metrics only when **n ≥ 5** per group.
- **Retention:** Session-based; no server storage. Export locally if needed.
- **Explainability:** Rule-based flags are transparent and editable via **bias_rules.csv**.
- **Compliance touchpoints (demo):** Title VII principles; AIR (4/5ths) as a rule-of-thumb.
- **Rules version:** `v1.1-lexicon-60`
    """)

    st.download_button(
        "⬇️ Download Current Reviews CSV",
        data=get_reviews_df().to_csv(index=False),
        file_name="reviews_export.csv",
        mime="text/csv"
    )

st.caption("© FairLens — educational demo. Replace lexicon/thresholds with org standards before deployment.")
