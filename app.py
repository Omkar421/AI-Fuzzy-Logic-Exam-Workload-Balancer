from __future__ import annotations

import streamlit as st

from ai_chain import explain_result, extract_study_signals
from fuzzy_engine import calculate_workload

st.set_page_config(page_title="Exam Workload Balancer", page_icon="🧠", layout="centered")

st.title("AI + Fuzzy Logic Exam Workload Balancer")
st.caption("A LangChain-assisted, explainable fuzzy inference system for exam planning.")

with st.sidebar:
    st.header("How it works")
    st.write("1. LangChain reads the student’s natural-language message.\n2. The fuzzy engine fuzzifies the signals.\n3. Rules are evaluated and defuzzified.\n4. LangChain explains the recommendation.")
    st.divider()
    st.info("For the live AI component, add HUGGINGFACEHUB_API_TOKEN in Streamlit Secrets. The app remains runnable in demo mode without it.")

example = "I have Data Structures in 5 days. It is difficulty 8, my confidence is 4, I need 12 hours, and I can study 24 hours weekly."
user_text = st.text_area("Describe your exam workload in your own words", value=example, height=120)

if st.button("Analyze workload", type="primary", use_container_width=True):
    extracted = extract_study_signals(user_text)
    st.session_state["extracted"] = extracted
    st.session_state["user_text"] = user_text

if "extracted" in st.session_state:
    extracted = st.session_state["extracted"]
    st.subheader("1. LangChain language understanding")
    st.write(extracted.get("ai_note", "LangChain extracted these structured signals from the message."))
    cols = st.columns(5)
    fields = [("Subject", extracted.get("subject", "Exam subject")), ("Difficulty", extracted.get("difficulty") or 6), ("Days left", extracted.get("days_left") or 10), ("Confidence", extracted.get("confidence") or 5), ("Study hours", extracted.get("study_hours") or 6)]
    for column, (label, value) in zip(cols, fields):
        column.metric(label, value)

    difficulty = float(extracted.get("difficulty") or 6)
    days_left = float(extracted.get("days_left") or 10)
    confidence = float(extracted.get("confidence") or 5)
    study_hours = float(extracted.get("study_hours") or 6)
    weekly_capacity = float(extracted.get("weekly_capacity") or 24)
    result = calculate_workload(difficulty, days_left, confidence, study_hours, weekly_capacity)
    result_dict = {"risk_score": result.risk_score, "risk_label": result.risk_label, "recommended_hours": result.recommended_hours}

    st.subheader("2. Fuzzy inference result")
    score_col, plan_col = st.columns(2)
    score_col.metric("Defuzzified workload risk", f"{result.risk_score:.0f}/100", result.risk_label)
    plan_col.metric("Recommended focus time", f"{result.recommended_hours:.1f} hours", "this week")
    st.progress(min(result.risk_score / 100, 1.0), text=f"Risk status: {result.risk_label}")

    with st.expander("Show fuzzification memberships"):
        st.json(result.memberships)
    with st.expander("Show rule activations"):
        st.json({key: round(value, 3) for key, value in result.rule_activations.items() if value > 0})

    st.subheader("3. LangChain explanation")
    explanation = explain_result(extracted.get("subject", "Exam subject"), extracted, result_dict)
    st.success(explanation)

st.divider()
st.markdown("**Viva note:** The LLM performs natural-language extraction and explanation. The risk score is produced by the fuzzy system, not by the LLM.")
