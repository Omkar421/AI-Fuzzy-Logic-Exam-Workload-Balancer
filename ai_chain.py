"""LangChain layer: extracts structured study signals and explains fuzzy output."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def _secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def _get_chain():
    token = _secret("HUGGINGFACEHUB_API_TOKEN") or _secret("HF_TOKEN")
    if not token:
        return None
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=_secret("HF_MODEL") or "HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=token,
        max_new_tokens=350,
        temperature=0.15,
    )
    model = ChatHuggingFace(llm=endpoint)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an education planning assistant. Return valid JSON only when asked. Be concise and do not invent missing numeric values."),
        ("human", "{instruction}"),
    ])
    return prompt | model | StrOutputParser()


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The model did not return JSON")
    return json.loads(match.group(0))


def extract_study_signals(user_text: str) -> Dict[str, Any]:
    chain = _get_chain()
    instruction = f"""Extract study-planning signals from this student message: {user_text!r}
Return exactly this JSON shape. Use null when a value is not stated. difficulty and confidence are 0-10, days_left is 0-30, study_hours is hours for this subject, weekly_capacity is total weekly study hours, subject is a short name, and explanation is one sentence.
{{\"subject\": \"...\", \"difficulty\": 0, \"days_left\": 0, \"confidence\": 0, \"study_hours\": 0, \"weekly_capacity\": 0, \"explanation\": \"...\"}}"""
    if chain is None:
        return _fallback_extract(user_text)
    try:
        result = _extract_json(chain.invoke({"instruction": instruction}))
        return _normalize(result)
    except Exception as exc:
        fallback = _fallback_extract(user_text)
        fallback["ai_note"] = f"LangChain extraction was unavailable for this request: {exc.__class__.__name__}."
        return fallback


def explain_result(subject: str, extracted: Dict[str, Any], result: Dict[str, Any]) -> str:
    chain = _get_chain()
    instruction = f"""Explain this fuzzy-logic study recommendation to a student in 3 short sentences. Mention the strongest signals and one practical next action. Do not claim it predicts marks.
Subject: {subject}
Inputs: {extracted}
Fuzzy result: {result}"""
    if chain is None:
        return _fallback_explanation(subject, extracted, result)
    try:
        return chain.invoke({"instruction": instruction}).strip()
    except Exception:
        return _fallback_explanation(subject, extracted, result)


def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("difficulty", "days_left", "confidence", "study_hours", "weekly_capacity"):
        value = data.get(key)
        try:
            data[key] = float(value) if value is not None else None
        except (TypeError, ValueError):
            data[key] = None
    return data


def _number(pattern: str, text: str, default: float | None) -> float | None:
    match = re.search(pattern, text.lower())
    return float(match.group(1)) if match else default


def _fallback_extract(text: str) -> Dict[str, Any]:
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    values = [float(value) for value in numbers]
    return {
        "subject": text.split(" ")[0].title() if text.strip() else "Exam subject",
        "difficulty": _number(r"(?:difficulty|hard|difficult)[^0-9]{0,15}(\d+(?:\.\d+)?)", text, values[0] if values else 6),
        "days_left": _number(r"(?:days?|day|in)\D{0,10}(\d+(?:\.\d+)?)", text, 10),
        "confidence": _number(r"(?:confidence|confident|weak)[^0-9]{0,15}(\d+(?:\.\d+)?)", text, 5),
        "study_hours": _number(r"(?:hours?|hrs?)[^0-9]{0,10}(\d+(?:\.\d+)?)", text, 6),
        "weekly_capacity": _number(r"(?:weekly|week)[^0-9]{0,15}(\d+(?:\.\d+)?)", text, 24),
        "explanation": "Fallback extraction used because no hosted LLM token is configured.",
        "ai_note": "Add HUGGINGFACEHUB_API_TOKEN to activate LangChain model extraction.",
    }


def _fallback_explanation(subject: str, extracted: Dict[str, Any], result: Dict[str, Any]) -> str:
    return (f"{subject} has a fuzzy workload risk of {result['risk_score']:.0f}/100 ({result['risk_label']}). "
            f"The strongest signals are difficulty {extracted.get('difficulty')}/10, {extracted.get('days_left')} days left, and confidence {extracted.get('confidence')}/10. "
            f"Start with a focused {min(60, max(25, int(result['recommended_hours'] * 4)))}-minute block and reassess after practice questions.")
