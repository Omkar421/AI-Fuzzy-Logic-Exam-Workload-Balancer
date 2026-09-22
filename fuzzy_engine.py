"""Explainable fuzzy inference engine for exam workload balancing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass
class FuzzyResult:
    risk_score: float
    risk_label: str
    recommended_hours: float
    memberships: Dict[str, Dict[str, float]]
    rule_activations: Dict[str, float]


def _build_system():
    difficulty = ctrl.Antecedent(np.arange(0, 11, 1), "difficulty")
    urgency = ctrl.Antecedent(np.arange(0, 31, 1), "urgency")
    confidence = ctrl.Antecedent(np.arange(0, 11, 1), "confidence")
    workload = ctrl.Antecedent(np.arange(0, 21, 1), "workload")
    risk = ctrl.Consequent(np.arange(0, 101, 1), "risk")

    difficulty["low"] = fuzz.trimf(difficulty.universe, [0, 0, 5])
    difficulty["medium"] = fuzz.trimf(difficulty.universe, [3, 5.5, 8])
    difficulty["high"] = fuzz.trimf(difficulty.universe, [6, 10, 10])

    urgency["low"] = fuzz.trimf(urgency.universe, [10, 30, 30])
    urgency["medium"] = fuzz.trimf(urgency.universe, [5, 14, 23])
    urgency["high"] = fuzz.trimf(urgency.universe, [0, 0, 12])

    confidence["low"] = fuzz.trimf(confidence.universe, [0, 0, 5])
    confidence["medium"] = fuzz.trimf(confidence.universe, [3, 5.5, 8])
    confidence["high"] = fuzz.trimf(confidence.universe, [6, 10, 10])

    workload["low"] = fuzz.trimf(workload.universe, [0, 0, 7])
    workload["medium"] = fuzz.trimf(workload.universe, [3, 9, 15])
    workload["high"] = fuzz.trimf(workload.universe, [10, 20, 20])

    risk["low"] = fuzz.trimf(risk.universe, [0, 0, 38])
    risk["medium"] = fuzz.trimf(risk.universe, [25, 48, 70])
    risk["high"] = fuzz.trimf(risk.universe, [58, 78, 92])
    risk["critical"] = fuzz.trimf(risk.universe, [82, 100, 100])

    rules = [
        ctrl.Rule(difficulty["high"] & urgency["high"], risk["critical"], label="high difficulty and high urgency"),
        ctrl.Rule(confidence["low"] & workload["high"], risk["critical"], label="low confidence and high study load"),
        ctrl.Rule(difficulty["high"] & confidence["low"], risk["high"], label="high difficulty and low confidence"),
        ctrl.Rule(urgency["high"], risk["high"], label="high urgency"),
        ctrl.Rule(workload["high"] & confidence["medium"], risk["high"], label="high study load and medium confidence"),
        ctrl.Rule(difficulty["medium"] & urgency["medium"], risk["medium"], label="medium difficulty and medium urgency"),
        ctrl.Rule(difficulty["low"] & confidence["high"] & urgency["low"], risk["low"], label="easy, confident, and not urgent"),
        ctrl.Rule(workload["low"] & confidence["high"], risk["low"], label="low study load and high confidence"),
    ]
    return (difficulty, urgency, confidence, workload, risk), ctrl.ControlSystem(rules)


def calculate_workload(difficulty: float, days_left: float, confidence: float, study_hours: float, weekly_capacity: float = 24) -> FuzzyResult:
    variables, system = _build_system()
    difficulty_var, urgency_var, confidence_var, workload_var, risk_var = variables
    simulation = ctrl.ControlSystemSimulation(system)
    simulation.input["difficulty"] = float(np.clip(difficulty, 0, 10))
    simulation.input["urgency"] = float(np.clip(days_left, 0, 30))
    simulation.input["confidence"] = float(np.clip(confidence, 0, 10))
    simulation.input["workload"] = float(np.clip(study_hours, 0, 20))
    simulation.compute()

    score = float(simulation.output["risk"])
    label = "Critical" if score >= 78 else "High" if score >= 58 else "Watch" if score >= 38 else "Low"
    total_input = max(float(study_hours), 1.0)
    recommended_hours = min(float(weekly_capacity), max(1.0, weekly_capacity * (0.45 + score / 180)))

    memberships = {
        "difficulty": {term: float(fuzz.interp_membership(difficulty_var.universe, difficulty_var[term].mf, difficulty)) for term in ("low", "medium", "high")},
        "urgency": {term: float(fuzz.interp_membership(urgency_var.universe, urgency_var[term].mf, days_left)) for term in ("low", "medium", "high")},
        "confidence": {term: float(fuzz.interp_membership(confidence_var.universe, confidence_var[term].mf, confidence)) for term in ("low", "medium", "high")},
        "study_load": {term: float(fuzz.interp_membership(workload_var.universe, workload_var[term].mf, study_hours)) for term in ("low", "medium", "high")},
    }
    rule_activations = {}
    for rule in system.rules:
        try:
            activation = float(rule.aggregate_firing[simulation].item())
        except Exception:
            activation = 0.0
        rule_activations[rule.label or "unnamed rule"] = activation
    _ = total_input  # retained as a readable part of the model explanation
    return FuzzyResult(score, label, round(recommended_hours, 1), memberships, rule_activations)
