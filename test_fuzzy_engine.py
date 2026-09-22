from fuzzy_engine import calculate_workload


def test_high_pressure_subject_has_high_risk():
    result = calculate_workload(difficulty=8, days_left=5, confidence=4, study_hours=12, weekly_capacity=24)
    assert 0 <= result.risk_score <= 100
    assert result.risk_score >= 58
    assert result.risk_label in {"High", "Critical"}
    assert result.recommended_hours > 0
    assert any(value > 0 for value in result.rule_activations.values())


def test_low_pressure_subject_has_lower_risk():
    result = calculate_workload(difficulty=2, days_left=25, confidence=9, study_hours=2, weekly_capacity=24)
    assert 0 <= result.risk_score <= 100
    assert result.risk_score < 58
