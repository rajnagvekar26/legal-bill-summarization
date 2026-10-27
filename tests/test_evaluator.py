"""
Unit Tests for Multi-Axis Legal Evaluator.
Tests ROUGE, BERTScore (proxy/real), and Legal Entity Recall integration.
"""

import pytest
from src.metrics.evaluator import LegalEvaluationResult, LegalEvaluator


@pytest.fixture
def evaluator():
    return LegalEvaluator(use_bertscore=False)


def test_rouge_calculation(evaluator):
    cand = "The Administrator shall award municipal water infrastructure grants under 33 U.S.C. 1251."
    ref = "The Administrator shall award municipal water infrastructure grants under 33 U.S.C. 1251."
    scores = evaluator.compute_rouge(cand, ref)

    assert scores["rouge1"] == 100.0
    assert scores["rouge2"] == 100.0
    assert scores["rougeL"] == 100.0


def test_rouge_partial_overlap(evaluator):
    cand = "The EPA Administrator awards clean water grants."
    ref = "The Administrator shall award municipal water infrastructure grants under 33 U.S.C. 1251."
    scores = evaluator.compute_rouge(cand, ref)

    assert 0.0 < scores["rouge1"] < 100.0
    assert scores["rougeL"] > 0.0


def test_full_evaluation_pipeline(evaluator):
    cand = (
        "Clean Water Infrastructure Act: Section 104 authorizes $450,000,000 for FY2027 "
        "pursuant to 33 U.S.C. 1251, reserving 15% for rural projects."
    )
    ref = (
        "Clean Water Infrastructure Act: Section 104 authorizes $450,000,000 for FY2027 "
        "pursuant to 33 U.S.C. 1251 and Section 2, reserving 15% for rural projects."
    )

    res = evaluator.evaluate(cand, ref)

    assert isinstance(res, LegalEvaluationResult)
    assert res.rouge1 > 80.0
    assert res.rougeL > 80.0
    assert res.legal_entity_recall_pct > 0.0
    assert len(res.matched_entities) > 0
    assert "sections" in res.candidate_entities
    assert "usc_citations" in res.candidate_entities
