"""
Unit Tests for Custom Regex-Based Legal Entity Extraction and Recall.
"""

import pytest
from src.metrics.entity_extractor import LegalEntityExtractor


@pytest.fixture
def extractor():
    return LegalEntityExtractor()


def test_extract_sections(extractor):
    sample = (
        "Pursuant to Section 104(a)(1) of the Act and Sec. 45, the rules in "
        "§ 101 and §§ 101-105 shall apply. In addition, section 12(b) applies."
    )
    entities = extractor.extract(sample)

    # Check sections detected
    expected_sections = ["Section 104(a)(1)", "Sec. 45", "§ 101", "§§ 101-105", "section 12(b)"]
    for exp in expected_sections:
        assert any(exp.lower() in s.lower() for s in entities.sections), f"Missing section {exp}"


def test_extract_usc_citations(extractor):
    sample = (
        "Federal jurisdiction is grounded in 42 U.S.C. § 1983 and 18 U.S.C. 1001. "
        "Further relief is sought under Title 28, United States Code, Section 1331."
    )
    entities = extractor.extract(sample)

    assert any("42 U.S.C." in u for u in entities.usc_citations)
    assert any("18 U.S.C." in u for u in entities.usc_citations)
    assert any("Title 28, United States Code" in u for u in entities.usc_citations)


def test_extract_monetary_amounts(extractor):
    sample = (
        "Authorizes $450,000,000 for FY2027, $12.5 billion for infrastructure, "
        "and $25,000 in minor fines. Additional 500 million dollars are held in reserve."
    )
    entities = extractor.extract(sample)

    assert any("$450,000,000" in m for m in entities.monetary_amounts)
    assert any("$12.5 billion" in m for m in entities.monetary_amounts)
    assert any("$25,000" in m for m in entities.monetary_amounts)
    assert any("500 million dollars" in m for m in entities.monetary_amounts)


def test_extract_percentages(extractor):
    sample = (
        "Requires at least 15% dedicated to rural areas, a 2.5 percent administrative fee, "
        "and an increase of 0.75 percentage points."
    )
    entities = extractor.extract(sample)

    assert any("15%" in p for p in entities.percentages)
    assert any("2.5 percent" in p for p in entities.percentages)
    assert any("0.75 percentage points" in p for p in entities.percentages)


def test_legal_entity_recall_perfect_match(extractor):
    ref = "Pursuant to 42 U.S.C. § 1983, $500,000,000 is authorized with a 15% set-aside under Section 4."
    cand = "Pursuant to 42 U.S.C. § 1983, $500,000,000 is authorized with a 15% set-aside under Section 4."
    metrics = extractor.compute_recall(candidate_text=cand, reference_text=ref)

    assert metrics["entity_recall"] == 100.0
    assert metrics["entity_precision"] == 100.0
    assert metrics["entity_f1"] == 100.0
    assert metrics["overlap_count"] == metrics["ref_entity_count"]


def test_legal_entity_recall_partial(extractor):
    ref = "Under Section 101, $1,000,000 is granted with 10% fee pursuant to 18 U.S.C. 1001."
    # Candidate only mentions Section 101 and $1,000,000 (2 out of 4)
    cand = "Under Section 101, $1,000,000 is granted."
    metrics = extractor.compute_recall(candidate_text=cand, reference_text=ref)

    assert metrics["entity_recall"] == 50.0
    assert metrics["entity_precision"] == 100.0
    assert metrics["overlap_count"] == 2
