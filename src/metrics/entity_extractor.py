"""
Legal Entity Extraction Module.

Extracts domain-specific legal entities using robust regex patterns:
- Sections: Section 123, Sec. 45, § 101, §§ 101-105, section 4(a)(1)
- U.S.C. Citations: 42 U.S.C. § 1983, 18 U.S.C. 1001, Title 28, United States Code, Section 1331
- Monetary Amounts: $1,000,000, $500 million, $12.5 billion
- Percentages: 15%, 2.5 percent, 0.75 percentage points
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class ExtractedEntities:
    sections: List[str] = field(default_factory=list)
    usc_citations: List[str] = field(default_factory=list)
    monetary_amounts: List[str] = field(default_factory=list)
    percentages: List[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return (
            len(self.sections)
            + len(self.usc_citations)
            + len(self.monetary_amounts)
            + len(self.percentages)
        )

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "sections": self.sections,
            "usc_citations": self.usc_citations,
            "monetary_amounts": self.monetary_amounts,
            "percentages": self.percentages,
        }

    def canonical_set(self) -> Set[str]:
        """Returns a normalized set of all entities prefixed by type."""
        canonical = set()
        for s in self.sections:
            norm = re.sub(r"\s+", " ", s.lower().strip())
            canonical.add(f"sec:{norm}")
        for u in self.usc_citations:
            norm = re.sub(r"[\s\.]+", " ", u.lower().strip())
            canonical.add(f"usc:{norm}")
        for m in self.monetary_amounts:
            norm = re.sub(r"\s+", " ", m.lower().strip().replace(",", ""))
            canonical.add(f"money:{norm}")
        for p in self.percentages:
            norm = re.sub(r"\s+", " ", p.lower().strip())
            canonical.add(f"pct:{norm}")
        return canonical


class LegalEntityExtractor:
    """Regex-based legal entity extractor for bills and statutory summaries."""

    # U.S.C. patterns: e.g. 42 U.S.C. § 1983, 18 U.S.C. 1001, Title 28, United States Code, Section 1331
    USC_PATTERNS = [
        re.compile(
            r"\b\d+\s+U\.?\s*S\.?\s*C\.?(?:\s*(?:App\.|§§?|Sec(?:tion)?\.?)?\s*[\d]+[a-zA-Z]*(?:\([a-zA-Z0-9]+\))*(?:[-–—]\d+[a-zA-Z]*)?(?:\s*et\s+seq\.)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bTitle\s+\d+[\s,]+United\s+States\s+Code[\s,]+(?:Section|Sec\.?|§§?)\s*[\d]+[a-zA-Z]*(?:\([a-zA-Z0-9]+\))*(?:[-–—]\d+[a-zA-Z]*)?",
            re.IGNORECASE,
        ),
    ]

    # Section patterns: § 101, §§ 101-105, Section 4(a)(1), Sec. 202
    SECTION_PATTERNS = [
        re.compile(
            r"(?:§§?\s*|\b(?:sections?|sec\.)\s+)(\d+[a-zA-Z]*(?:\([a-zA-Z0-9]+\))*(?:[-–—]\d+[a-zA-Z]*)?)",
            re.IGNORECASE,
        ),
    ]

    # Monetary patterns: $1,000,000, $500 million, $12.5 billion, $25,000
    MONEY_PATTERNS = [
        re.compile(
            r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:thousand|million|billion|trillion))?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:thousand|million|billion|trillion)\s*(?:dollars|USD)\b",
            re.IGNORECASE,
        ),
    ]

    # Percentage patterns: 15%, 2.5 percent, 0.75 percentage points
    PERCENTAGE_PATTERNS = [
        re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:%|percent(?:age\s+points)?|per\s+cent)(?!\w)",
            re.IGNORECASE,
        ),
    ]

    def __init__(self):
        pass

    def extract(self, text: str) -> ExtractedEntities:
        """Extracts all legal entities from the provided text."""
        if not text or not isinstance(text, str):
            return ExtractedEntities()

        # 1. Extract U.S.C. citations first (to avoid section sub-matches capturing USC section parts)
        usc_matches: List[str] = []
        usc_spans: List[Tuple[int, int]] = []
        for pat in self.USC_PATTERNS:
            for match in pat.finditer(text):
                usc_matches.append(match.group().strip())
                usc_spans.append(match.span())

        # 2. Extract Sections (excluding matches that overlap with USC spans)
        section_matches: List[str] = []
        for pat in self.SECTION_PATTERNS:
            for match in pat.finditer(text):
                m_start, m_end = match.span()
                overlaps_usc = any(
                    not (m_end <= u_start or m_start >= u_end)
                    for (u_start, u_end) in usc_spans
                )
                if not overlaps_usc:
                    section_matches.append(match.group().strip())

        # 3. Extract Monetary amounts
        money_matches: List[str] = []
        money_spans: List[Tuple[int, int]] = []
        for pat in self.MONEY_PATTERNS:
            for match in pat.finditer(text):
                m_start, m_end = match.span()
                if not any(not (m_end <= s or m_start >= e) for (s, e) in money_spans):
                    money_matches.append(match.group().strip())
                    money_spans.append((m_start, m_end))

        # 4. Extract Percentages
        pct_matches: List[str] = []
        pct_spans: List[Tuple[int, int]] = []
        for pat in self.PERCENTAGE_PATTERNS:
            for match in pat.finditer(text):
                p_start, p_end = match.span()
                if not any(not (p_end <= s or p_start >= e) for (s, e) in pct_spans):
                    pct_matches.append(match.group().strip())
                    pct_spans.append((p_start, p_end))

        # Deduplicate while preserving order
        def dedupe(seq: List[str]) -> List[str]:
            seen = set()
            out = []
            for item in seq:
                norm = item.strip()
                if norm and norm.lower() not in seen:
                    seen.add(norm.lower())
                    out.append(norm)
            return out

        return ExtractedEntities(
            sections=dedupe(section_matches),
            usc_citations=dedupe(usc_matches),
            monetary_amounts=dedupe(money_matches),
            percentages=dedupe(pct_matches),
        )

    def compute_recall(
        self, candidate_text: str, reference_text: str
    ) -> Dict[str, float]:
        """
        Computes Legal Entity Recall:
        Recall = |E_candidate ∩ E_reference| / |E_reference|
        Also computes precision and F1.
        """
        ref_entities = self.extract(reference_text)
        cand_entities = self.extract(candidate_text)

        ref_set = ref_entities.canonical_set()
        cand_set = cand_entities.canonical_set()

        if not ref_set:
            # If reference has no legal entities, recall is 1.0 if candidate also has none, else 1.0 baseline
            recall = 1.0 if not cand_set else 1.0
            precision = 1.0 if not cand_set else 0.0
            f1 = 1.0 if not cand_set else 0.0
            overlap_count = 0
        else:
            intersection = cand_set.intersection(ref_set)
            overlap_count = len(intersection)
            recall = overlap_count / len(ref_set)
            precision = overlap_count / len(cand_set) if cand_set else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        return {
            "entity_recall": round(recall * 100, 2),
            "entity_precision": round(precision * 100, 2),
            "entity_f1": round(f1 * 100, 2),
            "ref_entity_count": len(ref_set),
            "cand_entity_count": len(cand_set),
            "overlap_count": overlap_count,
        }
