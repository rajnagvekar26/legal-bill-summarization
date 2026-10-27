"""
Pydantic Request and Response Schemas for Legal Summarization API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class SummarizeRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Raw text of the legal bill to summarize")
    bill_text: Optional[str] = Field(default=None, description="Alias for text")
    models: Optional[List[str]] = Field(
        default=None,
        description="Optional model keys to run for a comparative request",
    )
    model_key: str = Field(
        default="bart_base_dense",
        description="Target model: bart_base_dense, led_base_sparse_4k, qwen25_1_5b_causal_qlora, or 'all'",
    )
    max_new_tokens: int = Field(default=256, ge=16, le=1024)
    min_new_tokens: int = Field(default=32, ge=8, le=256)
    num_beams: int = Field(default=4, ge=1, le=8)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    do_sample: bool = Field(default=False)
    dry_run: bool = Field(
        default=False,
        description="Run in deterministic dry-run mode for testing and latency simulation",
    )

    @model_validator(mode="after")
    def validate_text(self):
        content = self.text or self.bill_text
        if not content or not content.strip():
            raise ValueError("Either 'text' or 'bill_text' must be provided.")
        self.text = content.strip()
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError("'min_new_tokens' must be less than or equal to 'max_new_tokens'.")
        return self


class SingleSummaryItem(BaseModel):
    summary_text: str
    model_key: str
    hf_model_id: str
    arch_type: str
    latency_ms: float
    ms_per_token: float
    prompt_tokens: int
    summary_tokens: int
    total_tokens: int
    device: str
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class SummarizeResponse(BaseModel):
    results: List[SingleSummaryItem]
    fastest_model: Optional[str] = None
    min_latency_ms: Optional[float] = None


class EvaluateRequest(BaseModel):
    candidate_summary: Optional[str] = Field(default=None, description="Generated summary candidate")
    reference_summary: Optional[str] = Field(default=None, description="Ground-truth reference summary")
    prediction: Optional[str] = Field(default=None, description="Alias for candidate_summary")
    reference: Optional[str] = Field(default=None, description="Alias for reference_summary")
    source_text: Optional[str] = Field(
        default=None, description="Optional full legal bill source text"
    )
    use_bertscore: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_summaries(self):
        cand = self.candidate_summary or self.prediction
        ref = self.reference_summary or self.reference
        if not cand or not cand.strip():
            raise ValueError("Either 'candidate_summary' or 'prediction' must be provided.")
        if not ref or not ref.strip():
            raise ValueError("Either 'reference_summary' or 'reference' must be provided.")
        self.candidate_summary = cand.strip()
        self.reference_summary = ref.strip()
        return self


class EvaluateResponse(BaseModel):
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_f1: float
    bertscore_precision: float
    bertscore_recall: float
    legal_entity_recall_pct: float
    legal_entity_precision_pct: float
    legal_entity_f1_pct: float
    candidate_entities: Dict[str, List[str]]
    reference_entities: Dict[str, List[str]]
    matched_entities: List[str]
    missing_entities: List[str]


class ExtractEntitiesRequest(BaseModel):
    text: str = Field(..., description="Text to extract legal entities from")


class ExtractEntitiesResponse(BaseModel):
    sections: List[str]
    usc_citations: List[str]
    monetary_amounts: List[str]
    percentages: List[str]
    total_count: int


class ModelInfoItem(BaseModel):
    model_key: str
    hf_model_id: str
    arch_type: str
    max_source_length: int
    adapter_path: Optional[str]
    adapter_type: str
    total_params: int
    trainable_params: int
    trainable_pct: float
    config_fingerprint: Optional[str]
    description: str


class ParetoBenchmarkResponse(BaseModel):
    benchmarks: List[Dict[str, Any]]
    total_models: int
