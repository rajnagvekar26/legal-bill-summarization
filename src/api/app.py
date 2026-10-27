"""
FastAPI Backend for Legal Bill Summarization and Pareto Benchmarking.
"""

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    ExtractEntitiesRequest,
    ExtractEntitiesResponse,
    ModelInfoItem,
    ParetoBenchmarkResponse,
    SingleSummaryItem,
    SummarizeRequest,
    SummarizeResponse,
)
from src.metrics.entity_extractor import LegalEntityExtractor
from src.metrics.evaluator import LegalEvaluator
from src.models.registry import (
    MODEL_REGISTRY,
    get_all_model_metadata,
    get_pareto_benchmarks,
    get_summarizer,
    resolve_model_key,
)

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Cross-Architecture Legal Bill Summarization Engine",
    description="Inference, Multi-Axis Evaluation, and Pareto Benchmark API for BART, LED, and Qwen2.5",
    version="1.0.0",
)

# Enable CORS for Streamlit and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluator = LegalEvaluator()
entity_extractor = LegalEntityExtractor()


@app.get("/")
def root():
    return {
        "service": "Cross-Architecture Legal Bill Summarization Engine",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/api/health")
def health():
    import torch

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "status": "healthy",
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": device_name,
        "supported_models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/api/models", response_model=List[ModelInfoItem])
def get_models():
    """Returns metadata for all supported models and architectural guardrails."""
    return get_all_model_metadata()


@app.get("/api/benchmarks/pareto", response_model=ParetoBenchmarkResponse)
def get_benchmarks():
    """Returns the Track B Kaggle benchmark dataset for Pareto frontier analysis."""
    benchmarks = get_pareto_benchmarks()
    if not benchmarks:
        raise HTTPException(
            status_code=404, detail="Benchmark dataset track_b_pareto_summary.csv not found"
        )
    return ParetoBenchmarkResponse(benchmarks=benchmarks, total_models=len(benchmarks))


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize_text(req: SummarizeRequest):
    """
    Summarizes legal bill text.
    Runs selected models sequentially and returns comparative results.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    try:
        requested_keys = (
            req.models
            if req.models is not None
            else (
                list(MODEL_REGISTRY.keys())
                if req.model_key.lower() == "all"
                else [req.model_key]
            )
        )
        if not requested_keys:
            raise HTTPException(status_code=400, detail="Select at least one model.")
        target_keys = list(dict.fromkeys(resolve_model_key(key) for key in requested_keys))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    results: List[SingleSummaryItem] = []
    fastest_model = None
    min_latency = float("inf")

    for key in target_keys:
        try:
            summarizer = get_summarizer(key, dry_run=req.dry_run)
            res = summarizer.summarize(
                text=req.text,
                max_new_tokens=req.max_new_tokens,
                min_new_tokens=req.min_new_tokens,
                num_beams=req.num_beams,
                temperature=req.temperature,
                do_sample=req.do_sample,
            )
            item = SingleSummaryItem(**res.to_dict())
            results.append(item)

            if res.latency_ms < min_latency:
                min_latency = res.latency_ms
                fastest_model = key
        except Exception as e:
            logger.error("Error executing summarization for model %s: %s", key, e)
            raise HTTPException(
                status_code=500, detail=f"Inference error on model '{key}': {str(e)}"
            )

    return SummarizeResponse(
        results=results,
        fastest_model=fastest_model,
        min_latency_ms=round(min_latency, 2) if min_latency < float("inf") else None,
    )


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate_summary(req: EvaluateRequest):
    """
    Multi-axis evaluation: ROUGE-1/2/L, BERTScore F1, and Legal Entity Recall.
    """
    if not req.candidate_summary.strip() or not req.reference_summary.strip():
        raise HTTPException(
            status_code=400, detail="Both candidate and reference summaries are required."
        )

    try:
        eval_result = evaluator.evaluate(
            candidate=req.candidate_summary,
            reference=req.reference_summary,
            source_text=req.source_text,
            use_bertscore=req.use_bertscore,
        )
        return EvaluateResponse(**eval_result.to_dict())
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Evaluation pipeline failed: {str(e)}")


@app.post("/api/extract-entities", response_model=ExtractEntitiesResponse)
def extract_legal_entities(req: ExtractEntitiesRequest):
    """
    Extracts sections, U.S.C. citations, monetary amounts, and percentages using regex.
    """
    entities = entity_extractor.extract(req.text)
    return ExtractEntitiesResponse(
        sections=entities.sections,
        usc_citations=entities.usc_citations,
        monetary_amounts=entities.monetary_amounts,
        percentages=entities.percentages,
        total_count=entities.total_count,
    )
