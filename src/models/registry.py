"""
Model Registry and Factory for Legal Summarization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import pandas as pd

from src.models.bart_summarizer import BartSummarizer
from src.models.base import BaseSummarizer, ModelMetadata
from src.models.led_summarizer import LedSummarizer
from src.models.qwen_summarizer import QwenSummarizer

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[BaseSummarizer]] = {
    "bart_base_dense": BartSummarizer,
    "led_base_sparse_4k": LedSummarizer,
    "qwen25_1_5b_causal_qlora": QwenSummarizer,
}

MODEL_ALIASES: Dict[str, str] = {
    "bart": "bart_base_dense",
    "facebook/bart-base": "bart_base_dense",
    "led": "led_base_sparse_4k",
    "allenai/led-base-16384": "led_base_sparse_4k",
    "qwen": "qwen25_1_5b_causal_qlora",
    "qwen2.5": "qwen25_1_5b_causal_qlora",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen25_1_5b_causal_qlora",
}

_MODEL_CACHE: Dict[str, BaseSummarizer] = {}


def resolve_model_key(key: str) -> str:
    """Resolves model key or alias into canonical registry key."""
    clean = key.strip()
    if clean in MODEL_REGISTRY:
        return clean
    if clean in MODEL_ALIASES:
        return MODEL_ALIASES[clean]
    raise ValueError(
        f"Unknown model key '{key}'. Available models: {list(MODEL_REGISTRY.keys())}"
    )


def get_summarizer(
    model_key: str,
    device: str = "auto",
    dry_run: bool = False,
    adapter_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> BaseSummarizer:
    """
    Factory function returning an instantiated and loaded summarizer.
    """
    canonical_key = resolve_model_key(model_key)
    cache_key = f"{canonical_key}_{device}_{dry_run}"

    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    cls = MODEL_REGISTRY[canonical_key]
    kwargs = {"device": device, "dry_run": dry_run}
    if adapter_dir is not None:
        kwargs["adapter_dir"] = adapter_dir

    instance = cls(**kwargs)
    if use_cache:
        _MODEL_CACHE[cache_key] = instance
    return instance


def get_all_model_metadata() -> List[Dict]:
    """Returns metadata for all registered legal summarization models."""
    metadata_list = []
    for key, cls in MODEL_REGISTRY.items():
        temp = cls(dry_run=True)
        metadata_list.append(temp.get_metadata().to_dict())
    return metadata_list


def get_pareto_benchmarks() -> List[Dict]:
    """Loads Pareto benchmark metrics from artifacts CSV/JSON."""
    csv_candidates = [
        Path("artifacts/track_b_pareto_summary.csv"),
        Path("track_b_outputs/track_b_pareto_summary.csv"),
    ]
    for path in csv_candidates:
        if path.exists():
            df = pd.read_csv(path)
            return df.to_dict(orient="records")

    # Fallback to json checkpoint
    json_candidates = [
        Path("artifacts/track_b_metrics_checkpoint.json"),
        Path("track_b_outputs/track_b_metrics_checkpoint.json"),
    ]
    for path in json_candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(data.values())

    return []
