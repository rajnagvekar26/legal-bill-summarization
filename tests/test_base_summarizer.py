"""
Unit Tests for BaseSummarizer Interface and Model Registry.
"""

import pytest
from src.models.bart_summarizer import BartSummarizer
from src.models.base import BaseSummarizer, SummarizationResult
from src.models.led_summarizer import LedSummarizer
from src.models.qwen_summarizer import QwenSummarizer
from src.models.registry import (
    MODEL_REGISTRY,
    get_all_model_metadata,
    get_pareto_benchmarks,
    get_summarizer,
    resolve_model_key,
)


def test_registry_resolution():
    assert resolve_model_key("bart") == "bart_base_dense"
    assert resolve_model_key("led") == "led_base_sparse_4k"
    assert resolve_model_key("qwen") == "qwen25_1_5b_causal_qlora"
    assert resolve_model_key("facebook/bart-base") == "bart_base_dense"
    assert resolve_model_key("Qwen/Qwen2.5-1.5B-Instruct") == "qwen25_1_5b_causal_qlora"

    with pytest.raises(ValueError):
        resolve_model_key("nonexistent_model")


def test_model_factory_and_metadata():
    for key in MODEL_REGISTRY:
        summarizer = get_summarizer(key, dry_run=True)
        assert isinstance(summarizer, BaseSummarizer)
        meta = summarizer.get_metadata()
        assert meta.model_key == key
        assert meta.max_source_length > 0


def test_pareto_benchmarks_loading():
    benchmarks = get_pareto_benchmarks()
    assert len(benchmarks) >= 3
    keys = [b["model_key"] for b in benchmarks]
    assert "bart_base_dense" in keys
    assert "led_base_sparse_4k" in keys
    assert "qwen25_1_5b_causal_qlora" in keys


def test_bart_summarize_dry_run():
    bart = BartSummarizer(dry_run=True)
    res = bart.summarize("Short title: test bill text.")
    assert isinstance(res, SummarizationResult)
    assert res.model_key == "bart_base_dense"
    assert res.arch_type == "seq2seq_dense"
    assert res.latency_ms > 0
