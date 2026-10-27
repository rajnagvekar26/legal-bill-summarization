"""
Unit Tests for Qwen2.5-1.5B Causal Decoder Architectural Guardrails.
CRITICAL GUARDRAILS:
1. Enforce left-padding.
2. Explicitly slice out the input prompt tokens before decoding so the model does not echo the bill text.
"""

import pytest
from src.models.qwen_summarizer import QwenSummarizer


def test_qwen_metadata():
    summarizer = QwenSummarizer(dry_run=True)
    meta = summarizer.get_metadata()
    assert meta.model_key == "qwen25_1_5b_causal_qlora"
    assert meta.arch_type == "causal_decoder"
    assert meta.max_source_length == 2048
    assert meta.total_params == 907081216


def test_qwen_prompt_formatting():
    summarizer = QwenSummarizer(dry_run=True)
    bill = "SECTION 1. SHORT TITLE. Clean Air Act Amendments."
    prompt = summarizer.format_prompt(bill)

    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert bill in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_qwen_slice_prompt_tokens_guardrail():
    """Verify prompt token slicing strips the input prompt completely."""
    try:
        import torch

        prompt_len = 50
        gen_len = 30
        batch_size = 2

        # Create mock output_ids: prompt tokens (values 1..50), gen tokens (values 500..530)
        prompt_tokens = torch.arange(1, prompt_len + 1).unsqueeze(0).repeat(batch_size, 1)
        generated_tokens = torch.arange(500, 500 + gen_len).unsqueeze(0).repeat(batch_size, 1)
        output_ids = torch.cat([prompt_tokens, generated_tokens], dim=-1)

        assert output_ids.shape == (batch_size, prompt_len + gen_len)

        # Apply slicing
        sliced = QwenSummarizer.slice_prompt_tokens(output_ids, prompt_len)

        assert sliced.shape == (batch_size, gen_len), f"Expected shape ({batch_size}, {gen_len}), got {sliced.shape}"
        assert torch.equal(sliced, generated_tokens), "Sliced output must exactly match generated tokens without prompt tokens"

        # Check prompt tokens are completely stripped
        assert not any(t in sliced[0].tolist() for t in range(1, prompt_len + 1)), "Prompt tokens must not appear in sliced output"

    except ImportError:
        pytest.skip("PyTorch not yet installed in active environment")


def test_qwen_summarize_dry_run():
    summarizer = QwenSummarizer(dry_run=True)
    res = summarizer.summarize("SECTION 101. Veterans mental health expansion act.")
    assert res.model_key == "qwen25_1_5b_causal_qlora"
    assert res.arch_type == "causal_decoder"
    assert len(res.summary_text) > 0
    assert res.latency_ms > 0
