"""
Unit Tests for LED Longformer Architectural Guardrails.
CRITICAL GUARDRAIL: LED requires an explicit global_attention_mask with index 0 set to 1.
"""

import pytest
from src.models.led_summarizer import LedSummarizer


def test_led_metadata():
    summarizer = LedSummarizer(dry_run=True)
    meta = summarizer.get_metadata()
    assert meta.model_key == "led_base_sparse_4k"
    assert meta.arch_type == "seq2seq_sparse_led"
    assert meta.max_source_length == 4096
    assert meta.total_params == 164056320


def test_led_global_attention_mask_guardrail():
    """Verify LED global_attention_mask has index 0 set to 1 and rest 0."""
    try:
        import torch

        # Single sample (batch_size=1, seq_len=128)
        input_ids = torch.randint(10, 5000, (1, 128))
        mask = LedSummarizer.create_global_attention_mask(input_ids)

        assert mask.shape == input_ids.shape, "Mask shape must match input_ids exactly"
        assert mask[0, 0].item() == 1, "Index 0 must have global attention flag set to 1"
        assert (mask[0, 1:] == 0).all().item(), "All non-zero positions must be 0 for sparse sliding window"

        # Multi-batch (batch_size=4, seq_len=256)
        batch_ids = torch.randint(10, 5000, (4, 256))
        batch_mask = LedSummarizer.create_global_attention_mask(batch_ids)

        assert batch_mask.shape == (4, 256)
        for b in range(4):
            assert batch_mask[b, 0].item() == 1, f"Batch item {b} index 0 must be 1"
            assert (batch_mask[b, 1:] == 0).all().item(), f"Batch item {b} remaining positions must be 0"

    except ImportError:
        # Fallback tensor check if torch is not yet available in current runner
        class MockTensor:
            def __init__(self, shape):
                self.shape = shape
                self._data = [[0] * shape[1] for _ in range(shape[0])]

        pytest.skip("PyTorch not yet installed in active environment")


def test_led_summarize_dry_run():
    summarizer = LedSummarizer(dry_run=True)
    res = summarizer.summarize("This is a legal bill test for LED Longformer sparse attention.")
    assert res.model_key == "led_base_sparse_4k"
    assert res.arch_type == "seq2seq_sparse_led"
    assert len(res.summary_text) > 0
    assert res.latency_ms > 0
