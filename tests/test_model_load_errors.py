import builtins

import pytest

from src.models.bart_summarizer import BartSummarizer
from src.models.led_summarizer import LedSummarizer
from src.models.qwen_summarizer import QwenSummarizer


@pytest.mark.parametrize(
    "summarizer_class",
    [BartSummarizer, LedSummarizer, QwenSummarizer],
)
def test_real_model_load_failure_does_not_enable_dry_run(
    monkeypatch, summarizer_class
):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    summarizer = summarizer_class(device="cpu")

    with pytest.raises(RuntimeError, match="Failed to load real"):
        summarizer.load()

    assert not summarizer.dry_run
    assert not summarizer.is_loaded
