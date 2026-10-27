"""
Base Summarizer Interface and Data Structures.

Defines the contract for cross-architecture legal summarization models:
- Dense Seq2Seq (BART)
- Sparse Seq2Seq (LED)
- Autoregressive Causal Decoder (Qwen2.5)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """Standardized output container for model inference."""

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
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Architectural metadata for legal models."""

    model_key: str
    hf_model_id: str
    arch_type: str
    max_source_length: int
    adapter_path: Optional[str]
    adapter_type: str
    total_params: int
    trainable_params: int
    trainable_pct: float
    config_fingerprint: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseSummarizer(ABC):
    """Abstract base class for legal bill summarization models."""

    def __init__(
        self,
        model_key: str,
        hf_model_id: str,
        arch_type: str,
        adapter_dir: Optional[Union[str, Path]] = None,
        max_source_length: int = 1024,
        device: str = "auto",
        torch_dtype: str = "auto",
        dry_run: bool = False,
    ):
        self.model_key = model_key
        self.hf_model_id = hf_model_id
        self.arch_type = arch_type
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.max_source_length = max_source_length
        self.requested_device = device
        self.torch_dtype = torch_dtype
        self.dry_run = dry_run

        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def _resolve_device(self, device: str) -> str:
        """Resolves target hardware device."""
        if device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def find_adapter_dir(self) -> Optional[Path]:
        """Resolves path to adapter config & weights, checking best_adapter subdirectories."""
        if not self.adapter_dir:
            return None
        candidate = Path(self.adapter_dir)
        if (candidate / "adapter_config.json").exists():
            return candidate
        if (candidate / "best_adapter" / "adapter_config.json").exists():
            return candidate / "best_adapter"
        return candidate

    def _generate_dry_run_summary(
        self, text: str, max_new_tokens: int
    ) -> SummarizationResult:
        """Builds a lightweight placeholder result without loading model weights."""
        summary_text = text.strip()[: max(1, max_new_tokens * 4)]
        prompt_tokens = len(text.split())
        summary_tokens = len(summary_text.split())
        latency_ms = 0.01
        return SummarizationResult(
            summary_text=summary_text,
            model_key=self.model_key,
            hf_model_id=self.hf_model_id,
            arch_type=self.arch_type,
            latency_ms=latency_ms,
            ms_per_token=latency_ms / max(1, summary_tokens),
            prompt_tokens=prompt_tokens,
            summary_tokens=summary_tokens,
            total_tokens=prompt_tokens + summary_tokens,
            device=self.device,
            extra_metadata={"dry_run": True},
        )

    @abstractmethod
    def load(self) -> "BaseSummarizer":
        """Loads base model, tokenizer, and merges/attaches PEFT adapter."""
        pass

    @abstractmethod
    def summarize(
        self,
        bill_text: Optional[str] = None,
        text: Optional[str] = None,
        max_new_tokens: int = 256,
        min_new_tokens: int = 32,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> SummarizationResult:
        """Generates legal summary with latency profiling and token tracking."""
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Returns model metadata and parameters."""
        pass

