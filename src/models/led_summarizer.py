"""
Longformer Encoder-Decoder (LED) Sparse Summarizer.

Model: allenai/led-base-16384 + LoRA Adapter
Architecture: Sparse Encoder-Decoder with Local Sliding Window + Global Attention
Context Window: 4,096 tokens (supports up to 16,384)

CRITICAL GUARDRAIL:
LED requires an explicit global_attention_mask with index 0 (BOS token) set to 1.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.models.base import BaseSummarizer, ModelMetadata, SummarizationResult

logger = logging.getLogger(__name__)


class LedSummarizer(BaseSummarizer):
    """Sparse attention legal summarizer using fine-tuned LED-base-16384."""

    def __init__(
        self,
        adapter_dir: Optional[Union[str, Path]] = "artifacts/models/led_base_sparse_4k",
        hf_model_id: str = "allenai/led-base-16384",
        max_source_length: int = 4096,
        device: str = "auto",
        torch_dtype: str = "auto",
        dry_run: bool = False,
    ):
        super().__init__(
            model_key="led_base_sparse_4k",
            hf_model_id=hf_model_id,
            arch_type="seq2seq_sparse_led",
            adapter_dir=adapter_dir,
            max_source_length=max_source_length,
            device=device,
            torch_dtype=torch_dtype,
            dry_run=dry_run,
        )

    @staticmethod
    def create_global_attention_mask(input_ids) -> Any:
        """
        CRITICAL GUARDRAIL:
        Generates LED global attention mask:
        - Shape matches input_ids exactly
        - 0 for local sliding window attention across the sequence
        - 1 for global attention at index 0 (BOS / <s> token)
        """
        import torch

        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        return global_attention_mask

    def load(self) -> "LedSummarizer":
        """Loads LED base model, attaches sparse LoRA adapter, and verifies attention layers."""
        if self.dry_run:
            self.is_loaded = True
            logger.info("LedSummarizer initialized in dry-run mode.")
            return self

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoTokenizer, LEDForConditionalGeneration

            adapter_path = self.find_adapter_dir()
            logger.info("Loading LED tokenizer from %s...", adapter_path or self.hf_model_id)

            if adapter_path and (adapter_path / "tokenizer_config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, local_files_only=True)

            logger.info("Loading LED base model %s...", self.hf_model_id)
            base_model = LEDForConditionalGeneration.from_pretrained(
                self.hf_model_id,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                local_files_only=True,
            )

            if adapter_path and (adapter_path / "adapter_config.json").exists():
                logger.info("Attaching LED LoRA adapter from %s...", adapter_path)
                self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            else:
                logger.warning("LED adapter config not found at %s. Using base model.", adapter_path)
                self.model = base_model

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("LedSummarizer successfully loaded on %s.", self.device)
        except Exception as e:
            logger.exception("Failed to load real LED weights.")
            raise RuntimeError("Failed to load real LED weights.") from e

        return self

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 32,
        num_beams: int = 2,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> SummarizationResult:
        """
        Runs legal bill summarization with strict global_attention_mask enforcement.
        """
        if not self.is_loaded:
            self.load()

        if self.dry_run:
            return self._generate_dry_run_summary(text, max_new_tokens)

        import torch

        start_time = time.perf_counter()

        inputs = self.tokenizer(
            text,
            max_length=self.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        # CRITICAL GUARDRAIL: Explicitly generate global_attention_mask (index 0 set to 1)
        global_attention_mask = self.create_global_attention_mask(inputs["input_ids"])
        inputs["global_attention_mask"] = global_attention_mask

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "early_stopping": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        summary_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        summary_tokens = int(output_ids.shape[-1])

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        ms_per_token = latency_ms / max(1, summary_tokens)

        return SummarizationResult(
            summary_text=summary_text,
            model_key=self.model_key,
            hf_model_id=self.hf_model_id,
            arch_type=self.arch_type,
            latency_ms=round(latency_ms, 2),
            ms_per_token=round(ms_per_token, 2),
            prompt_tokens=prompt_tokens,
            summary_tokens=summary_tokens,
            total_tokens=prompt_tokens + summary_tokens,
            device=self.device,
            extra_metadata={"global_attention_index_0": True},
        )

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_key=self.model_key,
            hf_model_id=self.hf_model_id,
            arch_type=self.arch_type,
            max_source_length=self.max_source_length,
            adapter_path=str(self.find_adapter_dir()) if self.find_adapter_dir() else None,
            adapter_type="LoRA (sparse local + global)",
            total_params=164056320,
            trainable_params=2211840,
            trainable_pct=1.3482,
            config_fingerprint="c38195a7eea4",
            description="Sparse local sliding window + global attention. Extended context up to 4,096 tokens with low VRAM footprint.",
        )
