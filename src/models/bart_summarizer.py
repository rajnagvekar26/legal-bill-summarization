"""
BART Dense Seq2Seq Summarizer.

Model: facebook/bart-base + LoRA Adapter
Architecture: Dense Encoder-Decoder
Context Window: 1,024 tokens
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.models.base import BaseSummarizer, ModelMetadata, SummarizationResult

logger = logging.getLogger(__name__)


class BartSummarizer(BaseSummarizer):
    """Dense encoder-decoder legal summarizer using fine-tuned BART."""

    def __init__(
        self,
        adapter_dir: Optional[Union[str, Path]] = "artifacts/models/bart_base_dense",
        hf_model_id: str = "facebook/bart-base",
        max_source_length: int = 1024,
        device: str = "auto",
        torch_dtype: str = "auto",
        dry_run: bool = False,
    ):
        super().__init__(
            model_key="bart_base_dense",
            hf_model_id=hf_model_id,
            arch_type="seq2seq_dense",
            adapter_dir=adapter_dir,
            max_source_length=max_source_length,
            device=device,
            torch_dtype=torch_dtype,
            dry_run=dry_run,
        )

    def load(self) -> "BartSummarizer":
        """Loads BART base model, attaches fine-tuned LoRA weights, and prepares tokenizer."""
        if self.dry_run:
            self.is_loaded = True
            logger.info("BartSummarizer initialized in dry-run mode.")
            return self

        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            adapter_path = self.find_adapter_dir()
            logger.info("Loading BART tokenizer from %s...", adapter_path or self.hf_model_id)

            # Load tokenizer
            if adapter_path and (adapter_path / "tokenizer_config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, local_files_only=True)

            # Load base model
            logger.info("Loading BART base model %s...", self.hf_model_id)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.hf_model_id,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                local_files_only=True,
            )

            # Attach LoRA adapter
            if adapter_path and (adapter_path / "adapter_config.json").exists():
                logger.info("Attaching LoRA adapter from %s...", adapter_path)
                self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            else:
                logger.warning("Adapter config not found at %s. Using base model.", adapter_path)
                self.model = base_model

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("BartSummarizer successfully loaded on %s.", self.device)
        except Exception as e:
            logger.exception("Failed to load real BART weights.")
            raise RuntimeError("Failed to load real BART weights.") from e

        return self

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 32,
        num_beams: int = 4,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> SummarizationResult:
        """Runs legal bill summarization through dense encoder-decoder."""
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
        )

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_key=self.model_key,
            hf_model_id=self.hf_model_id,
            arch_type=self.arch_type,
            max_source_length=self.max_source_length,
            adapter_path=str(self.find_adapter_dir()) if self.find_adapter_dir() else None,
            adapter_type="LoRA (dense)",
            total_params=141189888,
            trainable_params=1769472,
            trainable_pct=1.2533,
            config_fingerprint="fc6c9cc1e722",
            description="Dense seq2seq architecture. Ultra-fast inference latency (5.2 ms/token) with high ROUGE/VRAM efficiency.",
        )
