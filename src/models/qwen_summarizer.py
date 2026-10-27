"""
Qwen2.5-1.5B Causal Decoder Summarizer.

Model: Qwen/Qwen2.5-1.5B-Instruct + QLoRA Adapter
Architecture: Autoregressive Decoder-Only
Context Window: 2,048 tokens

CRITICAL GUARDRAILS:
1. Enforce left-padding on tokenizer: tokenizer.padding_side = "left"
2. Explicitly slice out the input prompt tokens before decoding so the model does not echo the bill text.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.models.base import BaseSummarizer, ModelMetadata, SummarizationResult

logger = logging.getLogger(__name__)


class QwenSummarizer(BaseSummarizer):
    """Autoregressive causal decoder legal summarizer using fine-tuned Qwen2.5-1.5B."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a professional legislative analyst. Provide a clear, faithful summary of the "
        "following legal bill, highlighting key statutory provisions, sections, citations, "
        "appropriations, and percentages."
    )

    def __init__(
        self,
        adapter_dir: Optional[Union[str, Path]] = "artifacts/models/qwen25_1_5b_causal_qlora",
        hf_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_source_length: int = 2048,
        device: str = "auto",
        torch_dtype: str = "auto",
        dry_run: bool = False,
    ):
        super().__init__(
            model_key="qwen25_1_5b_causal_qlora",
            hf_model_id=hf_model_id,
            arch_type="causal_decoder",
            adapter_dir=adapter_dir,
            max_source_length=max_source_length,
            device=device,
            torch_dtype=torch_dtype,
            dry_run=dry_run,
        )

    @staticmethod
    def slice_prompt_tokens(output_ids, prompt_len: int) -> Any:
        """
        CRITICAL GUARDRAIL:
        Autoregressive causal decoders return [prompt_tokens + generated_tokens].
        Explicitly slice out the input prompt tokens before decoding so the model does not echo the bill text.
        """
        return output_ids[:, prompt_len:]

    def format_prompt(self, bill_text: str) -> str:
        """Formats the input text with standard Qwen ChatML instruction template."""
        return (
            f"<|im_start|>system\n{self.DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{bill_text.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def load(self) -> "QwenSummarizer":
        """Loads Qwen tokenizer with left-padding and attaches fine-tuned LoRA adapter."""
        if self.dry_run:
            self.is_loaded = True
            logger.info("QwenSummarizer initialized in dry-run mode.")
            return self

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            adapter_path = self.find_adapter_dir()
            logger.info("Loading Qwen tokenizer from %s...", adapter_path or self.hf_model_id)

            if adapter_path and (adapter_path / "tokenizer_config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, local_files_only=True)

            # CRITICAL GUARDRAIL: Enforce left-padding for causal decoder generation
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading Qwen base model %s...", self.hf_model_id)
            model_kwargs = {
                "torch_dtype": torch.float32 if self.device == "cpu" else torch.float16,
                "local_files_only": True,
            }

            # Attempt 4-bit quantization if bitsandbytes and CUDA are available
            if self.device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = bnb_config
                except Exception as bnb_err:
                    logger.warning("BitsAndBytes not available; using standard precision: %s", bnb_err)

            base_model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                **model_kwargs,
            )

            if adapter_path and (adapter_path / "adapter_config.json").exists():
                logger.info("Attaching Qwen LoRA adapter from %s...", adapter_path)
                self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            else:
                logger.warning("Qwen adapter config not found at %s. Using base model.", adapter_path)
                self.model = base_model

            if self.device != "cuda" or "quantization_config" not in model_kwargs:
                self.model.to(self.device)

            self.model.eval()
            self.is_loaded = True
            logger.info("QwenSummarizer successfully loaded on %s.", self.device)
        except Exception as e:
            logger.exception("Failed to load real Qwen weights.")
            raise RuntimeError("Failed to load real Qwen weights.") from e

        return self

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 256,
        min_new_tokens: int = 32,
        num_beams: int = 1,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs,
    ) -> SummarizationResult:
        """
        Runs legal bill summarization with prompt token slicing.
        """
        if not self.is_loaded:
            self.load()

        if self.dry_run:
            return self._generate_dry_run_summary(text, max_new_tokens)

        import torch

        start_time = time.perf_counter()

        formatted_prompt = self.format_prompt(text)

        # Tokenize with left-padding
        inputs = self.tokenizer(
            formatted_prompt,
            max_length=self.max_source_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["num_beams"] = num_beams

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # CRITICAL GUARDRAIL: Slicing out input prompt tokens before decoding
        sliced_tokens = self.slice_prompt_tokens(output_ids, prompt_len)
        summary_text = self.tokenizer.decode(sliced_tokens[0], skip_special_tokens=True).strip()
        summary_tokens = int(sliced_tokens.shape[-1])

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        ms_per_token = latency_ms / max(1, summary_tokens)

        return SummarizationResult(
            summary_text=summary_text,
            model_key=self.model_key,
            hf_model_id=self.hf_model_id,
            arch_type=self.arch_type,
            latency_ms=round(latency_ms, 2),
            ms_per_token=round(ms_per_token, 2),
            prompt_tokens=prompt_len,
            summary_tokens=summary_tokens,
            total_tokens=prompt_len + summary_tokens,
            device=self.device,
            extra_metadata={
                "left_padding_enforced": True,
                "prompt_tokens_sliced": True,
            },
        )

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_key=self.model_key,
            hf_model_id=self.hf_model_id,
            arch_type=self.arch_type,
            max_source_length=self.max_source_length,
            adapter_path=str(self.find_adapter_dir()) if self.find_adapter_dir() else None,
            adapter_type="QLoRA (NF4 4-bit base + LoRA)",
            total_params=907081216,
            trainable_params=18464768,
            trainable_pct=2.0356,
            config_fingerprint="f753edca969c",
            description="Autoregressive causal decoder. State-of-the-art generation quality (ROUGE-L 25.76, BERTScore 80.05).",
        )
