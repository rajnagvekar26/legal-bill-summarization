"""Helpers for normalizing model results returned by inference API variants."""


def find_model_result(data, model_key):
    """Find one model result in either the API list or legacy keyed response."""
    aliases = {
        "bart_base_dense": ("bart_base_dense", "facebook/bart-base", "bart"),
        "led_base_sparse_4k": ("led_base_sparse_4k", "allenai/led-base-16384", "led"),
        "qwen25_1_5b_causal_qlora": (
            "qwen25_1_5b_causal_qlora",
            "qwen/qwen2.5-1.5b-instruct",
            "qwen2.5",
            "qwen",
        ),
    }
    model_aliases = aliases.get(model_key, (model_key,))

    def matches(value):
        normalized = str(value or "").strip().lower()
        return any(normalized == alias for alias in model_aliases)

    if isinstance(data, dict):
        if model_key in data:
            return data[model_key]
        return next(
            (result for key, result in data.items() if matches(key)),
            None,
        )
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and any(
                matches(item.get(field))
                for field in ("model_key", "model", "name", "hf_model_id")
            ):
                return item
    return None
