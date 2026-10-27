from src.dashboard.results import find_model_result


API_RESULTS = [
    {
        "summary_text": "BART summary",
        "model_key": "bart_base_dense",
        "hf_model_id": "facebook/bart-base",
    },
    {
        "summary_text": "LED summary",
        "model_key": "led_base_sparse_4k",
        "hf_model_id": "allenai/led-base-16384",
    },
    {
        "summary_text": "Qwen summary",
        "model_key": "qwen25_1_5b_causal_qlora",
        "hf_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    },
]


def test_find_model_result_matches_all_api_model_keys():
    for result in API_RESULTS:
        assert find_model_result(API_RESULTS, result["model_key"]) == result


def test_find_model_result_matches_model_id_aliases():
    assert (
        find_model_result(API_RESULTS, "led_base_sparse_4k")
        == API_RESULTS[1]
    )
    assert (
        find_model_result(API_RESULTS, "qwen25_1_5b_causal_qlora")
        == API_RESULTS[2]
    )


def test_find_model_result_matches_legacy_keyed_results():
    results = {
        "bart_base_dense": {"summary": "BART summary"},
        "led_base_sparse_4k": {"summary": "LED summary"},
        "qwen25_1_5b_causal_qlora": {"summary": "Qwen summary"},
    }

    assert find_model_result(results, "led_base_sparse_4k") == results[
        "led_base_sparse_4k"
    ]
    assert find_model_result(results, "qwen25_1_5b_causal_qlora") == results[
        "qwen25_1_5b_causal_qlora"
    ]
