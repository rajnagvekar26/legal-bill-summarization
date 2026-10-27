"""
End-to-End Pipeline Test: California Restitution Bill Summarization & Multi-Axis Evaluation.
"""

import sys
import requests

API_BASE = "http://localhost:8000"

# 1. Define the input variables
bill_text = (
    "The people of the State of California do enact as follows: "
    "SECTION 1. Section 730.6 of the Welfare and Institutions Code is amended to read: "
    "730.6. (a) (1) It is the intent of the Legislature that a victim of conduct for which "
    "a minor is found to be a person described in Section 602 who incurs an economic loss "
    "as a result of the minor's conduct shall receive restitution directly from that minor."
)

reference_summary = (
    "Existing law provides that a minor who violates a criminal law may be adjudged to be a "
    "ward of the court. Existing law generally requires that the minor pay a restitution fine "
    "to be deposited into the Restitution Fund and restitution to any victim of his or her conduct. "
    "This bill would expand the definition of victim to include a corporation, estate, or other legal "
    "or commercial entity when that entity is a direct victim of a crime."
)

def run_test():
    print("=" * 70)
    print("  CROSS-ARCHITECTURE LEGAL BILL SUMMARIZATION: END-TO-END TEST")
    print("  Dataset Case: California Restitution Bill (Section 730.6)")
    print("=" * 70)

    # Check health first
    try:
        health_resp = requests.get(f"{API_BASE}/api/health", timeout=5)
        health_resp.raise_for_status()
        health_info = health_resp.json()
        print(f"\n[API Status] {health_info.get('status').upper()} | Device: {health_info.get('device')} | PyTorch: {health_info.get('torch_version')}")
    except Exception as e:
        print(f"\n[ERROR] Could not connect to API at {API_BASE}: {e}")
        sys.exit(1)

    # 2. Query the Summarization Endpoint
    print("\n--> Sending POST /api/summarize (model_key='all')...")
    summarize_payload = {
        "bill_text": bill_text,
        "model_key": "all",
        "max_new_tokens": 256,
        "min_new_tokens": 32,
    }

    try:
        sum_resp = requests.post(f"{API_BASE}/api/summarize", json=summarize_payload, timeout=60)
        sum_resp.raise_for_status()
        sum_data = sum_resp.json()
    except Exception as e:
        print(f"[ERROR] Summarize endpoint failed: {e}")
        if 'sum_resp' in locals():
            print(f"Response: {sum_resp.text}")
        sys.exit(1)

    results = sum_data.get("results", [])
    print(f"Received summaries from {len(results)} models. Fastest: {sum_data.get('fastest_model')} ({sum_data.get('min_latency_ms')} ms)\n")

    # 3. Query the Evaluation Endpoint & 4. Print and Format Output
    print("-" * 70)
    evaluated_results = []

    for item in results:
        model_name = item.get("model_key")
        summary_text = item.get("summary_text")
        latency_total = item.get("latency_ms")
        ms_per_token = item.get("ms_per_token")
        prompt_tokens = item.get("prompt_tokens")
        summary_tokens = item.get("summary_tokens")

        # Query /api/evaluate
        eval_payload = {
            "prediction": summary_text,
            "reference": reference_summary,
        }
        
        try:
            eval_resp = requests.post(f"{API_BASE}/api/evaluate", json=eval_payload, timeout=30)
            eval_resp.raise_for_status()
            eval_data = eval_resp.json()
        except Exception as e:
            print(f"[ERROR] Evaluation failed for {model_name}: {e}")
            eval_data = {}

        rouge_l = eval_data.get("rougeL", 0.0)
        rouge_1 = eval_data.get("rouge1", 0.0)
        rouge_2 = eval_data.get("rouge2", 0.0)
        bertscore_f1 = eval_data.get("bertscore_f1", 0.0)
        entity_recall = eval_data.get("legal_entity_recall_pct", 0.0)
        matched_entities = eval_data.get("matched_entities", [])
        missing_entities = eval_data.get("missing_entities", [])

        record = {
            "model_name": model_name,
            "summary_text": summary_text,
            "ms_per_token": ms_per_token,
            "latency_total": latency_total,
            "rouge_l": rouge_l,
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "bertscore_f1": bertscore_f1,
            "entity_recall": entity_recall,
            "matched_entities": matched_entities,
        }
        evaluated_results.append(record)

        # Print Formatted Output
        print(f"Model: {model_name}")
        print(f"Architecture: {item.get('arch_type')}")
        print(f"Generated Summary:\n  \"{summary_text}\"")
        print("Performance Metrics:")
        print(f"  * Latency:              {ms_per_token:.2f} ms/token (Total: {latency_total:.1f} ms)")
        print(f"  * Token Breakdown:      {prompt_tokens} prompt tokens -> {summary_tokens} generated tokens")
        print(f"  * ROUGE-1 / ROUGE-2:    {rouge_1:.2f}% / {rouge_2:.2f}%")
        print(f"  * ROUGE-L:              {rouge_l:.2f}%")
        print(f"  * BERTScore F1:         {bertscore_f1:.2f}%")
        print(f"  * Legal Entity Recall:  {entity_recall:.2f}%")
        print(f"  * Matched Entities:     {matched_entities}")
        print("-" * 70)

    # Comparative Summary Table
    print("\n" + "=" * 70)
    print("  CROSS-ARCHITECTURE COMPARISON MATRIX")
    print("=" * 70)
    print(f"{'Model Key':<26} | {'Speed (ms/tok)':<14} | {'ROUGE-L':<9} | {'BERTScore':<10} | {'Entity Recall':<13}")
    print("-" * 70)
    for r in evaluated_results:
        print(f"{r['model_name']:<26} | {r['ms_per_token']:<14.2f} | {r['rouge_l']:<9.2f} | {r['bertscore_f1']:<10.2f} | {r['entity_recall']:<13.2f}")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
