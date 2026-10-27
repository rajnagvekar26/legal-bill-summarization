"""
Cross-Architecture Legal Bill Summarization Dashboard.
Streamlit application for side-by-side inference, Pareto frontier visualization, and legal evaluation.
"""

import json
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.dashboard.results import find_model_result
from src.metrics.entity_extractor import LegalEntityExtractor
from src.metrics.evaluator import LegalEvaluator
from src.models.registry import (
    get_all_model_metadata,
    get_pareto_benchmarks,
)

# Page configuration
st.set_page_config(
    page_title="Legal Bill Summarization | Cross-Architecture Arena",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished legal tech aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .entity-tag-section {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 4px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .entity-tag-usc {
        background-color: #f3e8ff;
        color: #6b21a8;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 4px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .entity-tag-money {
        background-color: #dcfce7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 4px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .entity-tag-pct {
        background-color: #fef3c7;
        color: #92400e;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 4px;
        display: inline-block;
        margin-bottom: 4px;
    }
    .arch-badge {
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-bart { background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
    .badge-led { background-color: #ede9fe; color: #5b21b6; border: 1px solid #ddd6fe; }
    .badge-qwen { background-color: #fef08a; color: #854d0e; border: 1px solid #fde047; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_BILLS = {
    "Clean Water & Infrastructure Modernization Act": (
        "SECTION 1. SHORT TITLE.\n"
        "This Act may be cited as the 'Clean Water Infrastructure Modernization Act of 2026'.\n\n"
        "SEC. 2. FINDINGS AND PURPOSE.\n"
        "Congress finds that federal investments in municipal drinking water facilities under 33 U.S.C. 1251 "
        "have fallen below target modernization thresholds. Pursuant to Section 104(a)(1) of the Federal Water "
        "Pollution Control Act, the Administrator of the Environmental Protection Agency shall establish a grant program.\n\n"
        "SEC. 3. AUTHORIZATION OF APPROPRIATIONS.\n"
        "There is authorized to be appropriated to carry out this section $450,000,000 for fiscal year 2027, "
        "$550,000,000 for fiscal year 2028, and $650,000,000 for fiscal year 2029. Not less than 15% of such funds "
        "shall be dedicated to rural and economically distressed communities. An administrative set-aside of 2.5% "
        "is allocated to federal oversight pursuant to 42 U.S.C. § 1983 and 5 U.S.C. 552a."
    ),
    "Veterans Health Protection & Statutory Expansion": (
        "SECTION 1. SHORT TITLE.\n"
        "This Act may be cited as the 'Veterans Health Protection Act'.\n\n"
        "SEC. 101. EXPANSION OF COMPREHENSIVE SERVICES.\n"
        "Section 1710 of title 38, United States Code, is amended by inserting comprehensive mental health and telehealth coverage. "
        "The Secretary shall ensure that veterans residing in rural locations receive specialized clinical treatment.\n\n"
        "SEC. 102. FUNDING AND AUDIT REQUIREMENTS.\n"
        "To carry out amendments made by this title, there is hereby authorized to be appropriated $1,250,000,000 for each of fiscal "
        "years 2027 through 2030. Any non-profit healthcare partner receiving federal sub-awards under Title 28, United States Code, Section 1331 "
        "must maintain a minimum 85% expenditure ratio on direct patient services, with administrative penalties capping overhead at 4.5%."
    ),
    "Small Business Innovation & Fair Credit Standard": (
        "SECTION 1. SHORT TITLE.\n"
        "This Act may be cited as the 'Small Business Innovation and Fair Credit Access Act'.\n\n"
        "SEC. 2. ENHANCEMENT OF LENDING MECHANISMS.\n"
        "Section 7(a) of the Small Business Act (15 U.S.C. 636(a)) is amended to allow expedited micro-loan approvals for eligible "
        "technology startups. Pursuant to 18 U.S.C. 1001, any fraudulent statement submitted under this section shall result in federal disqualification.\n\n"
        "SEC. 3. STATUTORY GUARANTEES AND ALLOCATIONS.\n"
        "The Administrator shall guarantee loans up to an aggregate volume of $2,500,000,000 across all participating private lenders. "
        "The guarantee fee is hereby reduced from 3.75% to 1.5% for minority-owned enterprises, provided that lenders allocate at least 25% "
        "of their portfolio to underserved census tracts."
    ),
}


# Sidebar navigation
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/feathericons/feather/master/icons/book-open.svg",
        width=48,
    )
    st.title("⚖️ Legal Summarizer")
    st.caption("Track B Pareto Benchmark & Inference Engine")

    nav_selection = st.radio(
        "Navigation",
        [
            "🏛️ Model Arena (Side-by-Side)",
            "📊 Pareto Frontier Benchmarks",
            "🔬 Live Evaluation & Entity Lab",
            "📐 Architecture & Guardrails",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Inference Controls")
    dry_run_mode = st.toggle("Dry-Run / Fast Sim Mode", value=False, help="Toggle real weight inference vs fast benchmark simulation.")
    api_base_url = st.text_input(
        "Inference API URL",
        value=os.getenv("SUMMARIZATION_API_URL", "http://127.0.0.1:8000"),
        help="Local FastAPI URL or the current Colab tunnel URL. Do not include an endpoint path.",
    ).strip().rstrip("/")
    max_tokens = st.slider("Max New Tokens", min_value=64, max_value=512, value=256, step=32)
    min_tokens = st.slider("Min New Tokens", min_value=16, max_value=128, value=32, step=16)
    beam_count = st.slider("Beam Count", min_value=1, max_value=4, value=2)

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #64748b;'>Antigravity Cross-Architecture Legal Engine<br>Track B Kaggle Artifacts</div>",
        unsafe_allow_html=True,
    )

evaluator = LegalEvaluator()
entity_extractor = LegalEntityExtractor()

# TAB 1: MODEL ARENA
if nav_selection == "🏛️ Model Arena (Side-by-Side)":
    st.markdown("<div class='main-header'>Cross-Architecture Model Arena</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Compare Dense Seq2Seq (BART), Sparse Attention (LED), and Causal QLoRA (Qwen2.5) on legal statutes.</div>",
        unsafe_allow_html=True,
    )

    col_sample, col_models = st.columns([2, 1])
    with col_sample:
        selected_sample = st.selectbox(
            "Select Preloaded Congressional Bill:",
            ["Custom Input"] + list(SAMPLE_BILLS.keys()),
        )
    with col_models:
        selected_models = st.multiselect(
            "Select Models to Run:",
            options=["bart_base_dense", "led_base_sparse_4k", "qwen25_1_5b_causal_qlora"],
            default=["bart_base_dense", "led_base_sparse_4k", "qwen25_1_5b_causal_qlora"],
            format_func=lambda x: {
                "bart_base_dense": "BART Base (Dense Seq2Seq)",
                "led_base_sparse_4k": "LED Base (Sparse 4K)",
                "qwen25_1_5b_causal_qlora": "Qwen2.5-1.5B (Causal QLoRA)",
            }.get(x, x),
        )

    initial_text = SAMPLE_BILLS[selected_sample] if selected_sample != "Custom Input" else ""
    bill_input = st.text_area(
        "Legal Bill Text to Summarize:",
        value=initial_text,
        height=220,
        placeholder="Enter legislative bill text containing statutory sections, U.S.C. citations, and monetary appropriations...",
    )

    # Entity preview on input
    if bill_input.strip():
        detected_in_bill = entity_extractor.extract(bill_input)
        ent_col1, ent_col2, ent_col3, ent_col4 = st.columns(4)
        with ent_col1:
            st.caption(f"Sections Detected: **{len(detected_in_bill.sections)}**")
        with ent_col2:
            st.caption(f"U.S.C. Citations: **{len(detected_in_bill.usc_citations)}**")
        with ent_col3:
            st.caption(f"Monetary Figures: **{len(detected_in_bill.monetary_amounts)}**")
        with ent_col4:
            st.caption(f"Percentages: **{len(detected_in_bill.percentages)}**")

    if st.button("🚀 Run Comparative Summarization", type="primary", use_container_width=True):
        if not bill_input.strip():
            st.error("Please provide legal bill text to summarize.")
        elif not selected_models:
            st.warning("Please select at least one model.")
        elif not api_base_url:
            st.error("Enter the local or Colab inference API URL.")
        else:
            with st.spinner(f"Running comparative inference via {api_base_url}..."):
                payload = {
                    "text": bill_input,
                    "max_new_tokens": max_tokens,
                    "min_new_tokens": min_tokens,
                    "num_beams": beam_count,
                    "models": selected_models,
                    "dry_run": dry_run_mode,
                }
                headers = {
                    "Content-Type": "application/json",
                    "bypass-tunnel-reminder": "true",
                }
                try:
                    response = requests.post(
                        f"{api_base_url}/api/summarize",
                        json=payload,
                        headers=headers,
                        timeout=600,
                    )
                    response.raise_for_status()
                    res_json = response.json()
                    st.session_state["comparative_results"] = res_json["results"]
                    if dry_run_mode:
                        st.warning("Dry-run is enabled: these are placeholders, not model-generated summaries.")
                    else:
                        st.success("Comparative inference complete.")
                except requests.Timeout:
                    st.error("The inference request timed out. Check that Colab is still running and try again.")
                except requests.HTTPError as e:
                    detail = e.response.text if e.response is not None else str(e)
                    st.error(f"Inference API returned an error: {detail}")
                except requests.RequestException as e:
                    st.error(f"Could not reach the inference API at {api_base_url}: {e}")

    # Render unified comparative results from frontend state
    if "comparative_results" in st.session_state and st.session_state["comparative_results"]:
        results_data = st.session_state["comparative_results"]
        cols = st.columns(len(selected_models))

        for idx, model_key in enumerate(selected_models):
            with cols[idx]:
                badge_class = {
                    "bart_base_dense": "badge-bart",
                    "led_base_sparse_4k": "badge-led",
                    "qwen25_1_5b_causal_qlora": "badge-qwen",
                }.get(model_key, "badge-bart")

                arch_name = {
                    "bart_base_dense": "BART (Dense)",
                    "led_base_sparse_4k": "LED (Sparse 4K)",
                    "qwen25_1_5b_causal_qlora": "Qwen2.5 (QLoRA)",
                }.get(model_key, model_key)

                model_label = {
                    "bart_base_dense": "bart-base",
                    "led_base_sparse_4k": "led-base-16384",
                    "qwen25_1_5b_causal_qlora": "Qwen2.5-1.5B-Instruct",
                }.get(model_key, model_key)

                st.markdown(
                    f"<div class='arch-badge {badge_class}'>{arch_name}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {model_label}")

                model_res = find_model_result(results_data, model_key)
                if model_res is not None:
                    if isinstance(model_res, str):
                        summary_text = model_res
                        latency_ms = 0.0
                        ms_per_tok = 0.0
                    else:
                        summary_text = model_res.get("summary_text") or model_res.get("summary") or model_res.get("text") or "No summary returned."
                        latency_ms = float(model_res.get("latency_ms") or model_res.get("latency") or 0.0)
                        token_count = len(summary_text.split())
                        ms_per_tok = float(model_res.get("ms_per_token") or (round(latency_ms / max(1, token_count), 1) if latency_ms else 0.0))

                    # Summary box
                    st.info(summary_text)

                    # Metric indicators
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Latency", f"{latency_ms:.1f} ms" if latency_ms > 0 else "Colab GPU")
                    with m2:
                        st.metric("Speed", f"{ms_per_tok:.1f} ms/tok" if ms_per_tok > 0 else "Fast")

                    # Extracted legal entities
                    extracted = entity_extractor.extract(summary_text)
                    st.caption(f"Entities Extracted: {extracted.total_count}")

                    with st.expander("Show Extracted Legal Entities"):
                        for sec in extracted.sections:
                            st.markdown(f"<span class='entity-tag-section'>{sec}</span>", unsafe_allow_html=True)
                        for usc in extracted.usc_citations:
                            st.markdown(f"<span class='entity-tag-usc'>{usc}</span>", unsafe_allow_html=True)
                        for mny in extracted.monetary_amounts:
                            st.markdown(f"<span class='entity-tag-money'>{mny}</span>", unsafe_allow_html=True)
                        for pct in extracted.percentages:
                            st.markdown(f"<span class='entity-tag-pct'>{pct}</span>", unsafe_allow_html=True)
                else:
                    received_keys = (
                        [item.get("model_key") for item in results_data if isinstance(item, dict)]
                        if isinstance(results_data, list)
                        else list(results_data.keys())
                        if isinstance(results_data, dict)
                        else [type(results_data).__name__]
                    )
                    st.warning(
                        f"No summary data received for {model_key}. "
                        f"Models returned by API: {received_keys}"
                    )

# TAB 2: PARETO FRONTIER BENCHMARKS
elif nav_selection == "📊 Pareto Frontier Benchmarks":
    st.markdown("<div class='main-header'>Track B Pareto Frontier & Trade-offs</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Empirical evaluation results across Quality, Latency, Memory, and Carbon Emissions from Kaggle Track B.</div>",
        unsafe_allow_html=True,
    )

    benchmarks = get_pareto_benchmarks()
    if benchmarks:
        df = pd.DataFrame(benchmarks)

        # Overview KPI metric row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            fastest_model = df.sort_values("inference_latency_ms_per_sample").iloc[0]
            st.metric("⚡ Fastest Model", f"{fastest_model['model_key']}", f"{fastest_model['inference_latency_ms_per_sample']} ms")
        with kpi2:
            best_rouge = df.sort_values("rougeL", ascending=False).iloc[0]
            st.metric("🏆 Highest ROUGE-L", f"{best_rouge['model_key']}", f"{best_rouge['rougeL']:.2f}")
        with kpi3:
            best_bert = df.sort_values("bertscore_f1", ascending=False).iloc[0]
            st.metric("🎯 Best BERTScore F1", f"{best_bert['model_key']}", f"{best_bert['bertscore_f1']:.2f}")
        with kpi4:
            most_eco = df.sort_values("emissions_kg_co2").iloc[0]
            st.metric("🌱 Lowest CO₂", f"{most_eco['model_key']}", f"{most_eco['emissions_kg_co2']:.4f} kg")

        st.markdown("---")

        # Interactive Plotly Visualizations
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("1. Quality vs. Latency Pareto Curve")
            fig1 = px.scatter(
                df,
                x="inference_latency_ms_per_sample",
                y="rougeL",
                color="arch_type",
                text="model_key",
                size="total_params",
                hover_data=["rouge1", "rouge2", "bertscore_f1", "trainable_pct"],
                title="ROUGE-L vs Inference Latency (ms/sample)",
                labels={
                    "inference_latency_ms_per_sample": "Inference Latency (ms/sample)",
                    "rougeL": "ROUGE-L Score",
                },
            )
            fig1.update_traces(textposition="top center")
            fig1.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            st.subheader("2. Efficiency: ROUGE-L vs Peak VRAM")
            fig2 = px.scatter(
                df,
                x="peak_train_vram_gb",
                y="rougeL",
                color="model_key",
                size="rougeL_per_vram_gb",
                hover_data=["rougeL_per_vram_gb", "trainable_params"],
                title="ROUGE-L vs Peak VRAM (GB)",
                labels={"peak_train_vram_gb": "Peak VRAM (GB)", "rougeL": "ROUGE-L Score"},
            )
            fig2.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig2, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            st.subheader("3. Eco-Efficiency: BERTScore per Watt-Hour")
            fig3 = px.bar(
                df,
                x="model_key",
                y="bertscore_per_watt_hour",
                color="model_key",
                title="BERTScore F1 per Watt-Hour (Higher is Greener)",
                labels={"bertscore_per_watt_hour": "BERTScore / Wh", "model_key": "Model"},
            )
            fig3.update_layout(template="plotly_white", height=380, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with chart_col4:
            st.subheader("4. Energy & Carbon Footprint")
            fig4 = px.bar(
                df,
                x="model_key",
                y=["energy_kwh", "emissions_kg_co2"],
                barmode="group",
                title="Training Energy (kWh) and CO₂ Emissions (kg)",
                labels={"value": "Magnitude", "model_key": "Model", "variable": "Metric"},
            )
            fig4.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("📋 Comprehensive Benchmark Table")
        st.dataframe(df, use_container_width=True)

        # Artifact figure display if present
        img_path = Path("artifacts/pareto_frontier_track_b.png")
        if img_path.exists():
            with st.expander("View Original Kaggle Track B Pareto Frontier Plot"):
                st.image(str(img_path), caption="Kaggle Track B Pareto Frontier Diagram", use_container_width=True)
    else:
        st.warning("Benchmark data not found in artifacts/track_b_pareto_summary.csv")

# TAB 3: LIVE EVALUATION & ENTITY LAB
elif nav_selection == "🔬 Live Evaluation & Entity Lab":
    st.markdown("<div class='main-header'>Multi-Axis Legal Evaluation Lab</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Test generated summaries against gold-standard references using ROUGE, BERTScore, and Legal Entity Recall.</div>",
        unsafe_allow_html=True,
    )

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        cand_input = st.text_area(
            "Candidate Summary (Model Output):",
            value=(
                "The Clean Water Infrastructure Modernization Act of 2026 amends Section 104(a)(1) "
                "to provide $450,000,000 in FY 2027 and $550,000,000 in FY 2028 under 33 U.S.C. 1251. "
                "At least 15% of grants are dedicated to rural areas with a 2.5% oversight allocation under 42 U.S.C. § 1983."
            ),
            height=160,
        )
    with eval_col2:
        ref_input = st.text_area(
            "Reference Summary (Gold Standard):",
            value=(
                "Clean Water Infrastructure Modernization Act of 2026 - Directs the EPA under 33 U.S.C. 1251 "
                "and Section 104(a)(1) to award municipal water facility grants. Authorizes $450,000,000 for FY2027, "
                "$550,000,000 for FY2028, and $650,000,000 for FY2029. Reserves 15% for rural communities and 2.5% "
                "for administrative oversight under 42 U.S.C. § 1983 and 5 U.S.C. 552a."
            ),
            height=160,
        )

    if st.button("⚖️ Run Multi-Axis Evaluation", type="primary", use_container_width=True):
        if not cand_input.strip() or not ref_input.strip():
            st.error("Please provide both candidate and reference summaries.")
        else:
            with st.spinner("Computing ROUGE, BERTScore, and Legal Entity Recall..."):
                eval_res = evaluator.evaluate(candidate=cand_input, reference=ref_input)

                # Metric Cards
                m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
                with m_c1:
                    st.metric("ROUGE-1", f"{eval_res.rouge1:.2f}%")
                with m_c2:
                    st.metric("ROUGE-2", f"{eval_res.rouge2:.2f}%")
                with m_c3:
                    st.metric("ROUGE-L", f"{eval_res.rougeL:.2f}%")
                with m_c4:
                    st.metric("BERTScore F1", f"{eval_res.bertscore_f1:.2f}%")
                with m_c5:
                    st.metric("Entity Recall", f"{eval_res.legal_entity_recall_pct:.2f}%")

                st.markdown("---")

                # Entity Overlap Breakdown
                st.subheader("🏛️ Legal Entity Faithfulness Analysis")
                ent_c1, ent_c2 = st.columns(2)

                with ent_c1:
                    st.markdown("#### ✅ Matched Entities")
                    if eval_res.matched_entities:
                        for item in eval_res.matched_entities:
                            st.markdown(f"- `{item}`")
                    else:
                        st.write("No overlapping legal entities identified.")

                with ent_c2:
                    st.markdown("#### ❌ Missing in Candidate")
                    if eval_res.missing_entities:
                        for item in eval_res.missing_entities:
                            st.markdown(f"- `{item}`")
                    else:
                        st.write("All reference entities captured! (100% Entity Recall)")

# TAB 4: ARCHITECTURE & GUARDRAILS
elif nav_selection == "📐 Architecture & Guardrails":
    st.markdown("<div class='main-header'>Architectural Guardrails & Verification</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Technical documentation and strict guardrail implementations for LED and Qwen2.5.</div>",
        unsafe_allow_html=True,
    )

    models_info = get_all_model_metadata()
    for m in models_info:
        with st.expander(f"📦 Model: {m['model_key']} ({m['hf_model_id']})", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Architecture Type:** `{m['arch_type']}`")
                st.markdown(f"**Max Source Context:** `{m['max_source_length']}` tokens")
                st.markdown(f"**Adapter Type:** `{m['adapter_type']}`")
                st.markdown(f"**Adapter Path:** `{m['adapter_path']}`")
            with col_b:
                st.markdown(f"**Total Parameters:** `{m['total_params']:,}`")
                st.markdown(f"**Trainable Parameters:** `{m['trainable_params']:,}`")
                st.markdown(f"**Trainable Ratio:** `{m['trainable_pct']:.4f}%`")
                st.markdown(f"**Fingerprint:** `{m['config_fingerprint']}`")
            st.info(m["description"])

    st.markdown("---")
    st.subheader("🛡️ Implementation of Critical Guardrails")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("### 1. LED Global Attention Mask Guardrail")
        st.markdown(
            """
            LED Longformer uses sparse sliding-window attention to scale linearly $O(N)$ with sequence length. 
            However, the summary decoder must have global visibility into document-level representations.
            - **Constraint:** LED strictly requires a `global_attention_mask` tensor matching `input_ids.shape`.
            - **Implementation:** Index 0 (the BOS `<s>` token) is explicitly set to `1`, while remaining positions are `0`.
            """
        )
        st.code(
            """# Critical LED Guardrail in src/models/led_summarizer.py
global_attention_mask = torch.zeros_like(inputs["input_ids"])
global_attention_mask[:, 0] = 1
inputs["global_attention_mask"] = global_attention_mask
output_ids = self.model.generate(**inputs, ...)""",
            language="python",
        )

    with col_g2:
        st.markdown("### 2. Qwen Left-Padding & Prompt Slicing Guardrail")
        st.markdown(
            """
            Qwen2.5 is an autoregressive causal decoder. 
            Without proper controls, causal decoders echo the prompt text or produce misaligned attention with right-padding.
            - **Constraint 1:** Tokenizer padding side MUST be left: `tokenizer.padding_side = 'left'`.
            - **Constraint 2:** `output_ids` contains both prompt tokens and generated tokens. We explicitly slice `output_ids[:, input_length:]`.
            """
        )
        st.code(
            """# Critical Qwen Guardrails in src/models/qwen_summarizer.py
self.tokenizer.padding_side = "left"
inputs = self.tokenizer(formatted_prompt, ...)
prompt_len = inputs["input_ids"].shape[-1]

output_ids = self.model.generate(**inputs, ...)
# Slicing prompt tokens before decoding
sliced_tokens = output_ids[:, prompt_len:]
summary_text = self.tokenizer.decode(sliced_tokens[0], skip_special_tokens=True)""",
            language="python",
        )
