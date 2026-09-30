# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import io
import os
import re
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from .pipeline import (
    process_cases,
    read_any_table_from_upload,
)


def configured_demo_groups() -> list[dict]:
    specs = [
        (
            "paired_inspiratory_expiratory",
            "Inspiratory + Expiratory CT",
            "PGRAD_PAIRED_INSPIRATORY_EXPIRATORY_DEMO_PATH",
        ),
        (
            "single_inspiratory",
            "Inspiratory CT only",
            "PGRAD_SINGLE_INSPIRATORY_DEMO_PATH",
        ),
    ]
    groups = []
    for key, name, env_var in specs:
        path = os.getenv(env_var, "").strip()
        if path:
            groups.append({"key": key, "name": name, "path": Path(path).expanduser()})
    return groups
WORKFLOW_STEPS = [
    ("match_features", "Match uploaded columns"),
    ("load_assets", "Load model and evidence libraries"),
    ("build_evidence", "Build patient evidence"),
    ("setup_llm", "Configure API and LLM mode"),
    ("gate_reports", "Run gate logic and reports"),
    ("finalize", "Finalize result tables"),
]
LLM_PROVIDER_DEFAULTS = {
    "Qwen / DashScope": {
        "secret": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.6-plus",
    },
    "OpenAI": {
        "secret": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "Custom OpenAI-compatible": {
        "secret": "",
        "base_url": "",
        "model": "qwen3.6-plus",
    },
}


st.set_page_config(
    page_title="PGRAD",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


STYLE = """
<style>
    .main .block-container {
        padding: 1.1rem 1.4rem 2.2rem 1.4rem;
        max-width: 1480px;
        width: 100%;
    }
    div[data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding-left: 1.4rem;
        padding-right: 1.4rem;
    }
    [data-testid="stAppViewContainer"] { background: #f7f9fc; }
    [data-testid="stSidebar"] {
        background: #f3f6fa;
        border-right: 1px solid #d8dee8;
    }
    .hero {
        background: #ffffff;
        color: #102a43;
        border: 1px solid #d8dee8;
        border-left: 5px solid #123f6d;
        border-radius: 8px;
        padding: 22px 24px;
        text-align: left;
        box-shadow: none;
        margin-bottom: 18px;
    }
    .hero h1 { font-size: 29px; margin: 0 0 6px 0; font-weight: 750; }
    .hero p { margin: 0; font-size: 14px; color: #52616f; }
    .step-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        background: #fff;
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 10px;
        box-shadow: none;
        margin-bottom: 24px;
    }
    .step {
        border-radius: 6px;
        padding: 11px 14px;
        text-align: left;
        background: #f4f6f9;
        color: #52616f;
        font-weight: 650;
        border: 1px solid #e2e8f0;
    }
    .step.active {
        background: #123f6d;
        color: white;
        border-color: #123f6d;
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        margin-right: 8px;
        border-radius: 999px;
        background: rgba(18, 63, 109, 0.10);
        color: #123f6d;
        font-size: 12px;
    }
    .step.active .step-number {
        background: rgba(255, 255, 255, 0.18);
        color: #ffffff;
    }
    .info-card {
        border-radius: 8px;
        border: 1px solid #d8dee8;
        border-left: 4px solid #123f6d;
        background: #ffffff;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .warn-card {
        border-radius: 8px;
        border: 1px solid #e1e7ef;
        border-left: 4px solid #78909c;
        background: #fbfcfe;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        background: white;
    }
    .small-muted { color: #667085; font-size: 13px; }
    .status-line {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 10px 0;
        color: #102a43;
        font-size: 14px;
    }
    .status-icon {
        width: 16px;
        height: 16px;
        border-radius: 999px;
        border: 1px solid #9fb0c3;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 16px;
    }
    .status-line.ready .status-icon {
        background: #e8f5ee;
        border-color: #1f9d55;
        color: #137333;
        font-size: 11px;
        font-weight: 800;
    }
    .status-line.ready .status-icon::after { content: "\\2713"; }
    .status-line.waiting .status-icon::after {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: #8a97a6;
    }
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background-color: #123f6d;
        border-color: #123f6d;
        color: #ffffff;
        border-radius: 6px;
        box-shadow: none;
    }
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 6px;
        box-shadow: none;
    }
    .pgrad-report {
        --c-text: #1a1a1a;
        --c-muted: #666666;
        --c-nc: #006d5b;
        --c-prism: #d97706;
        --c-copd: #b91c1c;
        --c-keep: #15803d;
        --c-switch: #0369a1;
        --c-abstain: #7e22ce;
        --c-gate-on: #b91c1c;
        --c-gate-off: #888888;
        background: #f6f8fb;
        padding: 12px;
        border: 1px solid #d8dee8;
        border-radius: 6px;
        margin-top: 8px;
    }
    .pgrad-paper {
        max-width: 1120px;
        margin: 0 auto;
        background: #ffffff;
        padding: 28px;
        border: 1px solid #d0d7de;
        box-shadow: none;
        color: var(--c-text);
        font-family: Arial, Helvetica, sans-serif;
    }
    .pgrad-header-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        padding-bottom: 18px;
        border-bottom: 2px solid #000;
        margin-bottom: 16px;
        align-items: stretch;
    }
    .pgrad-head-box {
        border: 1px solid #eee;
        padding: 12px 15px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-radius: 4px;
        min-height: 86px;
    }
    .pgrad-head-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #666;
        margin-bottom: 4px;
        font-weight: 700;
    }
    .pgrad-head-val {
        font-family: "Times New Roman", Times, serif;
        font-size: 26px;
        font-weight: 900;
        line-height: 1.05;
    }
    .val-nc { color: var(--c-nc); }
    .val-prism { color: var(--c-prism); }
    .val-copd { color: var(--c-copd); }
    .gate-on {
        color: var(--c-gate-on);
        border-color: var(--c-gate-on);
        background: #fff5f5;
    }
    .gate-off { color: var(--c-gate-off); }
    .act-keep { color: var(--c-keep); }
    .act-switch { color: var(--c-switch); }
    .act-abstain {
        color: var(--c-abstain);
        border-color: var(--c-abstain);
        background: #fbf5ff;
    }
    .pgrad-info-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 12px;
        gap: 16px;
        flex-wrap: wrap;
    }
    .pgrad-case-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 18px;
        font-weight: 900;
        color: #000;
        padding-top: 2px;
    }
    .pgrad-prob-chart {
        width: 480px;
        max-width: 100%;
    }
    .pgrad-prob-item {
        display: grid;
        grid-template-columns: 72px 1fr 52px;
        gap: 8px;
        align-items: center;
        margin: 5px 0;
    }
    .pgrad-prob-label {
        font-family: Consolas, Monaco, monospace;
        font-size: 12px;
        font-weight: 800;
        color: #111;
    }
    .pgrad-prob-bar-bg {
        height: 10px;
        background: #ececec;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid #d5d5d5;
    }
    .pgrad-prob-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: currentColor;
    }
    .pgrad-prob-val {
        font-family: Consolas, Monaco, monospace;
        font-size: 12px;
        font-weight: 800;
        text-align: right;
    }
    .pgrad-section-title {
        font-family: "Times New Roman", Times, serif;
        font-weight: 900;
        font-size: 15px;
        border-bottom: 1px solid #000;
        margin-bottom: 8px;
        padding-bottom: 4px;
        text-transform: uppercase;
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
    }
    .pgrad-section-title .small {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 11px;
        color: #666;
        font-weight: 600;
        text-transform: none;
    }
    .pgrad-evidence-box {
        background: #fcfcfc;
        border: 1px solid #999;
        padding: 12px 10px 12px 12px;
        margin-bottom: 14px;
        font-family: Consolas, Monaco, monospace;
        font-size: 10.5px;
        line-height: 1.5;
        white-space: normal;
        height: 180px;
        overflow-y: auto;
        color: #333;
        border-radius: 4px;
    }
    .pgrad-evidence-box b { color: #000; font-weight: 900; }
    .pgrad-evidence-line { margin-bottom: 7px; }
    .pgrad-evidence-meta { color: #666; }
    .pgrad-z-high { color: var(--c-copd); font-weight: 900; }
    .pgrad-z-low { color: var(--c-nc); font-weight: 900; }
    .pgrad-retrieval-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 18px;
        margin-bottom: 12px;
    }
    .pgrad-ret-scroll {
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        background: #fff;
        max-height: 128px;
        overflow-y: auto;
    }
    .pgrad-ret-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
    }
    .pgrad-ret-table th {
        text-align: left;
        border-bottom: 1px solid #666;
        padding: 6px 8px;
        color: #444;
        font-weight: 900;
        background: #fafafa;
        position: sticky;
        top: 0;
    }
    .pgrad-ret-table td {
        border-bottom: 1px solid #eee;
        padding: 6px 8px;
        font-family: Consolas, Monaco, monospace;
        vertical-align: top;
    }
    .pgrad-ret-note {
        font-size: 10.5px;
        color: #666;
        margin-top: 4px;
        font-family: Arial, Helvetica, sans-serif;
    }
    .pgrad-summary-box {
        font-size: 11px;
        line-height: 1.5;
        text-align: justify;
        color: #222;
        height: 140px;
        overflow-y: auto;
        padding: 8px 10px;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        background: #fff;
    }
    .workflow-progress {
        display: grid;
        gap: 10px;
        margin: 12px 0 6px 0;
    }
    .workflow-step {
        display: grid;
        grid-template-columns: 28px 1fr 92px;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #ffffff;
        color: #475467;
        font-size: 14px;
    }
    .workflow-label { font-weight: 700; color: #1f2937; }
    .workflow-state { font-size: 12px; text-align: right; text-transform: uppercase; letter-spacing: 0.4px; }
    .workflow-dot {
        width: 18px;
        height: 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        border: 2px solid #cbd5e1;
        color: #ffffff;
        font-size: 12px;
        font-weight: 900;
    }
    .workflow-step.running {
        border-color: #123f6d;
        background: #f1f6fb;
        color: #123f6d;
    }
    .workflow-step.running .workflow-dot {
        border-color: #123f6d;
        border-top-color: transparent;
        animation: workflow-spin 0.8s linear infinite;
    }
    .workflow-step.complete {
        border-color: #16a34a;
        background: #f0fdf4;
        color: #166534;
    }
    .workflow-step.complete .workflow-dot {
        background: #16a34a;
        border-color: #16a34a;
    }
    .workflow-step.complete .workflow-dot::after { content: "\\2713"; }
    .workflow-step.error {
        border-color: #dc2626;
        background: #fff1f2;
        color: #991b1b;
    }
    .workflow-step.error .workflow-dot {
        background: #dc2626;
        border-color: #dc2626;
    }
    .workflow-step.error .workflow-dot::after { content: "!"; }
    @keyframes workflow-spin {
        to { transform: rotate(360deg); }
    }
    @media (max-width: 850px) {
        .pgrad-paper { padding: 22px; }
        .pgrad-header-grid, .pgrad-retrieval-grid { grid-template-columns: 1fr; }
        .pgrad-prob-item { grid-template-columns: 60px 1fr 48px; }
        .workflow-step { grid-template-columns: 24px 1fr; }
        .workflow-state { grid-column: 2; text-align: left; }
    }
</style>
"""


def render_header() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>PGRAD: QCT-Based Pulmonary Function Evaluation</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_steps(active: int) -> None:
    labels = ["Data & Analysis", "Results"]
    html = '<div class="step-row">'
    for idx, label in enumerate(labels, start=1):
        cls = "step active" if idx == active else "step"
        html += f'<div class="{cls}"><span class="step-number">{idx}</span>{esc(label)}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_workflow_progress(step_states: dict[str, str], target=None) -> None:
    target = target or st
    state_labels = {
        "pending": "Pending",
        "running": "Running",
        "complete": "Complete",
        "error": "Error",
    }
    html_parts = ['<div class="workflow-progress">']
    for key, label in WORKFLOW_STEPS:
        state = step_states.get(key, "pending")
        html_parts.append(
            f'<div class="workflow-step {esc(state)}">'
            f'<span class="workflow-dot"></span>'
            f'<span class="workflow-label">{esc(label)}</span>'
            f'<span class="workflow-state">{esc(state_labels.get(state, state))}</span>'
            "</div>"
        )
    html_parts.append("</div>")
    target.markdown("".join(html_parts), unsafe_allow_html=True)


def get_secret(name: str) -> str:
    if not name:
        return ""
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def esc(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def fmt_num(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def label_class(label: str) -> str:
    normalized = str(label or "").strip().lower()
    if normalized == "nc":
        return "val-nc"
    if normalized == "prism":
        return "val-prism"
    if normalized == "copd":
        return "val-copd"
    return ""


def decision_class(decision: str) -> str:
    normalized = str(decision or "").strip().lower()
    if normalized == "switch":
        return "act-switch"
    if normalized == "abstain":
        return "act-abstain"
    return "act-keep"


def render_prob_bars(case: dict) -> str:
    colors = {"NC": "#006d5b", "PRISm": "#d97706", "COPD": "#b91c1c"}
    rows = []
    for label in ["NC", "PRISm", "COPD"]:
        value = float(case.get(f"p_{label}", 0.0) or 0.0)
        width = max(0.0, min(100.0, value * 100.0))
        rows.append(
            f"""
            <div class="pgrad-prob-item">
                <div class="pgrad-prob-label">{label}</div>
                <div class="pgrad-prob-bar-bg">
                    <div class="pgrad-prob-bar-fill" style="width:{width:.1f}%; color:{colors[label]};"></div>
                </div>
                <div class="pgrad-prob-val">{value:.3f}</div>
            </div>
            """
        )
    return "\n".join(rows)


def render_evidence_chain(evidence_items: list[dict]) -> str:
    if not evidence_items:
        return '<div class="pgrad-evidence-line">No evidence available.</div>'
    rows = []
    for idx, item in enumerate(evidence_items[:15], start=1):
        z_value = item.get("z")
        z_class = "pgrad-z-high" if isinstance(z_value, (int, float)) and z_value >= 0 else "pgrad-z-low"
        score = fmt_num(item.get("score"))
        text = esc(item.get("text", ""))
        feature = esc(item.get("feature", ""))
        z_text = fmt_num(z_value)
        rows.append(
            f"""
            <div class="pgrad-evidence-line">
                <b>E{idx}</b> <b>{feature}</b><br>
                {text}<br>
                <span class="pgrad-evidence-meta">z=<span class="{z_class}">{z_text}</span>; score={score}</span>
            </div>
            """
        )
    return "\n".join(rows)


def render_retrieval_rows(items: list[dict], include_true: bool) -> str:
    if not items:
        return '<tr><td colspan="2" style="color:#999">None</td></tr>'
    rows = []
    for item in items[:10]:
        patient_id = esc(item.get("patient_id", ""))
        sim = fmt_num(item.get("sim"))
        true_label = esc(item.get("true_label", "")) if include_true else ""
        pred_base = esc(item.get("pred_base", ""))
        evidence = esc(" | ".join([str(x) for x in (item.get("matched_evidence") or [])[:2]]))
        meta_parts = []
        if include_true and true_label:
            meta_parts.append(f"true={true_label}")
        if pred_base:
            meta_parts.append(f"pred={pred_base}")
        meta = "; ".join(meta_parts)
        rows.append(
            f"""
            <tr>
                <td>{patient_id}<br><span style="color:#666">{meta}</span><br><span style="color:#777">{evidence}</span></td>
                <td>{sim}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def extract_summary(markdown_text: str) -> str:
    text = str(markdown_text or "").strip()
    match = re.search(r"###\s*8\).*?(?:\n|$)(.*?)(?=\n###\s*\d+\)|\Z)", text, flags=re.S)
    if match and match.group(1).strip():
        text = match.group(1).strip()
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = text.replace("**", "")
    return text.strip()


def local_diagnostic_summary(case: dict) -> str:
    final_label = str(case.get("pred_label", "") or "-")
    return f"This is the final diagnostic opinion of the base model: {final_label}."


def compact_html(markup: str) -> str:
    return "".join(line.strip() for line in str(markup).splitlines())


def render_interpretability_report(case: dict) -> None:
    final_label = str(case.get("pred_label", "") or "-")
    gate_on = bool(case.get("gate_triggered", False))
    decision = str(case.get("decision", "") or "-")
    gate_text = "ON" if gate_on else "OFF"
    margin_text = fmt_num(case.get("margin_final", case.get("margin_base")))
    risk = esc(case.get("risk_level", ""))
    risk_reason = esc(case.get("risk_reason", ""))
    if str(case.get("llm_model", "") or "").strip():
        summary = esc(extract_summary(case.get("markdown", ""))).replace("\n", "<br>")
    else:
        summary = esc(local_diagnostic_summary(case))
    support_count = len(case.get("support") or [])
    confuse_count = len(case.get("confuse") or [])
    report_html = f"""
    <div class="pgrad-report">
      <div class="pgrad-paper">
        <div class="pgrad-header-grid">
          <div class="pgrad-head-box">
            <div class="pgrad-head-label">Final Diagnosis</div>
            <div class="pgrad-head-val {label_class(final_label)}">{esc(final_label)}</div>
          </div>
          <div class="pgrad-head-box {'gate-on' if gate_on else 'gate-off'}">
            <div class="pgrad-head-label">Gate Status</div>
            <div class="pgrad-head-val">{gate_text}</div>
            <div style="font-size:10px; color:#666; margin-top:4px;">Confidence Margin: {margin_text}</div>
          </div>
          <div class="pgrad-head-box {decision_class(decision)}">
            <div class="pgrad-head-label">System Decision</div>
            <div class="pgrad-head-val">{esc(decision.upper())}</div>
            <div style="font-size:10px; color:#666; margin-top:4px;">Risk: {risk}</div>
          </div>
        </div>

        <div class="pgrad-info-row">
          <div class="pgrad-case-title">CASE: {esc(case.get("patient_id", ""))}</div>
          <div class="pgrad-prob-chart" aria-label="Probability chart">
            {render_prob_bars(case)}
          </div>
        </div>

        <div>
          <div class="pgrad-section-title">
            <span>Evidence Chain (Full Trace)</span>
            <span class="small">Top current-case evidence</span>
          </div>
          <div class="pgrad-evidence-box">{render_evidence_chain(case.get("query_evidence") or [])}</div>
        </div>

        <div class="pgrad-retrieval-grid">
          <div>
            <div class="pgrad-section-title"><span>Correct Supports</span><span class="small">n={support_count}</span></div>
            <div class="pgrad-ret-scroll">
              <table class="pgrad-ret-table">
                <thead><tr><th>Case ID</th><th>Sim</th></tr></thead>
                <tbody>{render_retrieval_rows(case.get("support") or [], include_true=False)}</tbody>
              </table>
            </div>
            <div class="pgrad-ret-note">Retrieved from the correct evidence library.</div>
          </div>
          <div>
            <div class="pgrad-section-title"><span>Confusions / Counterexamples</span><span class="small">n={confuse_count}</span></div>
            <div class="pgrad-ret-scroll">
              <table class="pgrad-ret-table">
                <thead><tr><th>Case ID</th><th>Sim</th></tr></thead>
                <tbody>{render_retrieval_rows(case.get("confuse") or [], include_true=True)}</tbody>
              </table>
            </div>
            <div class="pgrad-ret-note">{risk_reason}</div>
          </div>
        </div>

        <div>
          <div class="pgrad-section-title">Summary (from MD)</div>
          <div class="pgrad-summary-box">{summary or "No summary."}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(compact_html(report_html), unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "uploaded_df": None,
        "uploaded_name": "",
        "uploaded_type": "",
        "analysis": None,
        "data_load_notice": "",
        "active_step": 1,
        "pending_section": None,
        "upload_widget_key": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> str:
    st.sidebar.markdown("## Navigation")
    sections = ["Data & Analysis", "Results"]
    legacy_sections = {
        "Data Upload": "Data & Analysis",
        "Analysis Selection": "Data & Analysis",
        "Results Review": "Results",
    }
    active_index = max(0, min(int(st.session_state.get("active_step", 1)) - 1, len(sections) - 1))
    pending_section = st.session_state.pop("pending_section", None)
    pending_section = legacy_sections.get(pending_section, pending_section)
    if st.session_state.get("section_radio") in legacy_sections:
        st.session_state.section_radio = legacy_sections[st.session_state.section_radio]
    if pending_section in sections:
        st.session_state.active_step = sections.index(pending_section) + 1
        st.session_state.section_radio = pending_section
    elif st.session_state.get("section_radio") not in sections:
        st.session_state.section_radio = sections[active_index]

    radio_index = sections.index(st.session_state.section_radio)
    section = st.sidebar.radio(
        "Choose Section:",
        sections,
        index=radio_index,
        key="section_radio",
    )
    st.session_state.active_step = sections.index(section) + 1
    st.sidebar.divider()
    if st.sidebar.button("Reset Workflow", use_container_width=True):
        st.session_state.uploaded_df = None
        st.session_state.uploaded_name = ""
        st.session_state.uploaded_type = ""
        st.session_state.analysis = None
        st.session_state.active_step = 1
        st.session_state.pending_section = "Data & Analysis"
        st.session_state.upload_widget_key = int(st.session_state.get("upload_widget_key", 0)) + 1
        st.rerun()
    st.sidebar.divider()
    st.sidebar.markdown("### Current Status")
    data_class = "ready" if st.session_state.uploaded_df is not None else "waiting"
    analysis_class = "ready" if st.session_state.analysis is not None else "waiting"
    data_text = "Ready" if st.session_state.uploaded_df is not None else "Not loaded"
    analysis_text = "Completed" if st.session_state.analysis is not None else "Not run"
    st.sidebar.markdown(
        f"""
        <div class="status-line {data_class}">
            <span class="status-icon"></span><span><b>Data</b>: {data_text}</span>
        </div>
        <div class="status-line {analysis_class}">
            <span class="status-icon"></span><span><b>Analysis</b>: {analysis_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return section


def data_and_analysis_step() -> None:
    render_steps(1)
    st.markdown("## Data & Analysis")
    demo_groups = configured_demo_groups()
    left, right = st.columns([2, 1])
    with left:
        group_names = [group["name"] for group in demo_groups]
        if st.session_state.get("selected_data_type") not in group_names:
            st.session_state.selected_data_type = group_names[0]

        st.markdown("### Data Source")
        source_mode = st.radio(
            "Data source",
            ["Upload Dataset", "Use Demo Data"],
            horizontal=True,
            label_visibility="collapsed",
            key="data_source_mode",
        )

        st.markdown("### CT Acquisition")
        if source_mode == "Upload Dataset":
            selected_group_name = st.radio(
                "CT acquisition type",
                group_names,
                horizontal=True,
                key="selected_data_type",
            )
            uploaded = st.file_uploader(
                "Upload CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                key=f"upload_widget_{st.session_state.upload_widget_key}",
            )
            if uploaded is not None:
                upload_name = getattr(uploaded, "name", "")
                is_new_upload = (
                    st.session_state.uploaded_df is None
                    or st.session_state.uploaded_name != upload_name
                    or st.session_state.uploaded_type != selected_group_name
                )
                if is_new_upload:
                    try:
                        df = read_any_table_from_upload(uploaded)
                        st.session_state.uploaded_df = df
                        st.session_state.uploaded_name = upload_name
                        st.session_state.uploaded_type = selected_group_name
                        st.session_state.analysis = None
                        st.session_state.active_step = 1
                        st.session_state.data_load_notice = (
                            f"Dataset loaded: {len(df)} rows and {len(df.columns)} columns. "
                            "Raw values and patient identifiers are hidden."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to read uploaded file: {exc}")
        else:
            if not demo_groups:
                st.info("No demo dataset is configured.")
            else:
                st.caption("Select a configured demo cohort.")
                demo_cols = st.columns(len(demo_groups))
                for demo_col, group in zip(demo_cols, demo_groups):
                    with demo_col:
                        if st.button(group["name"], key=f"load_demo_{group['key']}", type="secondary", use_container_width=True):
                            try:
                                df = pd.read_csv(group["path"])
                                st.session_state.uploaded_df = df
                                st.session_state.uploaded_name = group["path"].name
                                st.session_state.uploaded_type = group["name"]
                                st.session_state.selected_data_type = group["name"]
                                st.session_state.analysis = None
                                st.session_state.active_step = 1
                                st.session_state.data_load_notice = (
                                    f"Demo data loaded: {group['name']} · "
                                    f"{len(df)} rows and {len(df.columns)} columns."
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed to load demo cases: {exc}")

        data_load_notice = st.session_state.pop("data_load_notice", "")
        if data_load_notice:
            st.success(data_load_notice)

        if st.session_state.uploaded_df is not None:
            loaded_df = st.session_state.uploaded_df
            loaded_type = st.session_state.uploaded_type or "Selected data"
            st.info(
                f"Data ready: {loaded_type} · {len(loaded_df)} rows · "
                f"{len(loaded_df.columns)} columns · preview disabled"
            )
        else:
            st.caption("No dataset loaded.")
    with right:
        st.markdown(
            """
            <div class="info-card">
                <h3>Required data</h3>
                <p><b>Patient ID</b></p>
                <p><b>Quantitative CT parameters</b></p>
                <p><b>CT acquisition type</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="warn-card">
                <h4>Data privacy</h4>
                <p>Uploaded data are processed within the current session and are not retained after the session ends.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Analysis")
    render_analysis_controls()


def render_analysis_controls() -> None:
    if st.session_state.uploaded_df is None:
        st.info("Load a dataset or demo cohort to enable analysis.")
        return

    df = st.session_state.uploaded_df
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("CT Acquisition", st.session_state.uploaded_type or "Selected")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        use_llm = st.toggle(
            "Generate optional narrative report",
            value=True,
            help="Requires a configured API key. Core PGRAD analysis runs without this option.",
        )
        limit = st.number_input(
            "Optional case limit",
            min_value=1,
            max_value=max(1, len(df)),
            value=1,
            step=1,
            help="Cannot exceed the number of loaded cases.",
        )
    with col_b:
        api_key = ""
        api_base_url = None
        model_name = None
        if use_llm:
            provider = st.selectbox("API Provider", list(LLM_PROVIDER_DEFAULTS.keys()))
            defaults = LLM_PROVIDER_DEFAULTS[provider]
            api_key = get_secret(defaults["secret"])
            api_base_url = st.text_input("Base URL", value=defaults["base_url"])
            model_name = st.text_input("Model name", value=defaults["model"])
            if not api_key:
                st.warning("Set the API key in Streamlit Secrets.")
            else:
                st.caption("API key loaded from Streamlit Secrets.")
        else:
            st.caption("Local deterministic analysis runs without an API key.")

    ran_analysis = False
    if st.button("Run Analysis", type="primary", use_container_width=True):
        step_states = {key: "pending" for key, _ in WORKFLOW_STEPS}
        progress_box = st.empty()
        render_workflow_progress(step_states, progress_box)

        def update_progress(step_key: str, state: str) -> None:
            step_states[step_key] = state
            render_workflow_progress(step_states, progress_box)

        with st.status("Running analysis...", expanded=True) as status:
            try:
                output = process_cases(
                    df,
                    use_llm=use_llm,
                    api_key=api_key or None,
                    limit=int(limit),
                    api_base_url=api_base_url or None,
                    model_name=model_name or None,
                    progress_callback=update_progress,
                )
                st.session_state.analysis = output
                st.session_state.active_step = 1
                ran_analysis = True
                status.update(label="Analysis completed.", state="complete")
                st.success("Analysis completed.")
                if output.get("missing"):
                    st.info(
                        "Unavailable model features were median-imputed by the trained pipeline: "
                        + ", ".join(output["missing"])
                    )
            except Exception as exc:
                for key, state in step_states.items():
                    if state == "running":
                        step_states[key] = "error"
                        break
                render_workflow_progress(step_states, progress_box)
                status.update(label="Analysis failed.", state="error")
                st.error(str(exc))

    if st.session_state.analysis is not None:
        if not ran_analysis:
            st.success("Analysis completed.")
        if st.button("View Results", key="view_results_after_workflow", type="primary", use_container_width=True):
            st.session_state.active_step = 2
            st.session_state.pending_section = "Results"
            st.rerun()


def results_step() -> None:
    render_steps(2)
    st.markdown("## Results")
    analysis = st.session_state.analysis
    if analysis is None:
        st.info("Run the analysis first.")
        return

    results = analysis["results"]
    cases = analysis["cases"]
    llm_mode = bool(
        not results.empty
        and "llm_model" in results
        and results["llm_model"].fillna("").astype(str).str.strip().ne("").any()
    )

    if not llm_mode:
        model_prediction = pd.DataFrame()
        if "patient_id" in results:
            model_prediction["patient_id"] = results["patient_id"]
        model_prediction["predicted_class"] = results["pred_base"] if "pred_base" in results else results["pred_label"]
        model_prediction["p_NC"] = results["p_base_NC"] if "p_base_NC" in results else results["p_NC"]
        model_prediction["p_PRISm"] = results["p_base_PRISm"] if "p_base_PRISm" in results else results["p_PRISm"]
        model_prediction["p_COPD"] = results["p_base_COPD"] if "p_base_COPD" in results else results["p_COPD"]
        st.markdown("### Model Prediction")
        st.dataframe(model_prediction, use_container_width=True, height=330)
        return

    st.markdown("### Overall Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cases", len(results))
    col2.metric(
        "Gate Triggered",
        int(results["gate_triggered"].sum()) if not results.empty and "gate_triggered" in results else 0,
    )
    col3.metric(
        "Review Flags",
        int(results["review_flag"].sum()) if not results.empty and "review_flag" in results else 0,
    )
    col4.metric(
        "Narrative Reports",
        int(results["llm_used"].sum()) if llm_mode and "llm_used" in results else 0,
    )

    if llm_mode:
        display_results = results
    else:
        simple_cols = ["patient_id", "pred_label", "p_NC", "p_PRISm", "p_COPD"]
        display_results = results[[col for col in simple_cols if col in results.columns]]

    st.markdown("### Classification / Evaluation Results")
    st.dataframe(display_results, use_container_width=True, height=330)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Download Results CSV",
            data=display_results.to_csv(index=False).encode("utf-8-sig"),
            file_name="external_final_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if llm_mode:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for case in cases:
                zf.writestr(f"cases/{case['patient_id']}.md", case["markdown"])
            zf.writestr("external_final_predictions.csv", results.to_csv(index=False))
        with dl_col2:
            st.download_button(
                "Download Case Reports ZIP",
                data=zip_buffer.getvalue(),
                file_name="pgrad_case_reports.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.markdown("### Patient-level Details")
    patient_ids = [case["patient_id"] for case in cases]
    selected = st.selectbox("Choose patient", patient_ids)
    case = next(item for item in cases if item["patient_id"] == selected)

    c1, c2, c3 = st.columns(3)
    c1.metric("Base Prediction", case.get("pred_base", "-"))
    c2.metric("Final Label", case.get("pred_label", "-"))
    c3.metric("Decision", case.get("decision", "-"))

    has_report = bool(str(case.get("markdown", "") or "").strip() or str(case.get("llm_model", "") or "").strip())

    def render_classification_tab() -> None:
        p1, p2, p3 = st.columns(3)
        p1.metric("NC", fmt_num(case.get("p_NC", case.get("p_base_NC", 0.0))))
        p2.metric("PRISm", fmt_num(case.get("p_PRISm", case.get("p_base_PRISm", 0.0))))
        p3.metric("COPD", fmt_num(case.get("p_COPD", case.get("p_base_COPD", 0.0))))
        patient_row = results[results["patient_id"].astype(str) == str(selected)]
        if not patient_row.empty:
            st.dataframe(patient_row, use_container_width=True, height=160)

    def render_evidence_tab() -> None:
        ev_df = pd.DataFrame(case.get("query_evidence") or [])
        if ev_df.empty:
            st.info("No evidence available for this case.")
        else:
            st.dataframe(ev_df, use_container_width=True)
        with st.expander("Correct supports"):
            st.dataframe(pd.DataFrame(case.get("support") or []), use_container_width=True)
        with st.expander("Confusions / counterexamples"):
            st.dataframe(pd.DataFrame(case.get("confuse") or []), use_container_width=True)

    def render_generated_report_tab() -> None:
        if has_report:
            render_interpretability_report(case)
            with st.expander("Raw Markdown Report"):
                st.markdown(case.get("markdown", ""))
        else:
            st.info("No generated narrative report for this run.")

    if has_report:
        report_tab, evidence_tab, classification_tab = st.tabs(
            ["Generated Report", "Evidence / QCT Findings", "Classification"]
        )
        with report_tab:
            render_generated_report_tab()
        with evidence_tab:
            render_evidence_tab()
        with classification_tab:
            render_classification_tab()
    else:
        classification_tab, evidence_tab, report_tab = st.tabs(
            ["Classification", "Evidence / QCT Findings", "Generated Report"]
        )
        with classification_tab:
            render_classification_tab()
        with evidence_tab:
            render_evidence_tab()
        with report_tab:
            render_generated_report_tab()


def main() -> None:
    init_state()
    section = render_sidebar()
    render_header()
    if section == "Data & Analysis":
        data_and_analysis_step()
        return
    if section == "Results":
        results_step()
        return


if __name__ == "__main__":
    main()
