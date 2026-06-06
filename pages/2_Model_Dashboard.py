"""
pages/2_Model_Dashboard.py — Model card, evaluation metrics, and predictions.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config

st.set_page_config(
    page_title='Model Dashboard · PatientTrialMapper',
    page_icon='📊',
    layout='wide',
)

st.markdown("""
<style>
.stApp { background: #f1f5f9; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }
footer { visibility: hidden; }
div[data-testid="stToolbar"] { visibility: hidden; }

.ptm-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.75rem;
    box-shadow: 0 4px 24px rgba(37,99,235,0.25);
}
.ptm-header h1 { color:#fff; font-size:1.7rem; font-weight:800; margin:0; letter-spacing:-0.5px; }
.ptm-header p  { color:#bfdbfe; font-size:0.88rem; margin:0.2rem 0 0; }

.section-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b; margin-bottom: 0.75rem;
}

/* ── Model info card ── */
.info-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0; border-bottom: 1px solid #f1f5f9;
    font-size: 0.875rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #64748b; font-weight: 500; }
.info-val { color: #0f172a; font-weight: 600; font-family: monospace; font-size: 0.85rem; }

/* ── Metric tiles ── */
.metric-tile {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.25rem; text-align: center;
}
.metric-value { font-size: 2.25rem; font-weight: 800; }
.metric-label { font-size: 0.78rem; color: #64748b; margin-top: 0.15rem; font-weight: 500; }
.metric-green { color: #15803d; }
.metric-blue  { color: #2563eb; }
.metric-amber { color: #b45309; }

/* ── Status chips ── */
.chip-green { display:inline-block; background:#dcfce7; color:#15803d; border-radius:5px; padding:2px 10px; font-size:0.78rem; font-weight:700; }
.chip-amber { display:inline-block; background:#fef9c3; color:#b45309; border-radius:5px; padding:2px 10px; font-size:0.78rem; font-weight:700; }
.chip-gray  { display:inline-block; background:#f1f5f9; color:#64748b; border-radius:5px; padding:2px 10px; font-size:0.78rem; font-weight:600; }

/* ── Placeholder ── */
.dash-placeholder {
    background: white; border: 2px dashed #e2e8f0; border-radius: 12px;
    padding: 2rem; text-align: center; color: #94a3b8; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ptm-header">
    <h1>📊 Model Dashboard</h1>
    <p>Model configuration, evaluation metrics, and test-set predictions.</p>
</div>
""", unsafe_allow_html=True)


# ── 1. Model card ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Model Configuration</div>', unsafe_allow_html=True)

model_status = (
    '<span class="chip-green">● Adapters Ready</span>'
    if config.ADAPTER_DIR.exists()
    else '<span class="chip-amber">⚡ Not Trained Yet</span>'
)

rows = [
    ('Base Model',       config.BASE_MODEL),
    ('Fine-tuning',      'QLoRA — 4-bit NF4 + LoRA adapters (peft + trl)'),
    ('LoRA Rank (r)',    str(config.LORA_R)),
    ('LoRA Alpha (α)',   str(config.LORA_ALPHA)),
    ('LoRA Dropout',     str(config.LORA_DROPOUT)),
    ('Target Modules',   ', '.join(config.LORA_MODULES)),
    ('Max Seq Length',   f'{config.MAX_SEQ_LEN} tokens'),
    ('Training Steps',   str(config.MAX_STEPS)),
    ('Batch Size',       f'{config.BATCH_SIZE} × {config.GRAD_ACC} grad acc = {config.BATCH_SIZE * config.GRAD_ACC} effective'),
    ('Learning Rate',    str(config.LEARNING_RATE)),
    ('Adapter Status',   model_status),
]

info_html = '<div class="info-card">'
for key, val in rows:
    info_html += f'<div class="info-row"><span class="info-key">{key}</span><span class="info-val">{val}</span></div>'
info_html += '</div>'
st.markdown(info_html, unsafe_allow_html=True)


# ── 2. Evaluation metrics ──────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem;">Evaluation Metrics</div>', unsafe_allow_html=True)

eval_path = config.DATA_DIR / 'eval_results.json'
if eval_path.exists():
    with open(eval_path) as f:
        metrics = json.load(f)

    def _color(v):
        if isinstance(v, float):
            if v >= 0.85: return 'metric-green'
            if v >= 0.70: return 'metric-blue'
            return 'metric-amber'
        return 'metric-blue'

    tiles = [
        ('F1 Score',   metrics.get('f1', '—'),        'Eligible class'),
        ('Precision',  metrics.get('precision', '—'),  'Eligible class'),
        ('Recall',     metrics.get('recall', '—'),     'Eligible class'),
        ('Accuracy',   metrics.get('accuracy', '—'),   'All classes'),
    ]
    cols = st.columns(4)
    for col, (label, val, sub) in zip(cols, tiles):
        display = f'{val:.2%}' if isinstance(val, float) else str(val)
        css     = _color(val)
        col.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value {css}">{display}</div>
            <div class="metric-label">{label}</div>
            <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    unknown = metrics.get('unknown', 0)
    if unknown:
        st.info(f'⚠️  {unknown} prediction(s) returned "Unknown" and were routed to human review.')
else:
    st.markdown("""
    <div class="dash-placeholder">
        <b>No evaluation results yet.</b><br>
        Run <code>python evaluate.py</code> after training to populate this section.
    </div>
    """, unsafe_allow_html=True)


# ── 3. Confusion matrix ────────────────────────────────────────────────────────
cm_path = config.DATA_DIR / 'confusion_matrix.png'
if cm_path.exists():
    st.markdown('<div class="section-label" style="margin-top:1.75rem;">Confusion Matrix</div>', unsafe_allow_html=True)
    col_img, col_gap = st.columns([1, 1])
    with col_img:
        st.image(str(cm_path), use_container_width=True)


# ── 4. Test predictions ────────────────────────────────────────────────────────
preds_path = config.DATA_DIR / 'test_predictions.json'
if preds_path.exists():
    st.markdown('<div class="section-label" style="margin-top:1.75rem;">Test-Set Predictions</div>', unsafe_allow_html=True)
    preds_df = pd.read_json(preds_path)

    # Select and rename for display
    display_cols = ['response', 'predicted', 'reasoning']
    display_cols = [c for c in display_cols if c in preds_df.columns]
    view = preds_df[display_cols].copy()
    view = view.rename(columns={'response': 'True Label', 'predicted': 'Predicted', 'reasoning': 'Reasoning'})

    def _highlight(row):
        correct = row.get('True Label') == row.get('Predicted')
        color   = '#f0fdf4' if correct else '#fef2f2'
        return [f'background-color: {color}'] * len(row)

    st.dataframe(
        view.style.apply(_highlight, axis=1),
        use_container_width=True,
        height=min(500, 50 + 35 * len(view)),
    )

    correct   = (preds_df['response'] == preds_df['predicted']).sum() if 'predicted' in preds_df else 0
    incorrect = len(preds_df) - correct
    st.markdown(f"""
    <div style="font-size:0.82rem;color:#64748b;margin-top:0.5rem;">
        <span style="color:#15803d;font-weight:700;">✓ {correct} correct</span>
        &nbsp;·&nbsp;
        <span style="color:#b91c1c;font-weight:700;">✗ {incorrect} incorrect</span>
        &nbsp;·&nbsp; {len(preds_df)} total
    </div>
    """, unsafe_allow_html=True)
