"""
app.py — PatientTrialMapper | Clinical Trial Eligibility Screener
Run: streamlit run app.py
"""

import traceback
from datetime import datetime

import streamlit as st

import config
try:
    from data_prep import SAMPLE_DATA
except Exception as _e:
    SAMPLE_DATA = []
    _SAMPLE_DATA_ERR = str(_e)
else:
    _SAMPLE_DATA_ERR = None

# ── Page config (must be first st call) ───────────────────────────────────────
st.set_page_config(
    page_title='PatientTrialMapper',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={'About': 'PatientTrialMapper — LLM-powered clinical trial eligibility screening.'},
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────────────────────── */
.stApp { background: #f1f5f9; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }
footer { visibility: hidden; }
div[data-testid="stToolbar"] { visibility: hidden; }

/* ── Page header ──────────────────────────────────────────────────────────── */
.ptm-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 4px 24px rgba(37,99,235,0.25);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
}
.ptm-header-left h1 {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.ptm-header-left p {
    color: #bfdbfe;
    font-size: 0.9rem;
    margin: 0.25rem 0 0;
}
.ptm-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    white-space: nowrap;
}
.ptm-badge-ready { background: rgba(74,222,128,0.2); border: 1px solid rgba(74,222,128,0.5); color: #bbf7d0; }
.ptm-badge-demo  { background: rgba(251,191,36,0.2);  border: 1px solid rgba(251,191,36,0.5);  color: #fde68a; }

/* ── Section labels ──────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 0.4rem;
}

/* ── Result card ─────────────────────────────────────────────────────────── */
.result-card { border-radius: 14px; padding: 1.5rem; animation: fadeUp 0.35s ease; }
@keyframes fadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }

.result-eligible     { background: #f0fdf4; border: 2px solid #16a34a; }
.result-not-eligible { background: #fef2f2; border: 2px solid #dc2626; }
.result-unknown      { background: #fffbeb; border: 2px solid #d97706; }

.result-verdict { font-size: 1.5rem; font-weight: 800; margin-bottom: 0.4rem; }
.verdict-eligible     { color: #15803d; }
.verdict-not-eligible { color: #b91c1c; }
.verdict-unknown      { color: #b45309; }

.result-meta { font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.9rem; }

.reasoning-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #64748b; margin-bottom: 0.4rem;
}
.reasoning-text {
    background: rgba(0,0,0,0.035);
    border-left: 4px solid #94a3b8;
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    color: #334155;
    font-size: 0.9rem;
    line-height: 1.65;
    margin: 0;
}

/* ── Placeholder card ────────────────────────────────────────────────────── */
.placeholder-card {
    background: white;
    border: 2px dashed #cbd5e1;
    border-radius: 14px;
    padding: 3rem 2rem;
    text-align: center;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
}
.placeholder-icon { font-size: 3rem; line-height: 1; }
.placeholder-title { font-size: 1.05rem; font-weight: 700; color: #475569; margin: 0; }
.placeholder-sub   { font-size: 0.85rem; color: #94a3b8; max-width: 240px; margin: 0; }

/* ── Setup card ──────────────────────────────────────────────────────────── */
.setup-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}
.setup-card h4 { color: #1e40af; font-size: 0.9rem; margin: 0 0 0.6rem; font-weight: 700; }
.setup-card ol { color: #1e40af; font-size: 0.85rem; margin: 0; padding-left: 1.2rem; }
.setup-card li { margin: 0.2rem 0; }
.setup-card code {
    background: #dbeafe; border-radius: 4px;
    padding: 1px 6px; font-size: 0.82rem; color: #1d4ed8;
}

/* ── History ─────────────────────────────────────────────────────────────── */
.hist-item {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin: 0.35rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.86rem;
    gap: 1rem;
}
.hist-verdict-e  { color: #15803d; font-weight: 700; white-space: nowrap; }
.hist-verdict-ne { color: #b91c1c; font-weight: 700; white-space: nowrap; }
.hist-verdict-u  { color: #b45309; font-weight: 700; white-space: nowrap; }
.hist-snippet    { color: #475569; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hist-time       { color: #94a3b8; font-size: 0.76rem; white-space: nowrap; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
.sb-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.sb-box-title {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 0.6rem;
}
.sb-status-on  { color: #15803d; font-weight: 700; font-size: 0.88rem; }
.sb-status-off { color: #b45309; font-weight: 700; font-size: 0.88rem; }
.sb-chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 0.78rem;
    color: #475569;
    margin: 2px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_model():
    try:
        from predict import load_model
        return load_model()
    except SystemExit:
        return None, None
    except Exception:
        return None, None


# ── Demo-mode prediction (no model needed) ────────────────────────────────────
def _demo_predict(patient: str, criteria: str) -> dict:
    """Return a pre-computed result when the model isn't loaded."""
    pat_lower = patient.strip().lower()
    for ex in SAMPLE_DATA:
        if ex['patient'].strip().lower()[:60] in pat_lower or pat_lower[:60] in ex['patient'].strip().lower():
            return {
                'label':      ex['label'],
                'reasoning':  ex.get('reasoning', ''),
                'confidence': 'Demo',
            }
    return {
        'label':      'Unknown',
        'reasoning':  'No pre-computed example matched. Train the model and set ADAPTER_DIR to run real inference.',
        'confidence': 'N/A',
    }


# ── Session state ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'patient_text' not in st.session_state:
    st.session_state.patient_text = ''
if 'criteria_text' not in st.session_state:
    st.session_state.criteria_text = ''


# ── Model status ──────────────────────────────────────────────────────────────
model_ready = config.ADAPTER_DIR.exists()

if _SAMPLE_DATA_ERR:
    st.warning(f'Sample data load error (examples unavailable): {_SAMPLE_DATA_ERR}')

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0 1rem;">
        <span style="font-size:2.2rem;">🏥</span>
        <div style="font-weight:800; font-size:1rem; color:#0f172a; margin-top:0.25rem;">PatientTrialMapper</div>
    </div>
    """, unsafe_allow_html=True)

    # Model status
    st.markdown('<div class="sb-box">', unsafe_allow_html=True)
    st.markdown('<div class="sb-box-title">Model Status</div>', unsafe_allow_html=True)
    if model_ready:
        st.markdown('<span class="sb-status-on">● Model Ready</span>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.78rem;color:#64748b;margin-top:4px;">Adapters: <code style="font-size:0.76rem;">{config.ADAPTER_DIR}</code></div>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="sb-status-off">● Demo Mode</span>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.78rem;color:#92400e;margin-top:4px;">Run <code style="font-size:0.76rem;">train.py</code> to enable live inference.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Example picker
    st.markdown('<div class="sb-box">', unsafe_allow_html=True)
    st.markdown('<div class="sb-box-title">Load Example</div>', unsafe_allow_html=True)
    example_names = [
        f"{'✅' if ex['label']=='Eligible' else '❌'} {ex['patient'][:45]}…"
        for ex in SAMPLE_DATA
    ]
    chosen = st.selectbox('Load Example', ['— select —'] + example_names, label_visibility='collapsed')
    if chosen != '— select —':
        idx = example_names.index(chosen)
        st.session_state.patient_text  = SAMPLE_DATA[idx]['patient']
        st.session_state.criteria_text = SAMPLE_DATA[idx]['criteria']
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Model config
    st.markdown('<div class="sb-box">', unsafe_allow_html=True)
    st.markdown('<div class="sb-box-title">Model Config</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#374151; line-height:1.8;">
        <div><b>Base:</b> Mistral-7B-Instruct v0.3</div>
        <div><b>Method:</b> QLoRA (4-bit NF4)</div>
        <div>
            <span class="sb-chip">r={config.LORA_R}</span>
            <span class="sb-chip">α={config.LORA_ALPHA}</span>
            <span class="sb-chip">dropout={config.LORA_DROPOUT}</span>
        </div>
        <div style="margin-top:4px;"><b>Steps:</b> {config.MAX_STEPS} &nbsp;|&nbsp; <b>LR:</b> {config.LEARNING_RATE}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.75rem;color:#94a3b8;text-align:center;padding-top:0.5rem;">Use the sidebar navigation above to access Batch Screener and Model Dashboard.</div>', unsafe_allow_html=True)


# ── Page header ───────────────────────────────────────────────────────────────
badge_html = (
    '<span class="ptm-badge ptm-badge-ready">● Model Ready</span>'
    if model_ready else
    '<span class="ptm-badge ptm-badge-demo">⚡ Demo Mode</span>'
)
st.markdown(f"""
<div class="ptm-header">
    <div class="ptm-header-left">
        <h1>🏥 PatientTrialMapper</h1>
        <p>AI-powered clinical trial eligibility screening · Fine-tuned Mistral-7B + QLoRA</p>
    </div>
    <div>{badge_html}</div>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.markdown("""
    <div class="setup-card">
        <h4>⚡ Running in Demo Mode</h4>
        <ol>
            <li>Run <code>python data_prep.py</code> to generate training data</li>
            <li>Run <code>python train.py</code> on a GPU (or upload to Kaggle and run there)</li>
            <li>Restart this app — the model will load automatically</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


# ── Main layout: inputs (left) + result (right) ───────────────────────────────
left, right = st.columns([6, 4], gap='large')

with left:
    st.markdown('<div class="section-label">Patient Narrative</div>', unsafe_allow_html=True)
    patient_input = st.text_area(
        label='Patient Narrative',
        placeholder='Describe the patient age, primary diagnosis, medications, relevant history, and any recent lab values...',
        height=170,
        label_visibility='collapsed',
        key='patient_text',
    )

    st.markdown('<div class="section-label" style="margin-top:0.75rem;">Trial Eligibility Criteria</div>', unsafe_allow_html=True)
    criteria_input = st.text_area(
        label='Trial Eligibility Criteria',
        placeholder='List inclusion and exclusion criteria, one per line (use "- " bullet points)…',
        height=170,
        label_visibility='collapsed',
        key='criteria_text',
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        screen_clicked = st.button('🔍  Screen Patient', type='primary', use_container_width=True)
    with col_clear:
        if st.button('Clear', use_container_width=True):
            st.session_state.patient_text = ''
            st.session_state.criteria_text = ''
            st.rerun()

with right:
    result_slot = st.empty()

    if not screen_clicked:
        result_slot.markdown("""
        <div class="placeholder-card">
            <div class="placeholder-icon">🔬</div>
            <p class="placeholder-title">Awaiting Screening</p>
            <p class="placeholder-sub">Enter a patient narrative and trial criteria, then click Screen Patient.</p>
        </div>
        """, unsafe_allow_html=True)


# ── Run prediction ────────────────────────────────────────────────────────────
if screen_clicked:
    patient_val  = patient_input.strip()
    criteria_val = criteria_input.strip()

    if not patient_val or not criteria_val:
        with right:
            st.warning('Please fill in both the Patient Narrative and Trial Criteria fields.')
    else:
        with right:
            with st.spinner('Analysing eligibility…'):
                if model_ready:
                    model, tokenizer = _load_model()
                    if model is None:
                        result = _demo_predict(patient_val, criteria_val)
                    else:
                        from predict import predict as _predict
                        result = _predict(patient_val, criteria_val, model, tokenizer)
                else:
                    result = _demo_predict(patient_val, criteria_val)

        label     = result['label']
        reasoning = result['reasoning']
        conf      = result['confidence']
        ts        = datetime.now().strftime('%H:%M:%S')

        card_class = {
            'Eligible':     'result-eligible',
            'Not Eligible': 'result-not-eligible',
        }.get(label, 'result-unknown')

        verdict_class = {
            'Eligible':     'verdict-eligible',
            'Not Eligible': 'verdict-not-eligible',
        }.get(label, 'verdict-unknown')

        verdict_icon = {'Eligible': '✅', 'Not Eligible': '❌'}.get(label, '⚠️')
        conf_tag     = f'<span style="background:#e2e8f0;border-radius:4px;padding:2px 8px;font-size:0.75rem;color:#475569;margin-left:8px;">{conf}</span>'

        reasoning_block = (
            f'<div class="reasoning-label">Assessment</div>'
            f'<p class="reasoning-text">{reasoning}</p>'
            if reasoning else ''
        )

        with right:
            result_slot.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-verdict {verdict_class}">
                    {verdict_icon} {label}{conf_tag}
                </div>
                <div class="result-meta">Screened at {ts}{' · Demo mode' if conf == 'Demo' else ''}</div>
                {reasoning_block}
            </div>
            """, unsafe_allow_html=True)

        # Append to history
        st.session_state.history.insert(0, {
            'label':    label,
            'patient':  patient_val[:80],
            'time':     ts,
        })
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[:20]


# ── Screening history ─────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<hr style="border:0;border-top:1px solid #e2e8f0;margin:2rem 0 1rem;">', unsafe_allow_html=True)
    with st.expander(f'📋  Screening History  ({len(st.session_state.history)} records)', expanded=False):
        for item in st.session_state.history:
            lbl   = item['label']
            vcls  = {'Eligible': 'hist-verdict-e', 'Not Eligible': 'hist-verdict-ne'}.get(lbl, 'hist-verdict-u')
            icon  = {'Eligible': '✅', 'Not Eligible': '❌'}.get(lbl, '⚠️')
            st.markdown(f"""
            <div class="hist-item">
                <span class="{vcls}">{icon} {lbl}</span>
                <span class="hist-snippet">{item['patient']}</span>
                <span class="hist-time">{item['time']}</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button('Clear history', key='clear_hist'):
            st.session_state.history = []
            st.rerun()
