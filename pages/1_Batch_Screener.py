"""
pages/1_Batch_Screener.py — Batch CSV screening page.
"""

import io
from datetime import datetime

import pandas as pd
import streamlit as st

import config

st.set_page_config(
    page_title='Batch Screener · PatientTrialMapper',
    page_icon='🔬',
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
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 4px 24px rgba(37,99,235,0.25);
}
.ptm-header h1 { color:#fff; font-size:1.7rem; font-weight:800; margin:0; letter-spacing:-0.5px; }
.ptm-header p  { color:#bfdbfe; font-size:0.88rem; margin:0.2rem 0 0; }

.section-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b; margin-bottom: 0.4rem;
}

.stat-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.1rem; text-align: center;
}
.stat-value { font-size: 2rem; font-weight: 800; }
.stat-label { font-size: 0.78rem; color: #64748b; margin-top: 0.15rem; }
.stat-total    { color: #2563eb; }
.stat-eligible { color: #15803d; }
.stat-not      { color: #b91c1c; }
.stat-unknown  { color: #b45309; }

.info-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;
}
.info-box p { color: #1e40af; font-size: 0.875rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ptm-header">
    <h1>🔬 Batch Screener</h1>
    <p>Upload a CSV with patient profiles and trial criteria to screen multiple candidates at once.</p>
</div>
""", unsafe_allow_html=True)

# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_model():
    try:
        from predict import load_model
        return load_model()
    except (SystemExit, Exception):
        return None, None


model_ready = config.ADAPTER_DIR.exists()


def _demo_predict(patient: str, criteria: str) -> dict:
    from data_prep import SAMPLE_DATA
    pat_lower = patient.strip().lower()
    for ex in SAMPLE_DATA:
        if ex['patient'].strip().lower()[:60] in pat_lower or pat_lower[:60] in ex['patient'].strip().lower():
            return {'label': ex['label'], 'reasoning': ex.get('reasoning', ''), 'confidence': 'Demo'}
    return {'label': 'Unknown', 'reasoning': 'No matching example found.', 'confidence': 'N/A'}


# ── Instructions + template download ──────────────────────────────────────────
with st.expander('📄  How to use & CSV template', expanded=True):
    st.markdown("""
    <div class="info-box">
        <p>Your CSV must have two columns: <b>patient</b> and <b>criteria</b>.<br>
        Each row is one patient–trial pair. The screener will add <b>prediction</b>, <b>reasoning</b>, and <b>confidence</b> columns.</p>
    </div>
    """, unsafe_allow_html=True)

    template_rows = [
        {
            'patient':  '65-year-old male with type 2 diabetes. HbA1c 7.2%, on metformin 1000mg twice daily. No cardiovascular disease.',
            'criteria': '- Adults 18-75 with type 2 diabetes\n- HbA1c between 7.0% and 10.0%\n- No prior cardiovascular events\n- Not on insulin therapy',
        },
        {
            'patient':  '52-year-old female with stage II breast cancer, mastectomy 6 months ago. No autoimmune disease.',
            'criteria': '- Women 40-65 with stage I-III breast cancer\n- Post-surgical completion\n- ECOG performance status 0-2\n- No concurrent diabetes or autoimmune disease',
        },
    ]
    template_df  = pd.DataFrame(template_rows)
    template_csv = template_df.to_csv(index=False)

    st.download_button(
        '⬇  Download CSV Template',
        data=template_csv,
        file_name='batch_template.csv',
        mime='text/csv',
    )

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload CSV File</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label='Upload CSV',
    type=['csv'],
    label_visibility='collapsed',
    help='CSV with columns: patient, criteria',
)

if uploaded is None:
    st.info('Upload a CSV file above to begin batch screening.')
    st.stop()

# ── Parse & validate ───────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f'Could not read CSV: {e}')
    st.stop()

missing = [c for c in ('patient', 'criteria') if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}. The CSV must have 'patient' and 'criteria' columns.")
    st.stop()

df = df.dropna(subset=['patient', 'criteria']).reset_index(drop=True)
if df.empty:
    st.warning('The CSV has no valid rows after removing blanks.')
    st.stop()

st.success(f'Loaded **{len(df)} rows**. Ready to screen.')

col_run, col_gap = st.columns([2, 8])
with col_run:
    run_batch = st.button('🚀  Run Batch Screening', type='primary', use_container_width=True)

if not run_batch:
    st.stop()

# ── Run predictions ────────────────────────────────────────────────────────────
progress_bar = st.progress(0, text='Starting…')
status_text  = st.empty()

predictions, reasonings, confidences = [], [], []

model, tokenizer = (_load_model() if model_ready else (None, None))

for i, row in df.iterrows():
    pct  = int((i + 1) / len(df) * 100)
    progress_bar.progress(pct, text=f'Screening row {i+1} of {len(df)}…')
    status_text.markdown(f'`{row["patient"][:70]}…`')

    if model_ready and model is not None:
        from predict import predict as _predict
        result = _predict(str(row['patient']), str(row['criteria']), model, tokenizer)
    else:
        result = _demo_predict(str(row['patient']), str(row['criteria']))

    predictions.append(result['label'])
    reasonings.append(result['reasoning'])
    confidences.append(result['confidence'])

progress_bar.empty()
status_text.empty()

results_df = df.copy()
results_df['prediction'] = predictions
results_df['reasoning']  = reasonings
results_df['confidence'] = confidences
results_df['screened_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ── Summary stats ──────────────────────────────────────────────────────────────
n_total   = len(results_df)
n_elig    = (results_df['prediction'] == 'Eligible').sum()
n_not     = (results_df['prediction'] == 'Not Eligible').sum()
n_unknown = (results_df['prediction'] == 'Unknown').sum()

st.markdown('<hr style="border:0;border-top:1px solid #e2e8f0;margin:1.5rem 0 1rem;">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Results Summary</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, value, label, css in [
    (c1, n_total,   'Total Screened',  'stat-total'),
    (c2, n_elig,    'Eligible',        'stat-eligible'),
    (c3, n_not,     'Not Eligible',    'stat-not'),
    (c4, n_unknown, 'Uncertain',       'stat-unknown'),
]:
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-value {css}">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Results table ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem;">Detailed Results</div>', unsafe_allow_html=True)

def _color_row(row):
    color = {'Eligible': '#f0fdf4', 'Not Eligible': '#fef2f2'}.get(row['prediction'], '#fffbeb')
    return [f'background-color: {color}'] * len(row)

display_df = results_df[['patient', 'prediction', 'confidence', 'reasoning']].copy()
display_df['patient'] = display_df['patient'].str[:80] + '…'
st.dataframe(
    display_df.style.apply(_color_row, axis=1),
    use_container_width=True,
    height=min(400, 50 + 35 * len(display_df)),
)

# ── Download ───────────────────────────────────────────────────────────────────
csv_out = results_df.to_csv(index=False)
st.download_button(
    '⬇  Download Results CSV',
    data=csv_out,
    file_name=f'screening_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    mime='text/csv',
    type='primary',
)
