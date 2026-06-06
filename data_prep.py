"""
data_prep.py — Run this first.

    python data_prep.py

Builds a dataset of patient–trial pairs, formats them as instruction prompts
with reasoning, splits into train/val/test, and saves to disk under data/.

For extra data: set OPENAI_API_KEY in config.py and the script will call
GPT-4o to generate synthetic examples.

For TREC 2022: place topics2022.xml, qrels2022.txt, and extracted trial XMLs
under data/raw/ (see TREC 2022 section below).
"""

import json
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset

import config

# ── Sample data ───────────────────────────────────────────────────────────────
# 20 hand-crafted patient–trial pairs. Balanced: 10 Eligible, 10 Not Eligible.
# Each entry includes a 'reasoning' field used in the training target.

SAMPLE_DATA = [
    {
        'patient': '65-year-old male with type 2 diabetes diagnosed 8 years ago. HbA1c 7.2%, on metformin 1000mg twice daily. No cardiovascular disease.',
        'criteria': '- Adults 18-75 with type 2 diabetes\n- HbA1c between 7.0% and 10.0%\n- No prior cardiovascular events\n- Not on insulin therapy',
        'label': 'Eligible',
        'reasoning': 'Patient meets all four criteria: age 65 is within 18–75, HbA1c 7.2% falls within 7.0–10.0%, no cardiovascular events are documented, and current therapy is metformin — not insulin.',
    },
    {
        'patient': '58-year-old female with type 2 diabetes on insulin glargine for 3 years. HbA1c 8.5%. History of myocardial infarction 2 years ago.',
        'criteria': '- Adults 18-75 with type 2 diabetes\n- HbA1c between 7.0% and 10.0%\n- No prior cardiovascular events\n- Not on insulin therapy',
        'label': 'Not Eligible',
        'reasoning': 'Patient fails two criteria: she is currently on insulin glargine, which is explicitly excluded, and has a documented myocardial infarction 2 years ago, which counts as a prior cardiovascular event.',
    },
    {
        'patient': '52-year-old female with stage II breast cancer, mastectomy 6 months ago. Currently on hormone therapy. No diabetes or autoimmune disease.',
        'criteria': '- Women 40-65 with stage I-III breast cancer\n- Post-surgical completion\n- ECOG performance status 0-2\n- No concurrent diabetes or autoimmune disease',
        'label': 'Eligible',
        'reasoning': 'All criteria are satisfied: age 52 is within 40–65, stage II is within the I–III range, mastectomy is complete, no diabetes or autoimmune disease is present, and functional status is consistent with ECOG 0–1.',
    },
    {
        'patient': '47-year-old female with stage IV breast cancer and active liver metastases. On third-line chemotherapy.',
        'criteria': '- Women 40-65 with stage I-III breast cancer\n- Post-surgical completion\n- ECOG performance status 0-2\n- No concurrent diabetes or autoimmune disease',
        'label': 'Not Eligible',
        'reasoning': 'Patient fails the stage criterion: stage IV disease exceeds the maximum allowed stage III. Active liver metastases also indicate distant spread inconsistent with the post-surgical, curative intent of this trial.',
    },
    {
        'patient': '63-year-old female with heart failure (EF 30%). Creatinine 1.4 mg/dL, eGFR 45. On carvedilol and sacubitril/valsartan for 6 months.',
        'criteria': '- Adults with heart failure and EF < 40%\n- On guideline-directed therapy\n- eGFR > 30 mL/min\n- Creatinine < 2.0 mg/dL',
        'label': 'Eligible',
        'reasoning': 'All four criteria are met: EF 30% is below the 40% threshold, the patient is on guideline-directed therapy (carvedilol and sacubitril/valsartan), eGFR 45 exceeds 30, and creatinine 1.4 mg/dL is below 2.0.',
    },
    {
        'patient': '70-year-old male with heart failure (EF 35%) and chronic kidney disease stage 3. Creatinine 2.1 mg/dL. On furosemide and ACE inhibitor.',
        'criteria': '- Adults with heart failure and EF < 40%\n- On guideline-directed therapy\n- eGFR > 30 mL/min\n- Creatinine < 2.0 mg/dL',
        'label': 'Not Eligible',
        'reasoning': "Patient's creatinine of 2.1 mg/dL exceeds the 2.0 mg/dL ceiling. This single violation makes the patient ineligible despite meeting the EF, therapy, and eGFR criteria.",
    },
    {
        'patient': '45-year-old non-smoker with moderate COPD. FEV1/FVC ratio 0.65. On inhaled bronchodilators. No exacerbations in past 6 months.',
        'criteria': '- COPD patients aged 30-75\n- FEV1/FVC ratio < 0.70 post-bronchodilator\n- Non-smoker or ex-smoker (quit > 1 year)\n- No hospitalizations for COPD in past 12 months',
        'label': 'Eligible',
        'reasoning': 'All criteria are met: age 45 is within 30–75, post-bronchodilator FEV1/FVC 0.65 confirms obstruction below 0.70, the patient is a non-smoker, and no COPD hospitalisations occurred in the past 12 months.',
    },
    {
        'patient': '55-year-old current smoker with severe COPD. FEV1/FVC 0.55. Two COPD hospitalizations in the last 8 months.',
        'criteria': '- COPD patients aged 30-75\n- FEV1/FVC ratio < 0.70 post-bronchodilator\n- Non-smoker or ex-smoker (quit > 1 year)\n- No hospitalizations for COPD in past 12 months',
        'label': 'Not Eligible',
        'reasoning': 'Patient fails two criteria: active smoking status does not meet the non-smoker or cessation > 1 year requirement, and two COPD hospitalisations within 8 months violates the zero-in-12-months limit.',
    },
    {
        'patient': '34-year-old male with moderate depression, PHQ-9 score 14. On escitalopram 10mg for 8 weeks. No prior suicide attempts.',
        'criteria': '- Adults 18-65 with moderate-to-severe depression\n- No suicide attempt in the past 5 years\n- On stable SSRI for at least 6 weeks',
        'label': 'Eligible',
        'reasoning': 'All three criteria are satisfied: age 34 is within 18–65, PHQ-9 of 14 confirms moderate-to-severe depression, no suicide attempts are documented, and escitalopram has been stable for 8 weeks — exceeding the 6-week minimum.',
    },
    {
        'patient': '58-year-old female with moderate depression, on sertraline 50mg for 3 months. Previous suicide attempt 2 years ago.',
        'criteria': '- Adults 18-65 with moderate-to-severe depression\n- No suicide attempt in the past 5 years\n- On stable SSRI for at least 6 weeks',
        'label': 'Not Eligible',
        'reasoning': "Patient's suicide attempt 2 years ago falls within the 5-year exclusion window, making her ineligible regardless of her depression severity or medication compliance.",
    },
    {
        'patient': '42-year-old male with rheumatoid arthritis on methotrexate for 2 years. DAS28 score 4.2. No active infections. No prior biologic use.',
        'criteria': '- RA patients 18-65 with moderate-to-high disease activity (DAS28 > 3.2)\n- On conventional DMARDs with inadequate response\n- No active infections\n- No prior biologic therapy',
        'label': 'Eligible',
        'reasoning': 'All criteria are met: age 42 is within 18–65, DAS28 4.2 exceeds 3.2 indicating moderate-to-high activity, patient is on conventional DMARD (methotrexate) with inadequate response, no active infections, and no prior biologic use.',
    },
    {
        'patient': '38-year-old female with RA, currently on adalimumab (biologic) for 18 months. DAS28 3.8. No infections.',
        'criteria': '- RA patients 18-65 with moderate-to-high disease activity (DAS28 > 3.2)\n- On conventional DMARDs with inadequate response\n- No active infections\n- No prior biologic therapy',
        'label': 'Not Eligible',
        'reasoning': 'Patient is currently on adalimumab, a biologic therapy — a category explicitly excluded by the last criterion. Prior or current biologic use disqualifies her from this trial.',
    },
    {
        'patient': "61-year-old male with Parkinson's disease, Hoehn & Yahr stage 2. On stable levodopa for 2 years. MoCA score 27. No DBS.",
        'criteria': "- Parkinson's patients 50-80, Hoehn & Yahr 1-3\n- On stable dopaminergic therapy\n- MoCA score > 24\n- No DBS implantation",
        'label': 'Eligible',
        'reasoning': 'All four criteria are satisfied: age 61 is within 50–80, Hoehn & Yahr stage 2 is within 1–3, levodopa therapy has been stable for 2 years, MoCA 27 exceeds 24, and no DBS implantation has been performed.',
    },
    {
        'patient': "67-year-old female with Parkinson's, Hoehn & Yahr stage 2. On levodopa/carbidopa. MoCA score 22 (mild cognitive impairment).",
        'criteria': "- Parkinson's patients 50-80, Hoehn & Yahr 1-3\n- On stable dopaminergic therapy\n- MoCA score > 24\n- No DBS implantation",
        'label': 'Not Eligible',
        'reasoning': "Patient's MoCA score of 22 falls below the required minimum of 24, indicating mild cognitive impairment that excludes her from this trial.",
    },
    {
        'patient': '33-year-old male with first-episode schizophrenia on risperidone 4mg daily. No substance use. No prior psychiatric hospitalizations.',
        'criteria': '- First-episode schizophrenia, aged 18-45\n- On antipsychotic monotherapy\n- No substance use disorder in past 12 months\n- No prior psychiatric hospitalizations',
        'label': 'Eligible',
        'reasoning': 'All four criteria are satisfied: first-episode schizophrenia in a 33-year-old within 18–45, on antipsychotic monotherapy (risperidone only), no substance use disorder in the past 12 months, and no prior psychiatric hospitalisations.',
    },
    {
        'patient': '29-year-old female with schizophrenia on clozapine plus aripiprazole. Cannabis use disorder in remission 8 months ago.',
        'criteria': '- First-episode schizophrenia, aged 18-45\n- On antipsychotic monotherapy\n- No substance use disorder in past 12 months\n- No prior psychiatric hospitalizations',
        'label': 'Not Eligible',
        'reasoning': 'Patient fails two criteria: she is on dual antipsychotic therapy (clozapine plus aripiprazole), which is not monotherapy, and cannabis use disorder remission of only 8 months does not clear the required 12-month window.',
    },
    {
        'patient': '55-year-old male with hypertension, BP 158/95 on amlodipine 5mg. No renal disease, no diabetes, eGFR 82.',
        'criteria': '- Adults 30-70 with uncontrolled hypertension (SBP > 140)\n- On at least one antihypertensive\n- eGFR > 60 mL/min\n- No diabetes or secondary hypertension',
        'label': 'Eligible',
        'reasoning': 'All criteria are satisfied: age 55 is within 30–70, SBP 158 mmHg confirms uncontrolled hypertension above 140, patient is on amlodipine (meets ≥ 1 antihypertensive requirement), eGFR 82 exceeds 60, and no diabetes or secondary cause is present.',
    },
    {
        'patient': '48-year-old female with hypertension secondary to renal artery stenosis. BP 170/100 on three medications.',
        'criteria': '- Adults 30-70 with uncontrolled hypertension (SBP > 140)\n- On at least one antihypertensive\n- eGFR > 60 mL/min\n- No diabetes or secondary hypertension',
        'label': 'Not Eligible',
        'reasoning': "Hypertension is secondary to renal artery stenosis, a category explicitly excluded by the last criterion. The trial targets essential (primary) hypertension, and secondary causes have fundamentally different pathophysiology and treatment response.",
    },
    {
        'patient': "72-year-old male with mild Alzheimer's disease (MMSE 22). On donepezil 10mg for 6 months. Lives at home with caregiver support.",
        'criteria': "- Alzheimer's patients 60-85, MMSE 18-26\n- On stable cholinesterase inhibitor > 3 months\n- Has a reliable caregiver\n- No other significant neurological conditions",
        'label': 'Eligible',
        'reasoning': 'All four criteria are met: age 72 is within 60–85, MMSE 22 falls within 18–26, donepezil has been stable for 6 months (exceeds the 3-month minimum), a reliable caregiver is present, and no other significant neurological conditions are documented.',
    },
    {
        'patient': "68-year-old female with moderate Alzheimer's disease (MMSE 14) and concurrent vascular dementia. On memantine and donepezil.",
        'criteria': "- Alzheimer's patients 60-85, MMSE 18-26\n- On stable cholinesterase inhibitor > 3 months\n- Has a reliable caregiver\n- No other significant neurological conditions",
        'label': 'Not Eligible',
        'reasoning': "Patient fails two criteria: MMSE of 14 falls below the minimum of 18, indicating moderate rather than mild disease, and concurrent vascular dementia constitutes a second significant neurological condition.",
    },
]


# ── TREC 2022 Clinical Trials loader (optional) ───────────────────────────────
# Download from NIST: https://trec.nist.gov/data/clinical2022.html
# Place files at:
#   data/raw/topics2022.xml
#   data/raw/qrels2022.txt
#   data/raw/trials/        ← extracted NCTxxxxxxxx.xml files

_TREC_TOPICS  = config.DATA_DIR / 'raw' / 'topics2022.xml'
_TREC_QRELS   = config.DATA_DIR / 'raw' / 'qrels2022.txt'
_TREC_TRIALS  = config.DATA_DIR / 'raw' / 'trials'


def load_trec_data() -> list[dict]:
    """Loads TREC 2022 pairs. Returns [] if raw files are absent."""
    if not (_TREC_TOPICS.exists() and _TREC_QRELS.exists() and _TREC_TRIALS.is_dir()):
        return []

    print('Loading TREC 2022 data...')
    root   = ET.parse(_TREC_TOPICS).getroot()
    topics = {t.get('number'): (t.text or '').strip() for t in root.findall('topic')}

    rows = []
    with open(_TREC_QRELS) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic_id, _, trial_id, rel = parts[0], parts[1], parts[2], parts[3]
            if rel not in ('1', '2'):
                continue
            patient = topics.get(topic_id, '')
            if not patient:
                continue
            trial_xml = _TREC_TRIALS / f'{trial_id}.xml'
            if not trial_xml.exists():
                continue
            try:
                tb = ET.parse(trial_xml).getroot().find('.//eligibility/criteria/textblock')
                criteria = (tb.text or '').strip()[:800] if tb is not None else ''
            except ET.ParseError:
                continue
            if criteria:
                rows.append({
                    'patient': patient,
                    'criteria': criteria,
                    'label': 'Eligible' if rel == '2' else 'Not Eligible',
                    'reasoning': '',
                })
    print(f'TREC 2022 pairs loaded: {len(rows)}')
    return rows


# ── Synthetic data generation (optional) ──────────────────────────────────────

_SYNTH_SYSTEM = (
    'You are a medical data generator. Create realistic clinical trial eligibility scenarios. '
    'Return a JSON object with exactly these keys: patient, criteria, label, reasoning.\n'
    '- patient: 2-3 sentences of patient medical history\n'
    '- criteria: 3-5 bullet points separated by \\n characters\n'
    '- label: exactly "Eligible" or "Not Eligible"\n'
    '- reasoning: one sentence explaining which criterion determines the decision'
)

_CONDITIONS = [
    'Type 2 diabetes', 'Hypertension', 'Heart failure', 'COPD',
    'Breast cancer', 'Lung cancer', 'Depression', 'Schizophrenia',
    'Rheumatoid arthritis', 'Multiple sclerosis',
]


def generate_synthetic(n_per_condition: int = 10) -> list[dict]:
    try:
        import openai
    except ImportError:
        print('openai not installed — skipping synthetic generation.')
        return []

    if not config.OPENAI_API_KEY:
        print('OPENAI_API_KEY not set in config.py — skipping synthetic generation.')
        return []

    client  = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    results = []

    for condition in _CONDITIONS:
        print(f'  Generating {n_per_condition} examples for: {condition}')
        for _ in range(n_per_condition):
            try:
                resp = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[
                        {'role': 'system', 'content': _SYNTH_SYSTEM},
                        {'role': 'user',   'content': f'Generate a scenario for: {condition}'},
                    ],
                    response_format={'type': 'json_object'},
                    temperature=0.9,
                )
                parsed = json.loads(resp.choices[0].message.content)
                # Validate required keys exist
                if all(k in parsed for k in ('patient', 'criteria', 'label')):
                    parsed.setdefault('reasoning', '')
                    results.append(parsed)
            except json.JSONDecodeError as e:
                print(f'    JSON parse error: {e}')
            except Exception as e:
                print(f'    Error: {e}')

    return results


# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_prompt(patient: str, criteria: str) -> str:
    return '\n'.join([
        '### Patient:',
        patient,
        '',
        '### Trial Criteria:',
        criteria,
        '',
        '### Is the patient eligible?',
    ])


def format_for_training(example: dict) -> dict:
    prompt    = format_prompt(example['patient'], example['criteria'])
    label     = example['label']
    reasoning = example.get('reasoning', '').strip()
    # Full training target includes reasoning so the model learns to explain
    response  = f'{label}. {reasoning}' if reasoning else label
    return {
        'prompt':   prompt,
        'response': label,     # label-only for evaluation metrics
        'text':     prompt + '\n' + response,
    }


# ── EDA ───────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    print('\n── Label distribution ──')
    print(df['response'].value_counts().to_string())

    print('\n── Prompt length stats ──')
    lengths = df['prompt'].str.len()
    print(f'  min={lengths.min()}  max={lengths.max()}  mean={lengths.mean():.0f}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(lengths, bins=15, color='steelblue', edgecolor='white')
    axes[0].set_title('Prompt Length Distribution')
    axes[0].set_xlabel('Characters')
    axes[0].set_ylabel('Count')

    counts = df['response'].value_counts()
    axes[1].bar(counts.index, counts.values, color=['steelblue', 'coral'], edgecolor='white')
    axes[1].set_title('Label Distribution')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    out = config.DATA_DIR / 'eda.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nEDA plot saved → {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config.DATA_DIR.mkdir(exist_ok=True)

    data = list(SAMPLE_DATA)
    print(f'Built-in examples: {len(data)}')

    trec = load_trec_data()
    if trec:
        data.extend(trec)
        print(f'After TREC 2022: {len(data)} total examples')

    print('\nGenerating synthetic data (skipped if no API key)...')
    synthetic = generate_synthetic(n_per_condition=10)
    if synthetic:
        data.extend(synthetic)
        print(f'After augmentation: {len(data)} total examples')

    formatted = [format_for_training(ex) for ex in data]
    df = pd.DataFrame(formatted)

    run_eda(df)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['response']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['response']
    )

    print(f'\nSplit → train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}')

    Dataset.from_pandas(train_df.reset_index(drop=True)).save_to_disk(str(config.TRAIN_DIR))
    Dataset.from_pandas(val_df.reset_index(drop=True)).save_to_disk(str(config.VAL_DIR))
    test_df.to_json(config.TEST_JSON, orient='records', indent=2)

    print('\nSaved:')
    print(f'  {config.TRAIN_DIR}')
    print(f'  {config.VAL_DIR}')
    print(f'  {config.TEST_JSON}')
    print('\ndata_prep.py done. Run: python train.py')


if __name__ == '__main__':
    main()
