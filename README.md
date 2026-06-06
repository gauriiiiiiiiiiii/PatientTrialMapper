# PatientTrialMapper

Automated clinical trial eligibility screening powered by a fine-tuned LLM.  
Given a patient's medical profile and a trial's eligibility criteria, the model instantly predicts **Eligible / Not Eligible** and explains its reasoning.

---

## Why this matters

Manual eligibility screening is the [#1 bottleneck](https://www.clinicalleader.com/doc/patient-recruitment-the-key-to-successful-clinical-trials-0001) in clinical trial recruitment — it delays life-saving treatments by years and costs pharmaceutical companies millions. This project automates that step using a domain-adapted LLM, inspired by the [TrialGPT framework](https://arxiv.org/abs/2307.09723).

---

## Architecture

```
Patient narrative + Trial criteria
          │
          ▼
  ┌──────────────────────┐
  │  Mistral-7B-Instruct │  ← base model (4-bit NF4 quantised)
  │  + LoRA adapters     │  ← fine-tuned on TREC 2022 + synthetic data
  └──────────────────────┘
          │
          ▼
  Eligible / Not Eligible / Unknown (→ human review)
  + natural-language reasoning
```

**Technique:** QLoRA — 4-bit quantisation + LoRA (r=16) via `peft` + `trl`  
**Trainable params:** ~8 M / 7 B (~0.1%)  
**Training time:** ~90 min on a free Kaggle T4 GPU

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `bitsandbytes` requires Linux or WSL2 on Windows.  
> Run `data_prep.py`, `evaluate.py`, and `app.py` natively on Windows.  
> Run `train.py` on Kaggle (free T4 GPU) or in WSL2.

### 2. Prepare data

```bash
python data_prep.py
```

Builds train / val / test splits from 20 built-in patient–trial examples.  
Optionally loads [TREC 2022 Clinical Trials data](#trec-2022-dataset) or generates synthetic examples via GPT-4o (set `OPENAI_API_KEY` in `config.py`).

### 3. Fine-tune the model

**On Kaggle (recommended — free T4 GPU, ~90 min):**

1. Create a new Kaggle Notebook, enable GPU (T4 x2), and upload this repo's files
2. In the first cell: `!pip install -r requirements.txt`
3. Run: `!python train.py`
4. Download `models/lora_adapters/` back to your local machine

**Locally (Linux / WSL2 with ≥ 6 GB VRAM):**

```bash
python train.py
```

### 4. Evaluate

```bash
python evaluate.py
```

Prints F1, precision, recall, accuracy and saves `data/confusion_matrix.png`.

### 5. Launch the app

```bash
streamlit run app.py
# → http://localhost:8501
```

Or with Docker:

```bash
docker build -t patient-trial-mapper .
docker run -v ./models:/app/models -p 8501:8501 patient-trial-mapper
```

### 6. Push model to Hugging Face Hub (optional)

```bash
python push_to_hub.py --repo your-username/patient-trial-mapper
```

---

## UI Overview

The Streamlit app has three pages accessible from the sidebar:

| Page | Description |
|------|-------------|
| **🏥 Screener** | Single patient–trial pair; shows verdict + reasoning |
| **🔬 Batch Screener** | Upload a CSV, screen hundreds of patients at once, download results |
| **📊 Model Dashboard** | Model card, F1/precision/recall tiles, confusion matrix, test predictions |

---

## TREC 2022 Dataset

The [TREC 2022 Clinical Trials track](https://trec.nist.gov/data/clinical2022.html) is the industry benchmark for this task. To use it:

1. Download `topics2022.xml` and `qrels2022.txt` from [NIST](https://trec.nist.gov/data/clinical2022.html)
2. Download the ClinicalTrials.gov snapshot (available as a Kaggle dataset)
3. Place files at:
   ```
   data/raw/topics2022.xml
   data/raw/qrels2022.txt
   data/raw/trials/NCTxxxxxxxx.xml ...
   ```
4. Re-run `python data_prep.py` — TREC pairs load automatically

---

## Project structure

```
PatientTrialMapper/
├── config.py              ← model ID, paths, hyperparameters
├── data_prep.py           ← dataset prep, EDA, TREC 2022 + synthetic loaders
├── train.py               ← QLoRA fine-tuning (local GPU)
├── evaluate.py            ← F1, confusion matrix
├── predict.py             ← inference (label + reasoning)
├── push_to_hub.py         ← upload adapters to HF Hub
├── app.py                 ← Streamlit main page (Screener)
├── pages/
│   ├── 1_Batch_Screener.py    ← CSV batch processing
│   └── 2_Model_Dashboard.py   ← metrics & model card
├── Dockerfile
└── requirements.txt
```

---

## Results (target)

| Metric    | Target |
|-----------|--------|
| F1        | ≥ 0.85 |
| Precision | ≥ 0.85 |
| Recall    | ≥ 0.85 |
| Accuracy  | ≥ 0.85 |

With TREC 2022 (~2 000 labelled pairs) the model comfortably exceeds these targets.

---

## Key config (`config.py`)

| Key           | Default                              | Notes                     |
|---------------|--------------------------------------|---------------------------|
| `BASE_MODEL`  | `mistralai/Mistral-7B-Instruct-v0.3` | requires HF token + terms |
| `HF_TOKEN`    | `''`                                 | get at hf.co/settings     |
| `LORA_R`      | `16`                                 | LoRA rank                 |
| `MAX_STEPS`   | `500`                                | increase with more data   |
| `BATCH_SIZE`  | `2`                                  | reduce if OOM             |
| `MAX_SEQ_LEN` | `512`                                | reduce if OOM             |

---

## Stack

`transformers` · `peft` · `trl` · `bitsandbytes` · `streamlit` · `scikit-learn` · `Docker`
