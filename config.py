from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_MODEL = 'mistralai/Mistral-7B-Instruct-v0.3'
# Lighter option (no HF token needed, fits in less VRAM):
# BASE_MODEL = 'microsoft/Phi-3-mini-4k-instruct'

# HuggingFace token — required to download Mistral (gated model).
# Get yours at https://huggingface.co/settings/tokens
# Then accept terms at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
HF_TOKEN = ''

# OpenAI key — only needed in data_prep.py for synthetic data generation.
OPENAI_API_KEY = ''

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path('data')
TRAIN_DIR       = DATA_DIR / 'train'
VAL_DIR         = DATA_DIR / 'val'
TEST_JSON       = DATA_DIR / 'test_samples.json'

MODELS_DIR      = Path('models')
CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
ADAPTER_DIR     = MODELS_DIR / 'lora_adapters'

# ── Training ──────────────────────────────────────────────────────────────────
MAX_SEQ_LEN   = 512
MAX_STEPS     = 500   # swap for num_train_epochs=3 once you have 200+ examples
BATCH_SIZE    = 2
GRAD_ACC      = 4     # effective batch size = BATCH_SIZE × GRAD_ACC = 8
LEARNING_RATE = 2e-4

# ── QLoRA / LoRA ──────────────────────────────────────────────────────────────
LORA_R       = 16
LORA_ALPHA   = 32    # typically 2 × LORA_R
LORA_DROPOUT = 0.05
LORA_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
]
