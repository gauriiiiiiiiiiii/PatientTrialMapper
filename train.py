"""
train.py — Run after data_prep.py.

    python train.py

Fine-tunes BASE_MODEL using QLoRA (4-bit quantization + LoRA adapters).
Saves adapter weights to models/lora_adapters/ (~50-100 MB, not the full model).

Requirements:
  - NVIDIA GPU with at least 6 GB VRAM (T4, RTX 3060, etc.)
  - bitsandbytes works on Linux/WSL2. On native Windows, prefer running on Colab.
  - For Mistral: set HF_TOKEN in config.py and accept terms at
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
"""

import sys
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

import config


def check_gpu():
    if not torch.cuda.is_available():
        print('ERROR: No GPU detected.')
        print('  - On Colab: Runtime → Change runtime type → T4 GPU')
        print('  - On Windows: use WSL2 or run on Colab')
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {name}  ({vram:.1f} GB VRAM)')
    if vram < 5:
        print('WARNING: Less than 5 GB VRAM — reduce BATCH_SIZE or MAX_SEQ_LEN in config.py')


def load_tokenizer() -> AutoTokenizer:
    kwargs = {'trust_remote_code': True}
    if config.HF_TOKEN:
        kwargs['token'] = config.HF_TOKEN
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def load_base_model() -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    kwargs = {
        'quantization_config': bnb_config,
        'device_map': 'auto',
        'trust_remote_code': True,
    }
    if config.HF_TOKEN:
        kwargs['token'] = config.HF_TOKEN
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **kwargs)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model


def apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    check_gpu()

    if not config.TRAIN_DIR.exists() or not config.VAL_DIR.exists():
        print('ERROR: data/ not found. Run python data_prep.py first.')
        sys.exit(1)

    print('\nLoading datasets...')
    train_dataset = load_from_disk(str(config.TRAIN_DIR))
    val_dataset   = load_from_disk(str(config.VAL_DIR))
    print(f'  Train: {len(train_dataset)}  Val: {len(val_dataset)}')

    print('\nLoading tokenizer...')
    tokenizer = load_tokenizer()

    print(f'\nLoading base model: {config.BASE_MODEL}')
    print('(This downloads ~14 GB on first run — be patient)')
    model = load_base_model()

    print('\nApplying QLoRA...')
    model = apply_lora(model)

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    import transformers, trl, re as _re
    def _ver(v): return tuple(int(x) for x in _re.split(r'[^0-9]+', v)[:2])
    eval_key        = 'eval_strategy'    if _ver(transformers.__version__) >= (4, 46) else 'evaluation_strategy'
    trainer_tok_key = 'processing_class' if _ver(trl.__version__)          >= (0, 9)  else 'tokenizer'

    training_args = TrainingArguments(
        output_dir=str(config.CHECKPOINTS_DIR),
        max_steps=config.MAX_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACC,
        gradient_checkpointing=True,
        optim='paged_adamw_32bit',
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        logging_steps=25,
        **{eval_key: 'steps'},
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        report_to='none',
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        **{trainer_tok_key: tokenizer},
        args=training_args,
        dataset_text_field='text',
        max_seq_length=config.MAX_SEQ_LEN,
        packing=False,
    )

    print('\nTraining started...')
    trainer.train()

    print(f'\nSaving adapters to {config.ADAPTER_DIR}')
    model.save_pretrained(str(config.ADAPTER_DIR))
    tokenizer.save_pretrained(str(config.ADAPTER_DIR))

    print('\ntrain.py done. Run: python evaluate.py')


if __name__ == '__main__':
    main()
