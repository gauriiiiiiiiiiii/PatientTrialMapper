"""
predict.py — Shared inference utilities used by evaluate.py and app.py.

Standalone usage:
    python predict.py \
        --patient "65-year-old male with type 2 diabetes, HbA1c 7.2%..." \
        --criteria "- Adults 18-75 with type 2 diabetes\n- HbA1c 7-10%..."
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

import config
from data_prep import format_prompt


def load_model():
    """Load fine-tuned model (base + LoRA adapters) in 4-bit quantisation."""
    if not config.ADAPTER_DIR.exists():
        print('ERROR: No trained adapters found. Run python train.py first.')
        sys.exit(1)

    tokenizer_path = config.ADAPTER_DIR / 'tokenizer_config.json'
    if not tokenizer_path.exists():
        print('ERROR: Tokenizer not found in adapter directory. Re-run train.py.')
        sys.exit(1)

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

    base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, **kwargs)
    model = PeftModel.from_pretrained(base, str(config.ADAPTER_DIR))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(config.ADAPTER_DIR))
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def predict(patient: str, criteria: str, model, tokenizer) -> dict:
    """
    Returns {'label': str, 'reasoning': str, 'confidence': str}.

    label      — 'Eligible', 'Not Eligible', or 'Unknown'
    reasoning  — explanation text from the model
    confidence — 'High' if label is clear, 'Low' if ambiguous
    'Unknown' means ambiguous output — route to human review.
    """
    prompt = format_prompt(patient, criteria) + '\n'
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output.split('### Is the patient eligible?\n')[-1].strip()
    answer_lower = answer.lower()

    if 'not eligible' in answer_lower:
        label = 'Not Eligible'
    elif 'eligible' in answer_lower:
        label = 'Eligible'
    else:
        label = 'Unknown'

    # Extract the reasoning sentence(s) that follow the verdict
    reasoning = ''
    for prefix in ('Not Eligible. ', 'not eligible. ', 'Eligible. ', 'eligible. ',
                   'Not Eligible.\n', 'not eligible.\n', 'Eligible.\n', 'eligible.\n'):
        if answer.lower().startswith(prefix.lower()):
            reasoning = answer[len(prefix):].strip()
            break
    if not reasoning:
        parts = answer.split('. ', 1)
        if len(parts) > 1:
            reasoning = parts[1].strip()

    return {
        'label': label,
        'reasoning': reasoning,
        'confidence': 'High' if label != 'Unknown' else 'Low',
    }


def main():
    parser = argparse.ArgumentParser(description='Predict clinical trial eligibility.')
    parser.add_argument('--patient', required=True, help='Patient narrative text')
    parser.add_argument('--criteria', required=True, help='Trial eligibility criteria text')
    args = parser.parse_args()

    print('Loading model...')
    model, tokenizer = load_model()

    result = predict(args.patient, args.criteria, model, tokenizer)
    print(f"\nLabel      : {result['label']}")
    print(f"Confidence : {result['confidence']}")
    if result['reasoning']:
        print(f"Reasoning  : {result['reasoning']}")


if __name__ == '__main__':
    main()
