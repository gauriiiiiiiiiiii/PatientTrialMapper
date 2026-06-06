"""
push_to_hub.py — Upload fine-tuned LoRA adapters to Hugging Face Hub.

    python push_to_hub.py --repo your-username/patient-trial-mapper

Requirements:
  - pip install huggingface_hub
  - huggingface-cli login   (or set HF_TOKEN in config.py)
  - models/lora_adapters/ must exist (run train.py first)
"""

import argparse
import io
import sys

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True,
                        help='HF Hub repo id, e.g. your-username/patient-trial-mapper')
    parser.add_argument('--private', action='store_true',
                        help='Create a private repo')
    args = parser.parse_args()

    if not config.ADAPTER_DIR.exists():
        print('ERROR: models/lora_adapters/ not found. Run python train.py first.')
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        print(f'Missing dependency: {e}')
        sys.exit(1)

    api = HfApi()
    api.create_repo(repo_id=args.repo, private=args.private, exist_ok=True,
                    token=config.HF_TOKEN or None)

    print(f'Pushing adapters to https://huggingface.co/{args.repo} ...')
    api.upload_folder(
        folder_path=str(config.ADAPTER_DIR),
        repo_id=args.repo,
        token=config.HF_TOKEN or None,
    )

    # Write a minimal model card
    card = f"""---
language: en
tags:
  - medical
  - clinical-trials
  - lora
  - mistral
  - qlora
base_model: {config.BASE_MODEL}
---

# PatientTrialMapper

Fine-tuned **{config.BASE_MODEL}** with QLoRA for automated clinical trial patient eligibility screening.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{config.BASE_MODEL}")
model = PeftModel.from_pretrained(base, "{args.repo}")
tokenizer = AutoTokenizer.from_pretrained("{args.repo}")

prompt = \"\"\"### Patient:
65-year-old male with type 2 diabetes. HbA1c 7.2%, on metformin.

### Trial Criteria:
- Adults 18-75 with type 2 diabetes
- HbA1c between 7.0% and 10.0%
- Not on insulin therapy

### Is the patient eligible?
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True).split("eligible?")[-1].strip())
```

## Training
- Dataset: TREC 2022 Clinical Trials + hand-crafted examples
- Technique: QLoRA (4-bit NF4 + LoRA r=16)
- Steps: {config.MAX_STEPS}
"""

    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo='README.md',
        repo_id=args.repo,
        token=config.HF_TOKEN or None,
    )
    print(f'Done. Model card written.')
    print(f'View at: https://huggingface.co/{args.repo}')


if __name__ == '__main__':
    main()
