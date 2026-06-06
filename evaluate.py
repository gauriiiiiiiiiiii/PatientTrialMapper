"""
evaluate.py — Run after train.py.

    python evaluate.py

Loads the fine-tuned model, runs predictions on the held-out test set,
and prints F1, precision, recall, accuracy + a confusion matrix saved to data/.
"""

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
from predict import load_model, predict


def run_evaluation(model, tokenizer, test_df: pd.DataFrame) -> pd.DataFrame:
    total = len(test_df)
    labels, reasonings = [], []

    print(f'\nRunning predictions on {total} test examples...')
    for i, row in test_df.iterrows():
        parts    = row['prompt'].split('### Trial Criteria:\n')
        patient  = parts[0].replace('### Patient:\n', '').strip()
        criteria = parts[1].split('\n\n### Is the patient eligible?')[0].strip()

        result = predict(patient, criteria, model, tokenizer)
        labels.append(result['label'])
        reasonings.append(result['reasoning'])
        print(f'  [{i+1}/{total}]  True: {row["response"]:<15}  Pred: {result["label"]}')

    out = test_df.copy()
    out['predicted'] = labels
    out['reasoning'] = reasonings
    return out


def print_metrics(test_df: pd.DataFrame) -> dict:
    true_labels = test_df['response'].tolist()
    pred_labels = test_df['predicted'].tolist()

    unknown_count = pred_labels.count('Unknown')

    # Build aligned pairs, keeping indices in sync
    pairs = [(t, p) for t, p in zip(true_labels, pred_labels) if p != 'Unknown']

    if not pairs:
        print('ERROR: All predictions returned Unknown — check the model output format.')
        sys.exit(1)

    true_valid, pred_valid = zip(*pairs)

    print('\n' + '=' * 52)
    print('  EVALUATION RESULTS')
    print('=' * 52)
    print(f'  Total test samples : {len(test_df)}')
    if unknown_count:
        print(f'  Unknown predictions: {unknown_count}  (routed to human review)')
    print()
    print(classification_report(
        list(true_valid), list(pred_valid),
        labels=['Eligible', 'Not Eligible'],
        target_names=['Eligible', 'Not Eligible'],
    ))

    metrics = {
        'accuracy' : round(accuracy_score(true_valid, pred_valid), 4),
        'f1'       : round(f1_score(true_valid, pred_valid, pos_label='Eligible'), 4),
        'precision': round(precision_score(true_valid, pred_valid, pos_label='Eligible'), 4),
        'recall'   : round(recall_score(true_valid, pred_valid, pos_label='Eligible'), 4),
        'unknown'  : unknown_count,
    }
    print('Summary:')
    for k, v in metrics.items():
        print(f'  {k:<12}: {v}')

    return metrics


def save_confusion_matrix(test_df: pd.DataFrame) -> None:
    # Build aligned (true, pred) pairs — both lists stay in sync
    pairs = [
        (t, p)
        for t, p in zip(test_df['response'].tolist(), test_df['predicted'].tolist())
        if p != 'Unknown'
    ]
    if not pairs:
        return
    true_valid, pred_valid = zip(*pairs)

    cm = confusion_matrix(list(true_valid), list(pred_valid),
                          labels=['Eligible', 'Not Eligible'])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Eligible', 'Not Eligible'],
        yticklabels=['Eligible', 'Not Eligible'],
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix — PatientTrialMapper')
    plt.tight_layout()

    out = config.DATA_DIR / 'confusion_matrix.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nConfusion matrix saved → {out}')


def main():
    if not config.TEST_JSON.exists():
        print('ERROR: data/test_samples.json not found. Run python data_prep.py first.')
        sys.exit(1)

    test_df = pd.read_json(config.TEST_JSON)
    print(f'Test set loaded: {len(test_df)} examples')

    print('\nLoading fine-tuned model...')
    model, tokenizer = load_model()

    test_df = run_evaluation(model, tokenizer, test_df)
    metrics = print_metrics(test_df)
    save_confusion_matrix(test_df)

    results_path = config.DATA_DIR / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    preds_path = config.DATA_DIR / 'test_predictions.json'
    test_df.to_json(preds_path, orient='records', indent=2)

    print(f'\nResults saved  → {results_path}')
    print(f'Predictions    → {preds_path}')
    print('\nevaluate.py done. Run: streamlit run app.py')


if __name__ == '__main__':
    main()
