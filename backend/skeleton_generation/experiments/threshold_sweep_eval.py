import argparse
import json
import os
from itertools import product
from typing import List

from skeleton_generation.experiments.run_full_evaluation import run_evaluation


def _parse_float_list(raw: str) -> List[float]:
    values = []
    for x in raw.split(','):
        x = x.strip()
        if not x:
            continue
        values.append(float(x))
    return values


def _parse_int_list(raw: str) -> List[int]:
    values = []
    for x in raw.split(','):
        x = x.strip()
        if not x:
            continue
        values.append(int(x))
    return values


def sweep(
    dataset_dir: str,
    output_dir: str,
    confidences: List[float],
    ious: List[float],
    target_classes: List[int],
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for conf, iou in product(confidences, ious):
        run_dir = os.path.join(output_dir, f"c{conf:.2f}_iou{iou:.2f}")
        report_path = run_evaluation(
            dataset_dir=dataset_dir,
            output_dir=run_dir,
            confidence=conf,
            iou=iou,
            target_classes=target_classes,
            benchmark_image="",
        )
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        metrics = report['aggregate']['metrics']
        latency = report['aggregate']['latency_ms']['mean']
        row = {
            'confidence': conf,
            'iou': iou,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'mean_latency_ms': latency,
            'samples': report['aggregate']['samples'],
            'report_path': report_path,
        }
        rows.append(row)

    # sort by research-priority objective: F1 -> Recall -> lower latency
    rows.sort(key=lambda r: (r['f1'], r['recall'], -r['mean_latency_ms']), reverse=True)

    best = rows[0] if rows else None
    out_json = os.path.join(output_dir, 'threshold_sweep_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'all': rows}, f, indent=2)

    out_md = os.path.join(output_dir, 'threshold_sweep_results.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('# Threshold Sweep Results\n\n')
        if best:
            f.write('## Best Setting\n\n')
            f.write(f"- confidence: `{best['confidence']}`\n")
            f.write(f"- iou: `{best['iou']}`\n")
            f.write(f"- f1: `{best['f1']}`\n")
            f.write(f"- recall: `{best['recall']}`\n")
            f.write(f"- precision: `{best['precision']}`\n")
            f.write(f"- mean latency (ms): `{best['mean_latency_ms']}`\n\n")

        f.write('| confidence | iou | precision | recall | f1 | accuracy | mean latency (ms) | samples |\n')
        f.write('|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(
                f"| {r['confidence']} | {r['iou']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['accuracy']} | {r['mean_latency_ms']} | {r['samples']} |\n"
            )

    return {'best': best, 'json': out_json, 'markdown': out_md}


def main():
    parser = argparse.ArgumentParser(description='Sweep confidence/IoU thresholds and choose best aggregate F1.')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', default='evaluation_outputs/threshold_sweep')
    parser.add_argument('--confidences', default='0.10,0.15,0.20,0.25,0.30')
    parser.add_argument('--ious', default='0.50,0.60,0.70')
    parser.add_argument('--target-classes', default='0,2,4')
    args = parser.parse_args()

    result = sweep(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        confidences=_parse_float_list(args.confidences),
        ious=_parse_float_list(args.ious),
        target_classes=_parse_int_list(args.target_classes),
    )
    print(result)


if __name__ == '__main__':
    main()
