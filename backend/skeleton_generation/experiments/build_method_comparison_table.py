import argparse
import json
import os
from collections import defaultdict

from skeleton_generation.experiments.benchmark_skeleton_methods import benchmark


def _safe_name(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem.replace(" ", "_")


def build_table(images, output_dir, confidence, iou):
    os.makedirs(output_dir, exist_ok=True)
    by_method = defaultdict(lambda: {"runtime_ms": [], "skeleton_pixels": [], "components": [], "errors": 0})

    for image_path in images:
        run_dir = os.path.join(output_dir, _safe_name(image_path))
        report_path = benchmark(image_path=image_path, output_dir=run_dir, confidence=confidence, iou=iou)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        for m in report.get("methods", []):
            slot = by_method[m["name"]]
            if m.get("error"):
                slot["errors"] += 1
                continue
            slot["runtime_ms"].append(float(m.get("runtime_ms", 0.0)))
            slot["skeleton_pixels"].append(float(m.get("skeleton_pixels", 0.0)))
            slot["components"].append(float(m.get("connected_components", 0.0)))

    rows = []
    for method, vals in sorted(by_method.items()):
        runtimes = vals["runtime_ms"]
        pixels = vals["skeleton_pixels"]
        components = vals["components"]
        rows.append(
            {
                "method": method,
                "runs_ok": len(runtimes),
                "runs_failed": vals["errors"],
                "mean_runtime_ms": round(sum(runtimes) / max(1, len(runtimes)), 3),
                "mean_skeleton_pixels": round(sum(pixels) / max(1, len(pixels)), 2),
                "mean_components": round(sum(components) / max(1, len(components)), 2),
            }
        )

    json_out = os.path.join(output_dir, "method_comparison_table.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    md_out = os.path.join(output_dir, "method_comparison_table.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Method Comparison Table\n\n")
        f.write("| Method | Runs OK | Runs Failed | Mean Runtime (ms) | Mean Skeleton Pixels | Mean Components |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['runs_ok']} | {row['runs_failed']} | {row['mean_runtime_ms']} | {row['mean_skeleton_pixels']} | {row['mean_components']} |\n"
            )

    return {"json": json_out, "markdown": md_out, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Build method comparison table across multiple images.")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output-dir", default="benchmark_outputs/multi")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.65)
    args = parser.parse_args()

    result = build_table(
        images=args.images,
        output_dir=args.output_dir,
        confidence=args.confidence,
        iou=args.iou,
    )
    print(result)


if __name__ == "__main__":
    main()
