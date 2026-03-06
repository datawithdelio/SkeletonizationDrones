import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _normalize(value: str) -> str:
    return str(value).strip().lower()


def _split_values(raw: str) -> List[str]:
    return [_normalize(x) for x in raw.split(",") if _normalize(x)]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_column(header: Iterable[str], preferred: str, fallbacks: List[str]) -> Optional[str]:
    header_map = {_normalize(h): h for h in header}
    pref_norm = _normalize(preferred)
    if pref_norm in header_map:
        return header_map[pref_norm]
    for fb in fallbacks:
        fb_norm = _normalize(fb)
        if fb_norm in header_map:
            return header_map[fb_norm]
    return None


@dataclass
class PrepareStats:
    rows: int = 0
    copied: int = 0
    skipped_missing: int = 0
    skipped_unmapped: int = 0


def _load_rows(labels_file: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    suffix = labels_file.suffix.lower()

    if suffix == ".csv":
        with labels_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file is missing header row")
            rows = list(reader)
            return reader.fieldnames, rows

    if suffix == ".xlsx":
        try:
            import openpyxl  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openpyxl is required for .xlsx labels. Install it or convert labels to CSV."
            ) from exc

        workbook = openpyxl.load_workbook(labels_file, read_only=True, data_only=True)
        sheet = workbook.active
        values = list(sheet.values)
        if not values:
            raise ValueError("XLSX file is empty")
        header = [str(x).strip() if x is not None else "" for x in values[0]]
        rows = []
        for raw in values[1:]:
            row = {}
            for idx, h in enumerate(header):
                if not h:
                    continue
                row[h] = "" if idx >= len(raw) or raw[idx] is None else str(raw[idx])
            rows.append(row)
        return header, rows

    raise ValueError("Unsupported labels format. Use .csv or .xlsx")


def _resolve_source(images_dir: Path, filename_value: str) -> Path:
    filename_value = str(filename_value).strip()
    candidate = Path(filename_value)
    if candidate.is_absolute():
        return candidate
    return images_dir / candidate


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        alt = parent / f"{stem}_{i}{suffix}"
        if not alt.exists():
            return alt
        i += 1


def prepare_dataset(
    labels_file: Path,
    images_dir: Path,
    output_dir: Path,
    filename_col: str,
    label_col: str,
    daynight_col: str,
    drone_values: List[str],
    bird_values: List[str],
    day_values: List[str],
    night_values: List[str],
    default_period: str,
    copy_files: bool,
    dry_run: bool,
    skip_missing: bool,
) -> PrepareStats:
    header, rows = _load_rows(labels_file)

    filename_key = _find_column(header, filename_col, ["filename", "file", "image", "image_path", "path"])
    label_key = _find_column(header, label_col, ["label", "class", "category", "target"])
    period_key = _find_column(header, daynight_col, ["period", "time_of_day", "split", "daynight"]) if daynight_col else None

    if not filename_key:
        raise KeyError(f"Could not find filename column. Tried: {filename_col}")
    if not label_key:
        raise KeyError(f"Could not find label column. Tried: {label_col}")

    for period in ["day", "night"]:
        for klass in ["drones", "birds"]:
            _ensure_dir(output_dir / period / klass)

    stats = PrepareStats()

    for row in rows:
        stats.rows += 1

        raw_label = _normalize(row.get(label_key, ""))
        if raw_label in drone_values:
            class_folder = "drones"
        elif raw_label in bird_values:
            class_folder = "birds"
        else:
            stats.skipped_unmapped += 1
            continue

        raw_period = _normalize(row.get(period_key, "")) if period_key else ""
        if raw_period in day_values:
            period_folder = "day"
        elif raw_period in night_values:
            period_folder = "night"
        else:
            period_folder = default_period

        source = _resolve_source(images_dir, row.get(filename_key, ""))
        if not source.exists():
            stats.skipped_missing += 1
            if skip_missing:
                continue
            raise FileNotFoundError(f"Missing image file: {source}")

        destination = output_dir / period_folder / class_folder / source.name
        destination = _unique_destination(destination)

        if not dry_run:
            if copy_files:
                shutil.copy2(source, destination)
            else:
                shutil.move(source, destination)

        stats.copied += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Kaggle-style labeled dataset (CSV/XLSX) into day/night + drones/birds folder layout."
    )
    parser.add_argument("--labels-file", required=True, help="Path to labels .csv or .xlsx")
    parser.add_argument("--images-dir", required=True, help="Root folder containing images referenced in labels")
    parser.add_argument("--output-dir", required=True, help="Output dataset root")
    parser.add_argument("--filename-col", default="filename")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--daynight-col", default="")
    parser.add_argument("--drone-values", default="drone,uav,quadcopter")
    parser.add_argument("--bird-values", default="bird")
    parser.add_argument("--day-values", default="day,daytime")
    parser.add_argument("--night-values", default="night,nighttime,lowlight")
    parser.add_argument("--default-period", choices=["day", "night"], default="day")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    args = parser.parse_args()

    stats = prepare_dataset(
        labels_file=Path(args.labels_file),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        filename_col=args.filename_col,
        label_col=args.label_col,
        daynight_col=args.daynight_col,
        drone_values=_split_values(args.drone_values),
        bird_values=_split_values(args.bird_values),
        day_values=_split_values(args.day_values),
        night_values=_split_values(args.night_values),
        default_period=args.default_period,
        copy_files=not args.move,
        dry_run=args.dry_run,
        skip_missing=args.skip_missing,
    )

    print(
        {
            "rows": stats.rows,
            "copied_or_moved": stats.copied,
            "skipped_missing": stats.skipped_missing,
            "skipped_unmapped": stats.skipped_unmapped,
            "dry_run": args.dry_run,
            "output_dir": args.output_dir,
        }
    )


if __name__ == "__main__":
    main()
