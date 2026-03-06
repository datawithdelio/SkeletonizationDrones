import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2 as cv


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


def _norm(s: str) -> str:
    return s.strip().lower()


def _split_csv(raw: str) -> List[str]:
    return [_norm(x) for x in raw.split(",") if _norm(x)]


def _has_any(text: str, keys: List[str]) -> bool:
    t = _norm(text)
    return any(k in t for k in keys)


@dataclass
class Stats:
    images_copied: int = 0
    frames_extracted: int = 0
    skipped_unmapped: int = 0
    skipped_unreadable: int = 0


def _classify(path_text: str, drone_keys: List[str], bird_keys: List[str]) -> str:
    if _has_any(path_text, drone_keys):
        return "drones"
    if _has_any(path_text, bird_keys):
        return "birds"
    return ""


def _period(path_text: str, night_keys: List[str]) -> str:
    return "night" if _has_any(path_text, night_keys) else "day"


def _extract_frames(
    src: Path,
    dst_dir: Path,
    frame_step: int,
    max_frames_per_video: int,
) -> int:
    cap = cv.VideoCapture(str(src))
    if not cap.isOpened():
        return -1

    frame_idx = 0
    written = 0
    stem = src.stem.replace(" ", "_")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            out = dst_dir / f"{stem}_f{frame_idx:06d}.jpg"
            cv.imwrite(str(out), frame)
            written += 1
            if written >= max_frames_per_video:
                break

        frame_idx += 1

    cap.release()
    return written


def prepare(
    input_dir: Path,
    output_dir: Path,
    drone_keywords: List[str],
    bird_keywords: List[str],
    night_keywords: List[str],
    frame_step: int,
    max_frames_per_video: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for period in ("day", "night"):
        for klass in ("drones", "birds"):
            (output_dir / period / klass).mkdir(parents=True, exist_ok=True)

    stats = Stats()

    for src in sorted(input_dir.rglob("*")):
        if not src.is_file():
            continue
        ext = src.suffix.lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            continue

        rel = str(src.relative_to(input_dir))
        klass = _classify(rel, drone_keywords, bird_keywords)
        if not klass:
            stats.skipped_unmapped += 1
            continue

        period = _period(rel, night_keywords)
        dst_dir = output_dir / period / klass

        if ext in IMAGE_EXTS:
            dst = dst_dir / src.name
            i = 1
            while dst.exists():
                dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
                i += 1
            shutil.copy2(src, dst)
            stats.images_copied += 1
            continue

        extracted = _extract_frames(
            src=src,
            dst_dir=dst_dir,
            frame_step=max(1, frame_step),
            max_frames_per_video=max(1, max_frames_per_video),
        )
        if extracted < 0:
            stats.skipped_unreadable += 1
        else:
            stats.frames_extracted += extracted

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare day/night drone-vs-bird evaluation dataset from mixed image/video media."
    )
    parser.add_argument("--input-dir", required=True, help="Input root directory with media.")
    parser.add_argument("--output-dir", required=True, help="Output evaluation dataset root.")
    parser.add_argument(
        "--drone-keywords",
        default="drone,uav,quadcopter,hexacopter,swarm,recreational drones,high-end camera drones,fixed wing",
    )
    parser.add_argument(
        "--bird-keywords",
        default="bird,fighter,jet,missile,munition,plane",
    )
    parser.add_argument("--night-keywords", default="night,lowlight,dark")
    parser.add_argument("--frame-step", type=int, default=20, help="Sample every Nth frame.")
    parser.add_argument("--max-frames-per-video", type=int, default=20)
    args = parser.parse_args()

    stats = prepare(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        drone_keywords=_split_csv(args.drone_keywords),
        bird_keywords=_split_csv(args.bird_keywords),
        night_keywords=_split_csv(args.night_keywords),
        frame_step=args.frame_step,
        max_frames_per_video=args.max_frames_per_video,
    )

    print(
        {
            "output_dir": args.output_dir,
            "images_copied": stats.images_copied,
            "frames_extracted": stats.frames_extracted,
            "skipped_unmapped": stats.skipped_unmapped,
            "skipped_unreadable": stats.skipped_unreadable,
        }
    )


if __name__ == "__main__":
    main()
