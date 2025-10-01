#!/usr/bin/env python3
"""
Batch runner for okwt with fixed presets over a directory tree.

Usage:
  python batch_presets.py --root <input_dir> --out <output_dir>

Notes:
  - Runs four presets for each audio file found recursively under --root:
      --mode fe --hop 8 --trim 0.1 --fade 200 200
      --mode fe --hop 4 --trim 0.1 --fade 200 200
      --mode fe --hop 2 --trim 0.1 --fade 200 200
      --mode slice --trim 0.1 --fade 200 200
  - Uses the current Python interpreter to execute: python -m okwt.app
    so it works from the repo without installing a console script.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


ALLOWED_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class Preset:
    mode: str
    hop: Optional[int] = None
    trim: float = 0.1
    fade_in: int = 200
    fade_out: int = 200

    def label(self) -> str:
        base = self.mode + (f"_hop{self.hop}" if self.mode == "fe" and self.hop is not None else "")
        return f"{base}_trim{self.trim}_fade{self.fade_in}-{self.fade_out}"


PRESETS: Sequence[Preset] = (
    Preset(mode="fe", hop=8, trim=0.1, fade_in=200, fade_out=200),
    Preset(mode="fe", hop=4, trim=0.1, fade_in=200, fade_out=200),
    Preset(mode="fe", hop=2, trim=0.1, fade_in=200, fade_out=200),
    Preset(mode="slice", hop=None, trim=0.1, fade_in=200, fade_out=200),
)


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTS


def gather_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if is_audio_file(p))


def build_cmd(infile: Path, outfile: Path, preset: Preset) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "okwt.app",
        "-i",
        infile.as_posix(),
        "-o",
        outfile.as_posix(),
        "--mode",
        preset.mode,
        "--trim",
        str(preset.trim),
        "--fade",
        str(preset.fade_in),
        str(preset.fade_out),
    ]
    if preset.mode == "fe" and preset.hop is not None:
        cmd += ["--hop", str(preset.hop)]
    return cmd


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="batch_presets",
        description="Run okwt.app over a directory with four fixed presets.",
    )
    p.add_argument("--root", type=Path, required=True, help="Input root directory to scan (recursive)")
    p.add_argument("--out", type=Path, required=True, help="Output directory root")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    p.add_argument("--jobs", type=int, default=1, help="Reserved for future parallel runs (currently serial)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root: Path = args.root.resolve()
    out_root: Path = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not root.exists() or not root.is_dir():
        print(f"Input root does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    files = gather_files(root)
    if not files:
        print(f"No audio files found under: {root}")
        return 0

    print(f"Found {len(files)} files under {root}")

    for f in files:
        rel = f.relative_to(root)
        rel_no_ext = rel.with_suffix("")
        for preset in PRESETS:
            label = preset.label()
            out_path = out_root / rel_no_ext.parent / f"{rel_no_ext.name}__{label}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = build_cmd(f, out_path, preset)
            if args.dry_run:
                print("DRY-RUN:", " ".join(shlex.quote(c) for c in cmd))
                continue
            print("RUN:", " ".join(shlex.quote(c) for c in cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[error] Failed: {f} [{label}] -> {e}", file=sys.stderr)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
