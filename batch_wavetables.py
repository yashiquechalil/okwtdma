from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # Plotting becomes optional if matplotlib isn't installed

from .constants import Constant
from .dsp import (
    fade,
    interpolate,
    maximize,
    normalize,
    processing_log,
    resize,
    trim,
    slice_cycle,
)
from .formats import InputFile
from .utils import (
    get_frame_size_from_hint,
    pad_audio_data,
    write_wav,
    to_mono,
)


@dataclass(frozen=True)
class Preset:
    mode: str  # "fe" or "slice"
    hop: Optional[int] = None  # fe-only
    trim_amt: Optional[float] = 0.1
    fade_in: int = 200
    fade_out: int = 200
    smoothing: float = 0.0  # optional fe smoothing, 0 disables
    name: Optional[str] = None  # override label in output filenames

    def label(self) -> str:
        base = self.name or (
            f"{self.mode}"
            + (f"_hop{self.hop}" if self.mode == "fe" and self.hop is not None else "")
        )
        return f"{base}_trim{self.trim_amt}_fade{self.fade_in}-{self.fade_out}"


DEFAULT_PRESETS: List[Preset] = [
    Preset(mode="fe", hop=8, trim_amt=0.1, fade_in=200, fade_out=200),
    Preset(mode="fe", hop=4, trim_amt=0.1, fade_in=200, fade_out=200),
    Preset(mode="fe", hop=2, trim_amt=0.1, fade_in=200, fade_out=200),
    Preset(mode="slice", hop=None, trim_amt=0.1, fade_in=200, fade_out=200),
]


def is_audio_file(path: Path, allow_exts: Sequence[str]) -> bool:
    return path.is_file() and path.suffix.lower() in {e.lower() for e in allow_exts}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_frames(frames: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    frames32 = frames.astype(np.float32)
    peak = np.max(np.abs(frames32)) + eps
    return (frames32 / peak).astype(np.float32)


def fe_pipeline(
    audio: np.ndarray,
    samplerate: int,
    frame_size: int,
    target_num_frames: int,
    hop: int,
    fade_io: Tuple[int, int],
    smoothing: float = 0.0,
) -> Tuple[np.ndarray, int]:
    # Slice by f0 cycles
    frames = slice_cycle(audio, samplerate, frame_size, target_num_frames, hop=hop)

    # Optional simple inter-frame smoothing (insert interpolants)
    if smoothing and smoothing > 0:
        smoothed = []
        for i in range(len(frames) - 1):
            a = frames[i]
            b = frames[i + 1]
            interp = a * (1 - smoothing) + b * smoothing
            smoothed.append(a)
            smoothed.append(interp)
        smoothed.append(frames[-1])
        frames = np.asarray(smoothed, dtype=frames.dtype)
        target_num_frames = len(frames)

    # Fades
    if fade_io is not None:
        frames = fade(frames, frame_size, [fade_io[0], fade_io[1]])

    frames = normalize_frames(frames)
    return frames, len(frames)


def slice_pipeline(
    audio: np.ndarray,
    frame_size: int,
    target_num_frames: int,
    fade_io: Tuple[int, int],
    resize_mode: Optional[str] = None,  # passthrough for future use
) -> Tuple[np.ndarray, int, int]:
    # Round up to whole frames
    num_frames_rounded = math.ceil(len(audio) / frame_size)
    # Clamp to default max frames
    target_num_frames = min(num_frames_rounded, Constant.DEFAULT_NUM_FRAMES)
    target_array_size = frame_size * target_num_frames

    # Fit audio data into target_num_frames by truncating or padding
    if len(audio) >= target_array_size:
        audio_fit = audio[:target_array_size]
    else:
        audio_fit = pad_audio_data(audio, frame_size, target_num_frames)

    # Make sure it's 2D frames
    frames = audio_fit.reshape(-1, frame_size)

    # Fades
    frames = fade(frames, frame_size, [fade_io[0], fade_io[1]])

    # Normalize
    frames = normalize_frames(frames)

    return frames, frame_size, len(frames)


def plot_wavetable(
    frames: np.ndarray,
    out_png: Path,
    facecolor: str = "#111111",
    linecolor: str = "#A0A0A0",
    highlight_idx: Optional[int] = None,
    highlight_color: str = "#FFA64D",
    dpi: int = 150,
) -> None:
    if plt is None:
        print(f"[plot] matplotlib not installed, skipping plot: {out_png}")
        return

    ensure_parent(out_png)

    num_frames, frame_size = frames.shape
    # Normalize again for consistent plotting
    f = normalize_frames(frames)

    # Layout parameters
    width = 14
    height = 7
    y_gap = 1.0  # vertical spacing between rows
    y_scale = 0.4  # within-row amplitude scale

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)

    x = np.linspace(0.0, 1.0, frame_size, dtype=np.float32)

    # Stack rows in bands
    # e.g., group frames into 3 bands to resemble the sample layout
    bands = 3 if num_frames >= 24 else 2 if num_frames >= 12 else 1
    per_band = math.ceil(num_frames / bands)

    for i in range(num_frames):
        band = i // per_band
        row = i % per_band
        y_offset = (bands - 1 - band) * (per_band * y_gap * 0.2) + row * y_gap

        y = f[i] * y_scale + y_offset

        if highlight_idx is not None and i == highlight_idx:
            ax.plot(x, y, color=highlight_color, linewidth=2.0, alpha=1.0, solid_capstyle="round")
        else:
            ax.plot(x, y, color=linecolor, linewidth=1.0, alpha=0.9, solid_capstyle="round")

    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_png, facecolor=facecolor, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_one_file(
    infile: Path,
    out_root: Path,
    presets: Sequence[Preset],
    samplerate: Optional[int] = None,
    frame_size_cli: Optional[int] = None,
    target_num_frames_cli: Optional[int] = None,
    write_plots: bool = True,
) -> List[Path]:
    """Process a single audio file with all presets. Returns list of created outputs."""
    created: List[Path] = []

    infile_obj = InputFile(infile.as_posix())
    content = infile_obj.recognize_type().parse()

    # Resolve settings similarly to your main()
    samplerate = samplerate or getattr(content, "samplerate", None) or Constant.DEFAULT_SAMPLERATE
    frame_size_hint = get_frame_size_from_hint(infile_obj.name)
    frame_size = (
        frame_size_cli
        or content.frame_size
        or frame_size_hint
        or Constant.DEFAULT_FRAME_SIZE
    )
    target_num_frames = target_num_frames_cli or Constant.DEFAULT_NUM_FRAMES

    # Source audio (to mono if needed)
    audio_src = content.audio_data
    if audio_src.ndim > 1:
        audio_src = to_mono(audio_src)

    rel = infile.with_suffix("")  # keep relative structure for subfolders later

    for preset in presets:
        # Fresh copy of source per preset
        audio = np.copy(audio_src)

        # Trim first (like your main)
        if preset.trim_amt and preset.trim_amt > 0:
            audio = trim(audio, preset.trim_amt)

        label = preset.label()
        out_wav = out_root.joinpath(infile.parent.name, f"{rel.name}__{label}.wav")
        ensure_parent(out_wav)

        if preset.mode == "fe":
            frames, out_num_frames = fe_pipeline(
                audio=audio,
                samplerate=samplerate,
                frame_size=frame_size,
                target_num_frames=target_num_frames,
                hop=int(preset.hop or 2),
                fade_io=(preset.fade_in, preset.fade_out),
                smoothing=preset.smoothing,
            )
        elif preset.mode == "slice":
            frames, frame_size, out_num_frames = slice_pipeline(
                audio=audio,
                frame_size=frame_size,
                target_num_frames=target_num_frames,
                fade_io=(preset.fade_in, preset.fade_out),
            )
        else:
            raise ValueError(f"Unsupported mode in preset: {preset.mode}")

        # Optional maximize/normalize pass (normalize already applied in pipelines)
        frames = maximize(frames, frame_size, 1.0)
        frames = normalize(frames, 1.0)

        write_wav(
            filename_out=out_wav.as_posix(),
            audio_data=frames,
            frame_size=frame_size,
            num_frames=out_num_frames,
            samplerate=samplerate,
            add_uhwt_chunk=True,  # mirrors single-file behavior when writing out
            add_srge_chunk=False,
            comment=f"{label}",
        )
        created.append(out_wav)

        if write_plots:
            out_png = out_wav.with_suffix(".png")
            # Highlight the last frame to echo your example
            highlight_idx = len(frames) - 1 if len(frames) > 0 else None
            plot_wavetable(frames, out_png, highlight_idx=highlight_idx)
            created.append(out_png)

        # If your DSP stack logs, print once per preset
        if processing_log:
            print(f"\n[{infile.name}] Processing log ({label}):")
            for i, msg in enumerate(processing_log, start=1):
                print(f"  {i}. {msg}")

        print(f"[ok] {infile.name} -> {out_wav.name}")

    return created


def gather_files(root: Path, recursive: bool, allow_exts: Sequence[str]) -> List[Path]:
    if recursive:
        paths = [p for p in root.rglob("*") if is_audio_file(p, allow_exts)]
    else:
        paths = [p for p in root.iterdir() if is_audio_file(p, allow_exts)]
    return sorted(paths)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="batch_wavetables",
        description="Batch process wavetables over a directory tree with presets and plots.",
    )
    p.add_argument("--root", type=Path, required=True, help="Input root directory to scan.")
    p.add_argument("--out", type=Path, required=True, help="Output directory root.")
    p.add_argument(
        "--ext",
        nargs="+",
        default=[".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".m4a"],
        help="Allowed file extensions to process.",
    )
    p.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories.")
    p.add_argument("--samplerate", type=int, default=None, help="Override samplerate.")
    p.add_argument("--frame-size", type=int, default=None, help="Override frame size.")
    p.add_argument("--num-frames", type=int, default=None, help="Override target number of frames.")
    p.add_argument("--no-plot", action="store_true", help="Disable plot generation.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    root: Path = args.root
    out_root: Path = args.out
    recursive = not args.no_recursive
    out_root.mkdir(parents=True, exist_ok=True)

    files = gather_files(root, recursive, args.ext)
    if not files:
        print(f"No audio files found under: {root}")
        return 0

    print(f"Found {len(files)} files under {root} (recursive={recursive})")
    for f in files:
        try:
            process_one_file(
                infile=f,
                out_root=out_root,
                presets=DEFAULT_PRESETS,
                samplerate=args.samplerate,
                frame_size_cli=args.frame_size,
                target_num_frames_cli=args.num_frames,
                write_plots=not args.no_plot,
            )
        except Exception as e:
            print(f"[error] Failed to process {f}: {e}", file=sys.stderr)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())