#!/usr/bin/env python3
"""Turn PDF pages or PNG images into a Keynote-friendly MP4 with hard cuts."""

from __future__ import annotations

import argparse
import glob
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def output_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size must look like 1920x1080") from exc
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError(
            "width and height must be positive even numbers"
        )
    return width, height


def input_files(inputs: list[Path]) -> tuple[list[Path], str]:
    """Expand and validate a homogeneous collection of PDF or PNG files."""
    files: list[Path] = []
    for raw_item in inputs:
        item = raw_item.expanduser()
        if glob.has_magic(str(item)):
            matches = [Path(match) for match in sorted(glob.glob(str(item)))]
            if not matches:
                raise ValueError(f"input pattern matched no files: {raw_item}")
        else:
            matches = [item]

        for match in matches:
            if match.is_dir():
                directory_files = sorted(
                    path
                    for path in match.iterdir()
                    if path.suffix.lower() in {".pdf", ".png"}
                )
                if not directory_files:
                    raise ValueError(f"directory contains no PDF or PNG files: {match}")
                files.extend(directory_files)
            elif match.is_file() and match.suffix.lower() in {".pdf", ".png"}:
                files.append(match)
            else:
                raise ValueError(f"not a PDF or PNG file or directory: {match}")

    kinds = {path.suffix.lower() for path in files}
    if len(kinds) > 1:
        raise ValueError("PDF and PNG inputs cannot be mixed in the same movie")
    if not files:
        raise ValueError("no PDF or PNG files found")
    return files, kinds.pop()


def require_program(name: str) -> str:
    # Apple Silicon machines sometimes retain an older Intel Homebrew earlier
    # on PATH. Prefer the native Homebrew prefix when it contains this program.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        native_homebrew = Path("/opt/homebrew/bin") / name
        if native_homebrew.is_file():
            return str(native_homebrew)
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required program not found: {name}")
    return path


def render_pages(
    pdftoppm: str, pdfs: list[Path], directory: Path, dpi: int, first_page: bool
) -> list[Path]:
    pages: list[Path] = []
    for index, pdf in enumerate(pdfs):
        prefix = directory / f"document-{index:05d}"
        command = [pdftoppm, "-png", "-r", str(dpi)]
        if first_page:
            command.extend(["-f", "1", "-singlefile"])
        command.extend([str(pdf), str(prefix)])
        subprocess.run(command, check=True)
        rendered = list(directory.glob(f"{prefix.name}*.png"))
        rendered.sort(
            key=lambda path: (
                int(path.stem.rsplit("-", 1)[-1]) if path.stem != prefix.name else 1
            )
        )
        if not rendered:
            raise RuntimeError(f"no pages were rendered from {pdf}")
        pages.extend(rendered)
    return pages


def build_frame_sequence(
    pages: list[Path], directory: Path, seconds_per_page: float, fps: int
) -> tuple[str, int, int]:
    """Create an exact, numbered PNG sequence using space-efficient links."""
    frames_per_page = round(seconds_per_page * fps)
    if frames_per_page < 1:
        raise ValueError("seconds per page is shorter than one movie frame")

    directory.mkdir()
    frame_number = 0
    for page in pages:
        for _ in range(frames_per_page):
            frame = directory / f"frame-{frame_number:09d}.png"
            frame.symlink_to(page.resolve())
            frame_number += 1
    return str(directory / "frame-%09d.png"), frame_number, frames_per_page


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Render PDF pages or PNG images as an H.264 MP4 suitable for Keynote. "
            "Each image is held unchanged and followed by a hard cut; no frame "
            "interpolation is used. PDF and PNG inputs cannot be mixed."
        )
    )
    result.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="PDFs, PNGs, directories, or quoted wildcard patterns",
    )
    result.add_argument(
        "-o", "--output", type=Path, required=True, help="output .mp4 file"
    )
    result.add_argument(
        "--seconds-per-page", type=float, default=1.0, help="page duration (default: 1)"
    )
    result.add_argument(
        "--fps", type=int, default=30, help="movie frame rate (default: 30)"
    )
    result.add_argument(
        "--size",
        type=output_size,
        default=(1920, 1080),
        help="frame size (default: 1920x1080)",
    )
    result.add_argument(
        "--dpi", type=int, default=150, help="PDF render resolution (default: 150)"
    )
    result.add_argument(
        "--first-page", action="store_true", help="use only the first page of each PDF"
    )
    result.add_argument(
        "--overwrite", action="store_true", help="replace an existing output"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        files, input_kind = input_files(args.inputs)
        if args.output.suffix.lower() != ".mp4":
            raise ValueError("output must have an .mp4 extension")
        if args.output.exists() and not args.overwrite:
            raise ValueError(
                f"output already exists: {args.output} (use --overwrite to replace it)"
            )
        if args.seconds_per_page <= 0 or args.fps <= 0 or args.dpi <= 0:
            raise ValueError("duration, fps, and dpi must be greater than zero")

        ffmpeg = require_program("ffmpeg")
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="images-to-movie-") as temporary:
            temp_dir = Path(temporary)
            if input_kind == ".pdf":
                pdftoppm = require_program("pdftoppm")
                pages = render_pages(
                    pdftoppm, files, temp_dir, args.dpi, args.first_page
                )
                input_description = "PDF page(s)"
            else:
                if args.first_page:
                    raise ValueError("--first-page can only be used with PDF inputs")
                pages = files
                input_description = "PNG image(s)"
            sequence, frame_count, frames_per_page = build_frame_sequence(
                pages, temp_dir / "sequence", args.seconds_per_page, args.fps
            )
            width, height = args.size
            video_filter = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=white,"
                "setsar=1,format=yuv420p"
            )
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-y" if args.overwrite else "-n",
                    "-framerate",
                    str(args.fps),
                    "-start_number",
                    "0",
                    "-i",
                    sequence,
                    "-vf",
                    video_filter,
                    "-frames:v",
                    str(frame_count),
                    "-c:v",
                    "libx264",
                    # Independent frames make seeking and hard cuts reliable in Keynote.
                    "-g",
                    "1",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-movflags",
                    "+faststart",
                    str(args.output),
                ],
                check=True,
            )
        print(
            f"Wrote {args.output}: {len(pages)} {input_description}, "
            f"{frames_per_page} frames per image, {frame_count} frames total."
        )
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
