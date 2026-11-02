"""Deliverable-aware evaluation helpers.

This module converts files produced by agentic tasks into bounded,
judge-readable evidence. It intentionally avoids assuming a single output
format: GDPval tasks can ask for documents, spreadsheets, images, audio,
video, archives, or mixed bundles.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import wave
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

LOGGER = logging.getLogger(__name__)
_VISION_UNAVAILABLE_CLIENTS: set[int] = set()

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".xml",
    ".html",
    ".htm",
    ".svg",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".sol",
    ".tf",
    ".tfvars",
    ".hcl",
    ".css",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh",
    ".log",
    ".overpassql",
    ".step",
    ".stp",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm"}
PRIMARY_DELIVERABLE_EXTENSIONS = (
    TEXT_EXTENSIONS
    | IMAGE_EXTENSIONS
    | AUDIO_EXTENSIONS
    | VIDEO_EXTENSIONS
    | SPREADSHEET_EXTENSIONS
    | {".docx", ".pptx", ".pdf", ".ipynb", ".zip", ".psd", ".step", ".stp", ".dwg"}
)
SUPPORTING_ONLY_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".css",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".csv",
    ".tsv",
    ".log",
    ".yaml",
    ".yml",
}
IGNORED_OUTPUT_FILENAMES = {"metadata.json", "trace.json", "response.txt"}
AUXILIARY_OUTPUT_DIR_NAMES = {
    "inputs",
    "input",
    "reference",
    "references",
    "refs",
    "__pycache__",
    ".git",
    ".pytest_cache",
}
AUXILIARY_OUTPUT_STEM_PATTERN = re.compile(
    r"^(?:"
    r"ref(?:erence)?|"
    r"input|"
    r"source|"
    r"original|"
    r"raw|"
    r"scratch|"
    r"tmp|"
    r"temp|"
    r"intermediate|"
    r"analysis|"
    r"analy[sz]e|"
    r"converted|"
    r"transcoded"
    r")(?:[_\-.]|$)",
    re.I,
)
_SPREADSHEET_OUTPUT_PATTERN = re.compile(
    r"\b(?:"
    r"(?:create|produce|deliver|export|save|submit|return|prepare|build|update|generate)"
    r"[\w\s,;:/\\.-]{0,120}"
    r"(?:xlsx|excel(?:\s+(?:workbook|output|file))?|spreadsheet|workbook)"
    r"|"
    r"(?:xlsx|excel(?:\s+(?:workbook|output|file))?|spreadsheet|workbook)"
    r"[\w\s,;:/\\.-]{0,80}"
    r"(?:deliverable|output|final file|submission|to submit)"
    r")\b",
    re.I,
)

_EXPECTED_EXT_PATTERNS: tuple[tuple[re.Pattern[str], set[str]], ...] = (
    (_SPREADSHEET_OUTPUT_PATTERN, {".xlsx", ".xlsm", ".xltx", ".xltm"}),
    (re.compile(r"\b(?:docx|word document|microsoft word)\b", re.I), {".docx"}),
    (re.compile(r"\b(?:pptx|powerpoint|slide deck|presentation)\b", re.I), {".pptx"}),
    (re.compile(r"\bpdf\b", re.I), {".pdf"}),
    (re.compile(r"\bpng\b", re.I), {".png"}),
    (re.compile(r"\b(?:mp4|video file|h\.?264|avc)\b", re.I), {".mp4"}),
    (re.compile(r"\b(?:wav|audio file|final mix|music track)\b", re.I), AUDIO_EXTENSIONS),
    (re.compile(r"\b(?:zip|archive)\b", re.I), {".zip"}),
    (re.compile(r"\b(?:ipynb|notebook)\b", re.I), {".ipynb"}),
)


@dataclass
class FileEvidence:
    """A bounded, text-first description of a deliverable or reference file."""

    path: str
    name: str
    suffix: str
    mime_type: str | None
    size_bytes: int
    sha256: str
    kind: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list["FileEvidence"] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["children"] = [child.to_dict() for child in self.children]
        return data

    def to_prompt_section(self, *, max_chars: int = 16_000, depth: int = 0) -> str:
        indent = "  " * depth
        meta = json.dumps(self.metadata, default=str, ensure_ascii=False)[:4000]
        text = self.text
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        lines = [
            f"{indent}### File: {self.name}",
            f"{indent}- path: {self.path}",
            f"{indent}- kind: {self.kind}",
            f"{indent}- mime_type: {self.mime_type}",
            f"{indent}- size_bytes: {self.size_bytes}",
            f"{indent}- sha256: {self.sha256}",
            f"{indent}- metadata: {meta}",
        ]
        if self.errors:
            lines.append(f"{indent}- extraction_errors: {json.dumps(self.errors, ensure_ascii=False)}")
        if text:
            lines.append(f"{indent}Content/evidence:\n{text}")
        for child in self.children:
            lines.append(child.to_prompt_section(max_chars=max_chars // 2, depth=depth + 1))
        return "\n".join(lines)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bounded(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _safe_read_text(path: Path, max_chars: int) -> str:
    return _bounded(path.read_text(encoding="utf-8", errors="ignore"), max_chars)


def _inspect_text(path: Path, max_chars: int) -> tuple[str, dict[str, Any], list[str]]:
    return _safe_read_text(path, max_chars), {}, []


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _inspect_notebook(path: Path, max_chars: int) -> tuple[str, dict[str, Any], list[str]]:
    errors: list[str] = []
    metadata: dict[str, Any] = {}
    parts: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        cells = notebook.get("cells", [])
        metadata["nbformat"] = notebook.get("nbformat")
        metadata["nbformat_minor"] = notebook.get("nbformat_minor")
        metadata["cell_count"] = len(cells) if isinstance(cells, list) else 0
        nb_meta = notebook.get("metadata", {}) if isinstance(notebook.get("metadata"), dict) else {}
        kernelspec = nb_meta.get("kernelspec", {}) if isinstance(nb_meta.get("kernelspec"), dict) else {}
        language_info = nb_meta.get("language_info", {}) if isinstance(nb_meta.get("language_info"), dict) else {}
        metadata["kernelspec"] = {
            key: kernelspec.get(key)
            for key in ("display_name", "language", "name")
            if kernelspec.get(key) is not None
        }
        metadata["language_info"] = {
            key: language_info.get(key)
            for key in ("name", "version", "mimetype")
            if language_info.get(key) is not None
        }

        if not isinstance(cells, list):
            return "", metadata, ["notebook cells field is not a list"]

        for idx, cell in enumerate(cells[:80], start=1):
            if not isinstance(cell, dict):
                continue
            cell_type = str(cell.get("cell_type") or "unknown")
            source = _cell_source(cell).strip()
            parts.append(f"# Cell {idx} ({cell_type})")
            if source:
                parts.append(source)
            outputs = cell.get("outputs", [])
            if isinstance(outputs, list) and outputs:
                output_parts: list[str] = []
                for output in outputs[:5]:
                    if not isinstance(output, dict):
                        continue
                    if output.get("ename") or output.get("evalue"):
                        output_parts.append(
                            "Error: "
                            + " ".join(
                                str(value)
                                for value in (output.get("ename"), output.get("evalue"))
                                if value
                            )
                        )
                    text = output.get("text")
                    if isinstance(text, list):
                        output_parts.append("".join(str(part) for part in text))
                    elif text:
                        output_parts.append(str(text))
                    data = output.get("data", {})
                    if isinstance(data, dict):
                        plain = data.get("text/plain")
                        if isinstance(plain, list):
                            output_parts.append("".join(str(part) for part in plain))
                        elif plain:
                            output_parts.append(str(plain))
                if output_parts:
                    parts.append("## Outputs")
                    parts.append(_bounded("\n".join(output_parts), max_chars // 4))
        if len(cells) > 80:
            parts.append(f"... [{len(cells) - 80} more cells]")
    except json.JSONDecodeError as exc:
        errors.append(f"notebook JSON parse error: {exc}")
    except Exception as exc:
        errors.append(f"notebook parse error: {exc}")
    return _bounded("\n\n".join(parts), max_chars), metadata, errors


def _inspect_spreadsheet(path: Path, max_chars: int) -> tuple[str, dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        from openpyxl import load_workbook
    except ImportError:
        return "", {}, ["openpyxl is not installed"]

    parts: list[str] = []
    metadata: dict[str, Any] = {}
    try:
        wb_values = load_workbook(str(path), data_only=True, read_only=True)
        wb_formulas = load_workbook(str(path), data_only=False, read_only=True)
        metadata["sheet_names"] = wb_values.sheetnames
        for ws_values, ws_formulas in zip(wb_values.worksheets, wb_formulas.worksheets):
            parts.append(f"# Sheet: {ws_values.title}")
            metadata.setdefault("sheets", []).append(
                {
                    "name": ws_values.title,
                    "max_row": ws_values.max_row,
                    "max_column": ws_values.max_column,
                }
            )
            row_limit = min(ws_values.max_row or 0, 80)
            col_limit = min(ws_values.max_column or 0, 30)
            for r_idx, row in enumerate(
                ws_values.iter_rows(min_row=1, max_row=row_limit, max_col=col_limit, values_only=True),
                start=1,
            ):
                formula_row = next(
                    ws_formulas.iter_rows(min_row=r_idx, max_row=r_idx, max_col=col_limit, values_only=True)
                )
                cells = []
                for value, formula in zip(row, formula_row):
                    if formula is not None and isinstance(formula, str) and formula.startswith("="):
                        cells.append(formula)
                    else:
                        cells.append("" if value is None else str(value))
                if any(cells):
                    parts.append("\t".join(cells))
            if (ws_values.max_row or 0) > row_limit:
                parts.append(f"... [{(ws_values.max_row or 0) - row_limit} more rows]")
        return _bounded("\n".join(parts), max_chars), metadata, errors
    except Exception as exc:
        return "", metadata, [f"spreadsheet parse error: {exc}"]


def _inspect_docx(path: Path, max_chars: int) -> tuple[str, dict[str, Any], list[str]]:
    try:
        import docx
    except ImportError:
        return "", {}, ["python-docx is not installed"]
    try:
        doc = docx.Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text]
        for t_idx, table in enumerate(doc.tables, start=1):
            parts.append(f"# Table {t_idx}")
            for row in table.rows:
                parts.append("\t".join(cell.text for cell in row.cells))
        return _bounded("\n".join(parts), max_chars), {"paragraphs": len(doc.paragraphs), "tables": len(doc.tables)}, []
    except Exception as exc:
        return "", {}, [f"docx parse error: {exc}"]


def _inspect_pptx(path: Path, max_chars: int) -> tuple[str, dict[str, Any], list[str]]:
    try:
        from pptx import Presentation
    except ImportError:
        return "", {}, ["python-pptx is not installed"]
    try:
        prs = Presentation(str(path))
        parts: list[str] = []
        for idx, slide in enumerate(prs.slides, start=1):
            parts.append(f"# Slide {idx}")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    parts.append(shape.text)
        return _bounded("\n".join(parts), max_chars), {"slides": len(prs.slides)}, []
    except Exception as exc:
        return "", {}, [f"pptx parse error: {exc}"]


def _inspect_pdf(path: Path, max_chars: int) -> tuple[str, dict[str, Any], list[str]]:
    errors: list[str] = []
    parts: list[str] = []
    metadata: dict[str, Any] = {}
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            metadata["pages"] = len(pdf.pages)
            for idx, page in enumerate(pdf.pages[:20], start=1):
                parts.append(f"# Page {idx}")
                parts.append(page.extract_text() or "")
            if len(pdf.pages) > 20:
                parts.append(f"... [{len(pdf.pages) - 20} more pages]")
    except ImportError:
        errors.append("pdfplumber is not installed")
    except Exception as exc:
        errors.append(f"pdf text parse error: {exc}")

    if not "".join(parts).strip():
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            metadata.setdefault("pages", len(reader.pages))
            for idx, page in enumerate(reader.pages[:20], start=1):
                parts.append(f"# Page {idx}")
                parts.append(page.extract_text() or "")
            if len(reader.pages) > 20:
                parts.append(f"... [{len(reader.pages) - 20} more pages]")
        except ImportError:
            errors.append("pypdf is not installed")
        except Exception as exc:
            errors.append(f"pypdf text parse error: {exc}")

    if not "".join(parts).strip():
        try:
            result = subprocess.run(
                ["pdftotext", str(path), "-"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.stdout:
                parts.append(result.stdout)
            if result.stderr:
                errors.append(result.stderr[:500])
        except Exception as exc:
            errors.append(f"pdftotext failed: {exc}")
    if "".join(parts).strip():
        errors = [
            error for error in errors
            if error != "pdfplumber is not installed"
        ]
    return _bounded("\n".join(parts), max_chars), metadata, errors


def _vision_model_name(client: Any) -> str:
    model = getattr(client, "model", None)
    if model:
        return str(model)
    config = getattr(client, "_config", {}) or {}
    return str(config.get("model") or "")


def _describe_image_with_vision(path: Path, client: Any) -> str | None:
    if client is None or id(client) in _VISION_UNAVAILABLE_CLIENTS:
        return None
    if not (hasattr(client, "_chat_completion") or hasattr(client, "stream_chat_completion")):
        return None
    try:
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "jpg":
            suffix = "jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        messages = [
            {"role": "system", "content": "You describe visual evidence for an evaluator."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image or video frame factually. Include visible "
                            "subjects, scene, composition, colors, text, labels, UI, charts, "
                            "and any artifacts relevant to professional evaluation."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{suffix};base64,{encoded}"},
                    },
                ],
            },
        ]
        model = _vision_model_name(client)
        if hasattr(client, "_chat_completion"):
            response = client._chat_completion(model, "", messages=messages)
        else:
            response = client.stream_chat_completion(model=model, prompt="", messages=messages)
        content = getattr(response, "content", "") or ""
        return content.strip() or None
    except Exception as exc:
        _VISION_UNAVAILABLE_CLIENTS.add(id(client))
        LOGGER.debug("Vision description unavailable: %s", exc)
        return None


def _inspect_image(
    path: Path,
    max_chars: int,
    *,
    vision_client: Any = None,
) -> tuple[str, dict[str, Any], list[str]]:
    errors: list[str] = []
    metadata: dict[str, Any] = {}
    text_parts: list[str] = []
    try:
        from PIL import Image, ImageStat

        with Image.open(str(path)) as img:
            metadata.update(
                {
                    "format": img.format,
                    "mode": img.mode,
                    "width": img.width,
                    "height": img.height,
                }
            )
            rgb = img.convert("RGB")
            stat_img = rgb.resize((min(rgb.width, 128), min(rgb.height, 128)))
            stat = ImageStat.Stat(stat_img)
            metadata["mean_rgb"] = [round(value, 2) for value in stat.mean[:3]]
            metadata["stddev_rgb"] = [round(value, 2) for value in stat.stddev[:3]]
    except Exception as exc:
        errors.append(f"image metadata error: {exc}")

    if shutil.which("tesseract"):
        try:
            result = subprocess.run(
                ["tesseract", str(path), "stdout"],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if result.stdout.strip():
                text_parts.append("# OCR text")
                text_parts.append(result.stdout)
            if result.stderr and result.returncode != 0:
                errors.append(result.stderr[:500])
        except Exception as exc:
            errors.append(f"image OCR failed: {exc}")
    else:
        metadata["ocr"] = "tesseract not available; visual content represented by metadata only"

    vision_description = _describe_image_with_vision(path, vision_client)
    if vision_description:
        text_parts.append("# Vision description")
        text_parts.append(vision_description)
        metadata["vision_description"] = "included"
    elif vision_client is not None:
        metadata["vision_description"] = "unavailable"

    return _bounded("\n".join(text_parts), max_chars), metadata, errors


def _inspect_wav_pcm_stats(wav: wave.Wave_read, *, max_frames: int = 2_000_000) -> dict[str, Any]:
    """Compute bounded PCM stats for judge-readable audio evidence."""
    sample_width = wav.getsampwidth()
    if sample_width not in {1, 2, 3, 4}:
        return {"pcm_stats_error": f"unsupported sample width: {sample_width}"}

    total_frames = wav.getnframes()
    frames_to_read = min(total_frames, max_frames)
    if frames_to_read <= 0:
        return {"measured_frames": 0, "measured_samples": 0}

    full_scale = float(1 << (8 * sample_width - 1))
    max_abs = 0
    sum_squares = 0.0
    sample_count = 0
    frames_read = 0
    wav.rewind()

    while frames_read < frames_to_read:
        chunk_frames = min(65_536, frames_to_read - frames_read)
        chunk = wav.readframes(chunk_frames)
        if not chunk:
            break
        frames_read += chunk_frames
        for offset in range(0, len(chunk) - sample_width + 1, sample_width):
            raw = chunk[offset:offset + sample_width]
            if sample_width == 1:
                value = raw[0] - 128
            else:
                value = int.from_bytes(raw, "little", signed=True)
            abs_value = abs(value)
            if abs_value > max_abs:
                max_abs = abs_value
            sum_squares += float(value) * float(value)
            sample_count += 1

    rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0

    def dbfs(value: float) -> float | None:
        if value <= 0:
            return None
        return round(20.0 * math.log10(value / full_scale), 3)

    return {
        "measured_frames": frames_read,
        "measured_samples": sample_count,
        "stats_truncated": total_frames > frames_read,
        "peak_amplitude": max_abs,
        "peak_dbfs": dbfs(float(max_abs)),
        "rms_amplitude": round(rms, 3),
        "rms_dbfs": dbfs(rms),
    }


def _inspect_audio(path: Path, max_chars: int) -> tuple[str, dict[str, Any], list[str]]:
    errors: list[str] = []
    metadata: dict[str, Any] = {}
    text_parts: list[str] = []
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                metadata.setdefault("duration_seconds", wav.getnframes() / float(wav.getframerate()))
                metadata["sample_rate"] = wav.getframerate()
                metadata["channels"] = wav.getnchannels()
                metadata["sample_width_bytes"] = wav.getsampwidth()
                metadata["bits_per_sample"] = wav.getsampwidth() * 8
                metadata["frames"] = wav.getnframes()
                metadata["pcm_stats"] = _inspect_wav_pcm_stats(wav)
        except Exception as exc:
            errors.append(f"wav metadata error: {exc}")
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(str(path))
        if audio is not None and audio.info is not None:
            metadata.setdefault("duration_seconds", getattr(audio.info, "length", None))
            metadata["bitrate"] = getattr(audio.info, "bitrate", None)
            metadata.setdefault("sample_rate", getattr(audio.info, "sample_rate", None))
            metadata.setdefault("channels", getattr(audio.info, "channels", None))
    except ImportError:
        metadata["mutagen"] = "not installed"
    except Exception as exc:
        errors.append(f"audio metadata error: {exc}")

    if os.getenv("IPW_ENABLE_MEDIA_TRANSCRIPTION", "").lower() in {"1", "true", "yes"}:
        try:
            import whisper

            model_name = os.getenv("IPW_WHISPER_MODEL", "base")
            model = whisper.load_model(model_name)
            result = model.transcribe(str(path), verbose=False)
            transcript = result.get("text") or ""
            if transcript:
                text_parts.append("# Transcript")
                text_parts.append(transcript)
        except ImportError:
            errors.append("whisper is not installed")
        except Exception as exc:
            errors.append(f"audio transcription failed: {exc}")
    else:
        metadata["transcription"] = "disabled; set IPW_ENABLE_MEDIA_TRANSCRIPTION=1 to enable"

    return _bounded("\n".join(text_parts), max_chars), metadata, errors


def _ffmpeg_executable() -> str | None:
    """Return an ffmpeg executable from PATH or the packaged imageio fallback."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run_media_command(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _ffprobe_metadata(path: Path) -> dict[str, Any]:
    if not shutil.which("ffprobe"):
        return {"ffprobe": "not available"}
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if not result.stdout:
        return {"ffprobe_error": result.stderr[:500]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"ffprobe_raw": result.stdout[:2000]}


def _parse_ffmpeg_banner(stderr: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        metadata["duration_seconds"] = (
            int(hours) * 3600
            + int(minutes) * 60
            + float(seconds)
        )
    video_match = re.search(
        r"Stream #\S+ Video:\s*([^,\n]+).*?(\d{2,5})x(\d{2,5})(?:[,\s].*?(\d+(?:\.\d+)?)\s*fps)?",
        stderr,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if video_match:
        codec, width, height, fps = video_match.groups()
        metadata["video_stream"] = {
            "codec": codec.strip(),
            "width": int(width),
            "height": int(height),
            "fps": float(fps) if fps else None,
        }
    audio_match = re.search(
        r"Stream #\S+ Audio:\s*([^,\n]+).*?(\d+)\s*Hz(?:,\s*([^,\n]+))?",
        stderr,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if audio_match:
        codec, sample_rate, channels = audio_match.groups()
        metadata["audio_stream"] = {
            "codec": codec.strip(),
            "sample_rate": int(sample_rate),
            "channels": channels.strip() if channels else None,
        }
    return metadata


def _inspect_video(
    path: Path,
    max_chars: int,
    *,
    vision_client: Any = None,
) -> tuple[str, dict[str, Any], list[str], list[FileEvidence]]:
    errors: list[str] = []
    metadata = _ffprobe_metadata(path)
    text_parts: list[str] = []
    children: list[FileEvidence] = []
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        metadata["video_decoder"] = "unavailable; install ffmpeg or the imageio-ffmpeg Python package"
        return "", metadata, errors, children

    metadata["ffmpeg"] = ffmpeg
    try:
        probe = _run_media_command([ffmpeg, "-hide_banner", "-i", str(path)], timeout=60)
        metadata.update(_parse_ffmpeg_banner(probe.stderr or ""))
    except Exception as exc:
        errors.append(f"ffmpeg metadata probe failed: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frame_pattern = root / "frame_%03d.png"
        frame_cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(path),
            "-vf",
            "fps=1/5,scale=640:-2",
            "-frames:v",
            "6",
            str(frame_pattern),
        ]
        try:
            result = _run_media_command(frame_cmd, timeout=180)
            frames = sorted(root.glob("frame_*.png"))
            if not frames:
                fallback = root / "frame_001.png"
                result = _run_media_command(
                    [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(path),
                        "-frames:v",
                        "1",
                        str(fallback),
                    ],
                    timeout=180,
                )
                frames = [fallback] if fallback.exists() else []
            if result.returncode != 0 and not frames:
                errors.append(f"video frame extraction failed: {(result.stderr or result.stdout)[:500]}")
            metadata["sampled_frames"] = len(frames)
            for idx, frame in enumerate(frames[:6], start=1):
                child = inspect_file(
                    frame,
                    max_chars=max_chars // 4,
                    archive_depth=0,
                    vision_client=vision_client,
                )
                child.name = f"{path.name} frame {idx}"
                child.path = f"{path}#frame_{idx}"
                child.kind = "video_frame"
                children.append(child)
        except Exception as exc:
            errors.append(f"video frame extraction failed: {exc}")

        wav = root / "extracted_audio.wav"
        try:
            audio_result = _run_media_command(
                [ffmpeg, "-y", "-i", str(path), "-vn", "-ar", "16000", "-ac", "1", str(wav)],
                timeout=180,
            )
            if wav.exists() and wav.stat().st_size > 44:
                audio_text, audio_meta, audio_errors = _inspect_audio(wav, max_chars)
                if audio_text:
                    text_parts.append("# Extracted audio evidence")
                    text_parts.append(audio_text)
                metadata["extracted_audio"] = audio_meta
                errors.extend(audio_errors)
            elif audio_result.returncode != 0:
                metadata["extracted_audio"] = "no readable audio stream extracted"
        except Exception as exc:
            errors.append(f"video audio extraction failed: {exc}")

    if os.getenv("IPW_ENABLE_MEDIA_TRANSCRIPTION", "").lower() not in {"1", "true", "yes"}:
        metadata["transcription"] = "disabled; set IPW_ENABLE_MEDIA_TRANSCRIPTION=1 to enable"

    return _bounded("\n".join(t for t in text_parts if t), max_chars), metadata, errors, children


def inspect_file(
    path: Path,
    *,
    max_chars: int = 16_000,
    archive_depth: int = 1,
    vision_client: Any = None,
) -> FileEvidence:
    """Return bounded evidence for a single file."""
    path = Path(path)
    suffix = path.suffix.lower()
    mime_type = mimetypes.guess_type(str(path))[0]
    stat = path.stat()
    evidence = FileEvidence(
        path=str(path),
        name=path.name,
        suffix=suffix,
        mime_type=mime_type,
        size_bytes=stat.st_size,
        sha256=_sha256(path),
        kind="unknown",
    )

    try:
        if suffix in TEXT_EXTENSIONS:
            evidence.kind = "text"
            evidence.text, evidence.metadata, evidence.errors = _inspect_text(path, max_chars)
        elif suffix in SPREADSHEET_EXTENSIONS:
            evidence.kind = "spreadsheet"
            evidence.text, evidence.metadata, evidence.errors = _inspect_spreadsheet(path, max_chars)
        elif suffix == ".docx":
            evidence.kind = "document"
            evidence.text, evidence.metadata, evidence.errors = _inspect_docx(path, max_chars)
        elif suffix == ".pptx":
            evidence.kind = "presentation"
            evidence.text, evidence.metadata, evidence.errors = _inspect_pptx(path, max_chars)
        elif suffix == ".pdf":
            evidence.kind = "pdf"
            evidence.text, evidence.metadata, evidence.errors = _inspect_pdf(path, max_chars)
        elif suffix == ".ipynb":
            evidence.kind = "notebook"
            evidence.text, evidence.metadata, evidence.errors = _inspect_notebook(path, max_chars)
        elif suffix in IMAGE_EXTENSIONS:
            evidence.kind = "image"
            evidence.text, evidence.metadata, evidence.errors = _inspect_image(
                path,
                max_chars,
                vision_client=vision_client,
            )
        elif suffix in AUDIO_EXTENSIONS:
            evidence.kind = "audio"
            evidence.text, evidence.metadata, evidence.errors = _inspect_audio(path, max_chars)
        elif suffix in VIDEO_EXTENSIONS:
            evidence.kind = "video"
            (
                evidence.text,
                evidence.metadata,
                evidence.errors,
                evidence.children,
            ) = _inspect_video(path, max_chars, vision_client=vision_client)
        elif suffix == ".zip":
            evidence.kind = "archive"
            evidence.metadata = {"archive_type": "zip"}
            if archive_depth > 0:
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    with zipfile.ZipFile(str(path)) as zf:
                        names = zf.namelist()
                        evidence.metadata["entries"] = names[:200]
                        evidence.metadata["entry_count"] = len(names)
                        for member in zf.infolist():
                            target = (root / member.filename).resolve()
                            if not str(target).startswith(str(root.resolve())):
                                evidence.errors.append(f"skipped unsafe archive entry: {member.filename}")
                                continue
                            if member.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(member) as src, target.open("wb") as dst:
                                shutil.copyfileobj(src, dst)
                    for child in sorted(root.rglob("*")):
                        if child.is_file() and len(evidence.children) < 30:
                            evidence.children.append(
                                inspect_file(
                                    child,
                                    max_chars=max_chars // 2,
                                    archive_depth=archive_depth - 1,
                                    vision_client=vision_client,
                                )
                            )
            else:
                with zipfile.ZipFile(str(path)) as zf:
                    names = zf.namelist()
                evidence.text = "\n".join(names[:500])
                evidence.metadata["entry_count"] = len(names)
        else:
            evidence.kind = "binary"
            evidence.errors.append("No specialized parser for this file type; using metadata only")
    except Exception as exc:
        evidence.errors.append(f"file inspection failed: {exc}")

    return evidence


def collect_file_evidence(
    paths: Iterable[str | Path],
    *,
    max_files: int = 60,
    max_chars_per_file: int = 16_000,
    vision_client: Any = None,
) -> list[FileEvidence]:
    """Inspect all existing files in paths, expanding directories recursively."""
    files: list[Path] = []
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        elif path.is_file():
            files.append(path)

    evidence: list[FileEvidence] = []
    seen: set[Path] = set()
    for path in files:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if len(evidence) >= max_files:
            break
        evidence.append(
            inspect_file(
                path,
                max_chars=max_chars_per_file,
                vision_client=vision_client,
            )
        )
    return evidence


def _coerce_path_list(raw: Any) -> list[Path]:
    if not raw:
        return []
    if isinstance(raw, (str, Path)):
        if isinstance(raw, str) and raw.strip().startswith("["):
            try:
                return _coerce_path_list(json.loads(raw))
            except Exception:
                pass
        return [Path(raw)]
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray, dict)):
        return [Path(str(item)) for item in raw if item]
    return []


def _resolve_path_candidates(path: Path, bases: Iterable[Path]) -> list[Path]:
    if path.is_absolute():
        return [path]
    candidates = [base / path for base in bases if base]
    existing = [candidate for candidate in candidates if candidate.exists()]
    return existing or candidates or [path]


def expand_existing_files(paths: Iterable[str | Path]) -> tuple[list[Path], list[Path]]:
    """Expand files/directories and return ``(files, missing_paths)``."""
    files: list[Path] = []
    missing: list[Path] = []
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            missing.append(path)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        try:
            key = path.resolve()
        except Exception:
            key = path
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped, missing


def deliverable_paths_from_metadata(metadata: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    """Return ``(deliverable_paths, reference_paths)`` from GDPval metadata."""
    deliverable_paths: list[Path] = []
    reference_paths: list[Path] = []

    outputs_dir = metadata.get("gdpval_outputs_dir")
    output_bases = _coerce_path_list(outputs_dir) if outputs_dir else []
    submitted_paths: list[Path] = []
    for key in (
        "gdpval_submitted_files",
        "submitted_paths",
        "submitted_files",
        "deliverable_paths",
    ):
        submitted_paths.extend(_coerce_path_list(metadata.get(key)))
    for path in submitted_paths:
        deliverable_paths.extend(_resolve_path_candidates(path, output_bases))
    if outputs_dir:
        deliverable_paths.append(Path(str(outputs_dir)))

    inputs_dir = metadata.get("gdpval_inputs_dir")
    if inputs_dir:
        reference_paths.append(Path(str(inputs_dir)))
    for base in output_bases:
        candidate_inputs = base / "inputs"
        if candidate_inputs.exists():
            reference_paths.append(candidate_inputs)
    for key in ("reference_paths", "input_files", "gdpval_reference_files"):
        reference_paths.extend(_coerce_path_list(metadata.get(key)))

    def dedupe(paths: list[Path]) -> list[Path]:
        out: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
        return out

    return dedupe(deliverable_paths), dedupe(reference_paths)


def infer_expected_extensions(
    *,
    problem: str = "",
    metadata: dict[str, Any] | None = None,
) -> set[str]:
    """Infer expected primary deliverable extensions from metadata and prompt text."""
    metadata = metadata or {}
    expected: set[str] = set()
    for key in ("deliverable_files", "gdpval_expected_files"):
        for item in _coerce_path_list(metadata.get(key)):
            suffix = item.suffix.lower()
            if suffix:
                expected.add(suffix)
    for uri in metadata.get("deliverable_file_hf_uris") or []:
        suffix = Path(str(uri).split("?")[0]).suffix.lower()
        if suffix:
            expected.add(suffix)

    text = "\n".join(
        part
        for part in [
            _task_specific_problem_text(problem or ""),
            str(metadata.get("rubric_pretty") or ""),
            str(metadata.get("rubric_json") or ""),
        ]
        if part
    )
    for pattern, exts in _EXPECTED_EXT_PATTERNS:
        if pattern.search(text):
            expected.update(exts)
    return expected


def _task_specific_problem_text(problem: str) -> str:
    """Return the task section from wrapped GDPval prompts when present."""
    task_match = re.search(r"(?im)^##\s+Task\s*$", problem)
    if not task_match:
        return problem
    task_text = problem[task_match.end() :]
    next_heading = re.search(r"(?im)^##\s+", task_text)
    if next_heading:
        task_text = task_text[: next_heading.start()]
    return task_text.strip() or problem


def _expected_deliverable_names(metadata: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("deliverable_files", "gdpval_expected_files"):
        for item in _coerce_path_list(metadata.get(key)):
            if item.name:
                names.add(unquote(item.name).lower())
    for uri in metadata.get("deliverable_file_hf_uris") or []:
        name = unquote(Path(str(uri).split("?")[0]).name)
        if name:
            names.add(name.lower())
    return names


def _normalized_deliverable_name(name: str) -> str:
    path = Path(unquote(name))
    stem = re.sub(r"[\s_\-.]+", " ", path.stem.lower()).strip()
    suffix = path.suffix.lower()
    return f"{stem}{suffix}"


def _has_explicit_submission(metadata: dict[str, Any]) -> bool:
    return any(
        bool(_coerce_path_list(metadata.get(key)))
        for key in (
            "gdpval_submitted_files",
            "submitted_paths",
            "submitted_files",
            "deliverable_paths",
        )
    )


def _is_ignored_output_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return (
        path.name in IGNORED_OUTPUT_FILENAMES
        or bool(parts & AUXILIARY_OUTPUT_DIR_NAMES)
    )


def _is_auxiliary_fallback_candidate(path: Path) -> bool:
    """Return True for scratch/reference files found by workspace scanning.

    These names are allowed when explicitly submitted, but they should not let
    a no-finish/no-deliverable run pass deterministic validation.
    """
    stem = path.stem.lower()
    if path.suffix.lower() in VIDEO_EXTENSIONS and re.match(
        r"^(?:clip[_\-.]?\d*|test(?:[_\-.]|$))", stem
    ):
        return True
    return bool(AUXILIARY_OUTPUT_STEM_PATTERN.search(stem))


def _resolved_path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _file_content_signature(path: Path) -> tuple[int, str] | None:
    try:
        return (path.stat().st_size, _sha256(path))
    except Exception:
        return None


def _evidence_path_key(item: FileEvidence) -> str:
    raw_path = item.path.split("#", 1)[0]
    return _resolved_path_key(Path(raw_path))


def validate_deliverable_contract(
    *,
    paths: Iterable[str | Path],
    problem: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic checks before LLM judging.

    This does not decide task quality. It classifies harness-level failures
    such as missing files, only scripts/research being submitted, unreadable
    files, or expected file types not appearing.
    """
    metadata = metadata or {}
    files, missing = expand_existing_files(paths)
    if not files and not missing:
        metadata_paths, _ = deliverable_paths_from_metadata(metadata)
        files, missing = expand_existing_files(metadata_paths)
    _, reference_paths = deliverable_paths_from_metadata(metadata)
    reference_files, _ = expand_existing_files(reference_paths)
    reference_file_keys = {_resolved_path_key(path) for path in reference_files}
    reference_file_signatures = {
        signature
        for path in reference_files
        if (signature := _file_content_signature(path)) is not None
    }
    expected_extensions = infer_expected_extensions(problem=problem, metadata=metadata)
    expected_names = _expected_deliverable_names(metadata)
    has_explicit_submission = _has_explicit_submission(metadata)

    issues: list[dict[str, Any]] = []
    if missing:
        issues.append(
            {
                "code": "submitted_file_missing",
                "paths": [str(path) for path in missing],
            }
        )
    output_files = [
        path
        for path in files
        if not _is_ignored_output_path(path)
        and _resolved_path_key(path) not in reference_file_keys
        and _file_content_signature(path) not in reference_file_signatures
    ]
    if not output_files:
        issues.append({"code": "no_deliverables_found"})
    primary_files = [
        path for path in output_files if path.suffix.lower() in PRIMARY_DELIVERABLE_EXTENSIONS
    ]
    candidate_primary = [
        path
        for path in primary_files
        if path.suffix.lower() not in SUPPORTING_ONLY_EXTENSIONS
        or path.suffix.lower() in expected_extensions
    ]
    discarded_candidates: list[dict[str, str]] = []
    if expected_extensions:
        filtered_primary: list[Path] = []
        fallback_primary: list[Path] = []
        normalized_expected_names = {
            _normalized_deliverable_name(name) for name in expected_names
        }
        for path in candidate_primary:
            suffix = path.suffix.lower()
            if suffix not in expected_extensions:
                continue
            normalized_path_name = _normalized_deliverable_name(path.name)
            if (
                not has_explicit_submission
                and normalized_path_name not in normalized_expected_names
                and _is_auxiliary_fallback_candidate(path)
            ):
                discarded_candidates.append(
                    {
                        "file": path.name,
                        "reason": "workspace fallback candidate looks like reference or scratch output",
                    }
                )
                continue
            if (
                expected_names
                and not has_explicit_submission
                and normalized_path_name not in normalized_expected_names
            ):
                discarded_candidates.append(
                    {
                        "file": path.name,
                        "reason": "workspace fallback filename differed from expected deliverable filename",
                    }
                )
                fallback_primary.append(path)
                continue
            filtered_primary.append(path)
        if not filtered_primary and fallback_primary:
            issues.append(
                {
                    "code": "expected_named_file_mismatch",
                    "files": [path.name for path in fallback_primary[:50]],
                    "expected_filenames": sorted(expected_names),
                    "discarded_candidates": discarded_candidates,
                }
            )
            filtered_primary = fallback_primary
        non_supporting_primary = filtered_primary
    else:
        non_supporting_primary = candidate_primary

    if expected_names and not has_explicit_submission:
        normalized_expected_names = {
            _normalized_deliverable_name(name) for name in expected_names
        }
        normalized_candidate_names = {
            _normalized_deliverable_name(path.name)
            for path in candidate_primary
        }
        expected_suffixes = {
            Path(name).suffix.lower()
            for name in expected_names
            if Path(name).suffix
        }
        present_candidate_suffixes = {
            path.suffix.lower()
            for path in candidate_primary
            if path.suffix
        }
        missing_expected_suffixes = sorted(
            suffix
            for suffix in expected_suffixes
            if suffix and suffix not in present_candidate_suffixes
        )
        if (
            normalized_expected_names
            and normalized_candidate_names.isdisjoint(normalized_expected_names)
            and not bool(expected_suffixes & present_candidate_suffixes)
        ):
            issues.append(
                {
                    "code": "expected_named_file_missing",
                    "files": [path.name for path in candidate_primary[:50]],
                    "expected_filenames": sorted(expected_names),
                    "expected_suffixes": sorted(expected_suffixes),
                    "present_extensions": sorted(present_candidate_suffixes),
                    "discarded_candidates": discarded_candidates,
                }
            )
        elif missing_expected_suffixes and not any(
            issue.get("code") == "expected_named_file_missing" for issue in issues
        ):
            issues.append(
                {
                    "code": "expected_named_file_missing",
                    "files": [path.name for path in candidate_primary[:50]],
                    "expected_filenames": sorted(expected_names),
                    "expected_suffixes": sorted(expected_suffixes),
                    "missing_expected_suffixes": missing_expected_suffixes,
                    "present_extensions": sorted(present_candidate_suffixes),
                    "discarded_candidates": discarded_candidates,
                }
            )

    if (
        expected_extensions
        and output_files
        and primary_files
        and not non_supporting_primary
        and not any(issue.get("code") == "expected_named_file_missing" for issue in issues)
    ):
        issues.append(
            {
                "code": (
                    "expected_named_file_missing"
                    if expected_names and not has_explicit_submission
                    else "only_supporting_files"
                ),
                "files": [path.name for path in output_files[:50]],
                "expected_filenames": sorted(expected_names),
                "discarded_candidates": discarded_candidates,
            }
        )

    if expected_extensions and output_files:
        present = {
            path.suffix.lower()
            for path in primary_files
            if path.suffix.lower() not in SUPPORTING_ONLY_EXTENSIONS
            or path.suffix.lower() in expected_extensions
        }
        if not (present & expected_extensions):
            issues.append(
                {
                    "code": "expected_file_type_missing",
                    "expected_extensions": sorted(expected_extensions),
                    "present_extensions": sorted(ext for ext in present if ext),
                }
            )

    evidence = collect_file_evidence(output_files)
    all_evidence = list(_walk_evidence(evidence))
    inspection_errors = [
        {"file": item.name, "errors": item.errors}
        for item in all_evidence
        if item.errors and item.kind not in {"audio", "video"}
    ]
    if inspection_errors:
        issues.append({"code": "file_inspection_errors", "files": inspection_errors})
    unreadable_archives = [
        {"file": item.name, "errors": item.errors}
        for item in all_evidence
        if item.errors and item.kind == "archive"
    ]
    if unreadable_archives:
        issues.append({"code": "unreadable_archive", "files": unreadable_archives})
    unreadable_deliverables = _unreadable_deliverable_issues(
        all_evidence,
        primary_path_keys={
            _resolved_path_key(path)
            for path in non_supporting_primary
        },
    )
    if unreadable_deliverables:
        issues.append({"code": "unreadable_deliverable", "files": unreadable_deliverables})
    media_issues = _media_validation_issues(
        all_evidence,
        primary_path_keys={
            _resolved_path_key(path)
            for path in non_supporting_primary
        },
    )
    if media_issues["unavailable"]:
        issues.append({"code": "media_inspection_unavailable", "files": media_issues["unavailable"]})
    if media_issues["invalid"]:
        issues.append({"code": "invalid_media", "files": media_issues["invalid"]})

    blocking_codes = {
        "submitted_file_missing",
        "no_deliverables_found",
        "only_supporting_files",
        "expected_file_type_missing",
        "unreadable_archive",
        "unreadable_deliverable",
        "media_inspection_unavailable",
        "invalid_media",
    }
    status = "ok"
    if any(issue["code"] in blocking_codes for issue in issues):
        status = "invalid"
    elif (
        not non_supporting_primary
        and any(issue["code"] == "expected_named_file_missing" for issue in issues)
    ):
        status = "invalid"
    elif issues:
        status = "warning"

    return {
        "status": status,
        "issues": issues,
        "file_count": len(output_files),
        "files": [str(path) for path in output_files],
        "primary_files": [str(path) for path in non_supporting_primary],
        "expected_extensions": sorted(expected_extensions),
        "expected_filenames": sorted(expected_names),
        "explicit_submission": has_explicit_submission,
    }


def _walk_evidence(items: Iterable[FileEvidence]) -> Iterable[FileEvidence]:
    for item in items:
        yield item
        yield from _walk_evidence(item.children)


def _media_validation_issues(
    items: Iterable[FileEvidence],
    *,
    primary_path_keys: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    issues: dict[str, list[dict[str, Any]]] = {"unavailable": [], "invalid": []}
    videos: list[FileEvidence] = []
    for item in items:
        if item.kind != "video":
            continue
        if primary_path_keys is not None and _evidence_path_key(item) not in primary_path_keys:
            continue
        videos.append(item)

    valid_video_present = any(
        not str(item.metadata.get("video_decoder") or "").startswith("unavailable")
        and int(item.metadata.get("sampled_frames") or 0) > 0
        for item in videos
    )
    if valid_video_present:
        return issues

    for item in videos:
        if str(item.metadata.get("video_decoder") or "").startswith("unavailable"):
            issues["unavailable"].append(
                {
                    "file": item.name,
                    "reason": "video decoder unavailable; install ffmpeg or imageio-ffmpeg",
                }
            )
        elif int(item.metadata.get("sampled_frames") or 0) <= 0:
            detail = "no video frames could be decoded from the submitted media"
            if item.errors:
                detail += f"; extraction errors: {'; '.join(item.errors)[:500]}"
            elif item.metadata.get("ffprobe_error"):
                detail += f"; ffprobe error: {str(item.metadata['ffprobe_error'])[:500]}"
            issues["invalid"].append(
                {
                    "file": item.name,
                    "reason": detail,
                }
            )
    return issues


_STRUCTURED_DELIVERABLE_KINDS = {
    "pdf",
    "document",
    "presentation",
    "spreadsheet",
    "notebook",
}

_CORRUPT_FILE_ERROR_PATTERN = re.compile(
    r"not a (?:pdf|zip)|"
    r"no /root object|"
    r"couldn'?t (?:find|read)|"
    r"stream has ended unexpectedly|"
    r"bad zip file|"
    r"file is not a zip file|"
    r"parse error",
    re.I,
)

_PARSER_UNAVAILABLE_ERROR_PATTERN = re.compile(
    r"is not installed|"
    r"no such file or directory",
    re.I,
)


def _unreadable_deliverable_issues(
    items: Iterable[FileEvidence],
    *,
    primary_path_keys: set[str],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not primary_path_keys:
        return issues
    for item in items:
        if item.kind not in _STRUCTURED_DELIVERABLE_KINDS:
            continue
        if _evidence_path_key(item) not in primary_path_keys:
            continue
        if not item.errors:
            continue
        if item.text.strip():
            continue
        error_text = "\n".join(str(error) for error in item.errors)
        if (
            _PARSER_UNAVAILABLE_ERROR_PATTERN.search(error_text)
            and not _CORRUPT_FILE_ERROR_PATTERN.search(error_text)
        ):
            continue
        if not _CORRUPT_FILE_ERROR_PATTERN.search(error_text):
            continue
        issues.append(
            {
                "file": item.name,
                "kind": item.kind,
                "errors": item.errors,
                "reason": "primary deliverable could not be parsed as its declared file type",
            }
        )
    return issues


def find_visual_evidence_gaps(evidence: Iterable[FileEvidence]) -> list[dict[str, Any]]:
    """Return visual files/frames that have no semantic evidence for a text judge."""
    gaps: list[dict[str, Any]] = []
    for item in _walk_evidence(evidence):
        if item.kind not in {"image", "video_frame"}:
            continue
        if item.metadata.get("vision_description") == "included":
            continue
        if item.text.strip():
            continue
        gaps.append(
            {
                "file": item.name,
                "kind": item.kind,
                "reason": "no OCR text or vision description available",
            }
        )
    return gaps


def render_evidence_sections(title: str, evidence: list[FileEvidence], *, max_chars: int = 80_000) -> str:
    """Render evidence as a bounded prompt section."""
    parts = [f"## {title}"]
    for item in evidence:
        parts.append(item.to_prompt_section())
    rendered = "\n\n".join(parts)
    return _bounded(rendered, max_chars)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from a judge response."""
    if not text:
        return None
    stripped = text.strip()

    fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    for block in fenced_blocks:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            parsed, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


__all__ = [
    "FileEvidence",
    "collect_file_evidence",
    "deliverable_paths_from_metadata",
    "extract_json_object",
    "expand_existing_files",
    "find_visual_evidence_gaps",
    "infer_expected_extensions",
    "inspect_file",
    "render_evidence_sections",
    "validate_deliverable_contract",
]
