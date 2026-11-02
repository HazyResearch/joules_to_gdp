"""GDPval evaluation: rubric-based LLM-as-judge.

GDPval tasks have no reference answers - each task ships a ``rubric_json``
listing criteria with point values. We ask the judge model to score the
candidate answer against each criterion, then aggregate:

- A criterion is satisfied iff the judge returns "yes".
- Score = (achieved points) / (max points).
- ``is_correct`` is True iff every ``required: true`` criterion is satisfied
  AND the achieved score is at least the configured pass threshold (default
  0.5).
- If the rubric is malformed or no criteria are present, returns ``None``
  (unscorable).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.registry import EvaluationRegistry
from .base import EvaluationHandler
from .deliverables import (
    collect_file_evidence,
    deliverable_paths_from_metadata,
    find_visual_evidence_gaps,
    render_evidence_sections,
    validate_deliverable_contract,
)

LOGGER = logging.getLogger(__name__)

_PASS_THRESHOLD = 0.5
_OUTPUTS_BUDGET_CHARS = 12_000  # per-file budget when summarising deliverables
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_OFFICE_EXTS = {".xlsx", ".xls", ".xlsm", ".docx", ".pptx"}


def _describe_image_with_vision(path: Path, client) -> str:
    """One-shot image describer using a vision-capable judge model.

    Encodes the image as base64 and asks the judge to produce a detailed
    factual description. The result is reused across every criterion call
    for this deliverable, so cost is O(1 vision call per image), not
    O(n_criteria).
    """
    if not hasattr(client, "_chat_completion"):
        return "<vision unavailable: client lacks _chat_completion>"
    try:
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as exc:
        return f"<image read error: {exc}>"
    suf = path.suffix.lower().lstrip(".")
    if suf == "jpg":
        suf = "jpeg"
    data_url = f"data:image/{suf};base64,{b64}"
    messages = [
        {"role": "system", "content": "You are a careful visual describer."},
        {"role": "user", "content": [
            {"type": "text", "text": (
                "Describe this image factually in 4-8 sentences. Cover: "
                "subject, layout/composition, any visible text or labels, "
                "colour palette, and notable details. Be specific and "
                "literal - no speculation about purpose."
            )},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    try:
        resp = client._chat_completion(client.model, "", messages=messages)
        return (resp.content or "").strip() or "<empty vision response>"
    except Exception as exc:
        return f"<vision call failed: {exc}>"


def _render_office_to_pdf(src: Path) -> Optional[Path]:
    """Render .docx/.xlsx/.pptx to .pdf using libreoffice headless.

    Returns path to the generated PDF, or None on failure. The PDF
    captures charts, conditional formatting, page layout, embedded
    images etc. that python-docx / openpyxl / python-pptx skip.
    """
    if not src.exists() or src.suffix.lower() not in _OFFICE_EXTS:
        return None
    if not (Path("/usr/bin/soffice").exists() or Path("/usr/bin/libreoffice").exists()):
        return None
    out_dir = Path(tempfile.mkdtemp(prefix="lo_render_"))
    try:
        subprocess.run(
            ["soffice", "--headless", "--norestore", "--convert-to", "pdf",
             "--outdir", str(out_dir), str(src)],
            check=False, capture_output=True, timeout=90,
        )
    except Exception:
        return None
    pdf = out_dir / (src.stem + ".pdf")
    return pdf if pdf.exists() else None

_JUDGE_PROMPT = """You are grading a candidate submission against a single rubric criterion.

Use the submitted deliverable-file evidence as the primary source of truth when
it is present. The plain-text answer may contain harness control text such as
"Submit your final answer now..." or repair/checklist scaffolding; ignore that
control text instead of treating it as the work product.

## Task given to the candidate
{problem}

## Candidate response / finish summary
{response}
{deliverables_section}
## Rubric criterion
{criterion}

## Criterion interpretation
{criterion_guidance}

Does the candidate submission satisfy this criterion?

Reply with exactly one line in this format:
verdict: <yes or no>
reason: <one short sentence>
"""


def _criterion_guidance(
    *,
    criterion_text: str,
    positive_points: float,
    penalty_points: float,
) -> str:
    """Return judge guidance for positive, penalty, and conditional criteria."""
    guidance = [
        "Prefer concrete evidence from opened/parsed files over the candidate's prose summary.",
        "For file type, filename, sheet/tab, page, slide, stream, or archive checks, "
        "the deliverable evidence and metadata are authoritative.",
        "For conditional criteria using wording such as 'if', 'when', or 'where', "
        "answer yes when the condition is not triggered by the task/submission unless "
        "the criterion explicitly requires proving that condition.",
    ]
    if penalty_points > 0 and positive_points == 0:
        guidance.append(
            "This is a negative-point penalty criterion. Answer yes only if the "
            "defect described by the criterion is actually present and the penalty "
            "should be applied. If the submission avoids the defect, answer no."
        )
    else:
        guidance.append(
            "This is a positive-point criterion. Answer yes only if the requested "
            "requirement is satisfied by the response or submitted files."
        )
    if re.search(r"\b(?:fail(?:s|ed)?|missing|omit(?:s|ted)?|lacks?|empty|incorrect)\b", criterion_text, re.I):
        guidance.append(
            "Be especially careful with negatively worded criteria: a reason saying "
            "the submission does not fail, does not omit, or does include the "
            "requested item should normally correspond to verdict: no for a penalty."
        )
    return "\n".join(f"- {line}" for line in guidance)


def _extract_deliverable_text(outputs_dir: Path, vision_client: Any = None) -> str:
    """Return a concatenated text view of any deliverable files.

    Used so the judge can verify rubric criteria that reference produced
    files (e.g. ``Workbook contains tab 'Sample'``).

    Enrichments:

    - **LibreOffice rendering**: for .xlsx/.docx/.pptx, additionally render
      to PDF via headless soffice and extract that text; catches charts,
      conditional formatting, page layout, and embedded images that the
      Python office libraries silently drop.
    - **Vision-based image description**: when ``vision_client`` is
      supplied (a client whose model accepts image_url content), every
      image deliverable gets a 4-8 sentence factual description appended
      to its text section.
    """
    if not outputs_dir or not Path(outputs_dir).is_dir():
        return ""
    sections: list[str] = []
    for path in sorted(Path(outputs_dir).iterdir()):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        try:
            if suf in (".txt", ".md", ".csv", ".tsv", ".json"):
                text = path.read_text(encoding="utf-8", errors="ignore")
            elif suf in (".xlsx", ".xls", ".xlsm"):
                try:
                    from openpyxl import load_workbook
                except ImportError:
                    text = "<openpyxl not installed>"
                else:
                    wb = load_workbook(filename=str(path), data_only=True, read_only=True)
                    parts: list[str] = []
                    for sheet in wb.worksheets:
                        parts.append(f"# Sheet: {sheet.title}")
                        for row in sheet.iter_rows(values_only=True):
                            cells = [("" if v is None else str(v)) for v in row]
                            parts.append("\t".join(cells))
                    text = "\n".join(parts)
                # Also render to PDF via libreoffice to catch charts /
                # conditional formatting / page layout that openpyxl drops.
                pdf = _render_office_to_pdf(path)
                if pdf is not None:
                    try:
                        import pdfplumber
                        pages = []
                        with pdfplumber.open(str(pdf)) as p:
                            for page in p.pages:
                                pages.append(page.extract_text() or "")
                        rendered = "\n\n".join(pages).strip()
                        if rendered:
                            text += "\n\n## Rendered (libreoffice PDF):\n" + rendered
                    except Exception:
                        pass
            elif suf == ".pdf":
                try:
                    import pdfplumber
                except ImportError:
                    text = "<pdfplumber not installed>"
                else:
                    pages = []
                    with pdfplumber.open(str(path)) as pdf:
                        for page in pdf.pages:
                            pages.append(page.extract_text() or "")
                    text = "\n\n".join(pages)
            elif suf == ".docx":
                try:
                    import docx
                except ImportError:
                    text = "<python-docx not installed>"
                else:
                    text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
                pdf = _render_office_to_pdf(path)
                if pdf is not None:
                    try:
                        import pdfplumber
                        with pdfplumber.open(str(pdf)) as p:
                            rendered = "\n\n".join(
                                page.extract_text() or "" for page in p.pages
                            ).strip()
                        if rendered:
                            text += "\n\n## Rendered (libreoffice PDF):\n" + rendered
                    except Exception:
                        pass
            elif suf == ".pptx":
                try:
                    from pptx import Presentation
                except ImportError:
                    text = "<python-pptx not installed>"
                else:
                    prs = Presentation(str(path))
                    parts = []
                    for i, slide in enumerate(prs.slides):
                        parts.append(f"## Slide {i + 1}")
                        for shape in slide.shapes:
                            if hasattr(shape, "text") and shape.text:
                                parts.append(shape.text)
                    text = "\n".join(parts)
                pdf = _render_office_to_pdf(path)
                if pdf is not None:
                    try:
                        import pdfplumber
                        with pdfplumber.open(str(pdf)) as p:
                            rendered = "\n\n".join(
                                page.extract_text() or "" for page in p.pages
                            ).strip()
                        if rendered:
                            text += "\n\n## Rendered (libreoffice PDF):\n" + rendered
                    except Exception:
                        pass
            elif suf == ".ipynb":
                try:
                    nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                    parts = []
                    for i, cell in enumerate(nb.get("cells") or []):
                        src = cell.get("source")
                        if isinstance(src, list):
                            src = "".join(src)
                        parts.append(f"## Cell {i + 1} ({cell.get('cell_type', '?')})\n{src or ''}")
                    text = "\n\n".join(parts)
                except Exception as exc:
                    text = f"<ipynb parse error: {exc}>"
            elif suf == ".zip":
                try:
                    import zipfile
                    with zipfile.ZipFile(str(path)) as z:
                        names = z.namelist()
                    text = "Archive contents:\n" + "\n".join(
                        f"  {n}" for n in names[:200]
                    ) + (f"\n  ... [{len(names) - 200} more]" if len(names) > 200 else "")
                except Exception as exc:
                    text = f"<zip read error: {exc}>"
            elif suf in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
                try:
                    from PIL import Image
                    img = Image.open(str(path))
                    text = f"Image file. format={img.format} mode={img.mode} size={img.size}"
                except Exception as exc:
                    text = f"<image read error: {exc}>"
                # When a vision-capable client is available, ask it to
                # describe the image so the rubric judge can evaluate visual
                # criteria (composition, labels, color, etc.) using text.
                if vision_client is not None:
                    desc = _describe_image_with_vision(path, vision_client)
                    if desc:
                        text += "\n\n## Vision description:\n" + desc
            elif suf in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
                # Transcribe with whisper (~base model). Limited to first
                # ~30 s of audio to keep grade time bounded.
                try:
                    import whisper
                    model = whisper.load_model("base")
                    result = model.transcribe(str(path), verbose=False)
                    transcript = (result.get("text") or "")[:_OUTPUTS_BUDGET_CHARS]
                    text = f"Audio transcript:\n{transcript}"
                except Exception as exc:
                    text = f"<audio transcribe error: {exc}>"
            elif suf == ".mp4":
                # Extract audio with ffmpeg, transcribe with whisper
                import subprocess
                import tempfile
                wav = path.with_suffix(".wav")
                try:
                    if not wav.exists():
                        wav = Path(tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        ).name)
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", str(path), "-vn", "-ar", "16000",
                             "-ac", "1", str(wav)],
                            capture_output=True, check=True, timeout=120,
                        )
                    import whisper
                    model = whisper.load_model("base")
                    result = model.transcribe(str(wav), verbose=False)
                    transcript = (result.get("text") or "")[:_OUTPUTS_BUDGET_CHARS]
                    text = f"Video file (audio transcribed):\n{transcript}"
                except Exception as exc:
                    text = f"<video transcribe error: {exc}>"
            elif suf == ".psd":
                try:
                    from psd_tools import PSDImage
                    psd = PSDImage.open(str(path))
                    text = (
                        f"PSD file. size={psd.size} layers="
                        + ",".join(layer.name for layer in psd.descendants() if layer.name)[:1000]
                    )
                except Exception as exc:
                    text = f"<psd read error: {exc}>"
            elif suf == ".step":
                # CAD file. No installable reader on this box; describe only.
                text = f"STEP CAD file. size={path.stat().st_size}B  (binary CAD content not parsed)"
            elif suf == ".py":
                # Treat py deliverables like text (judge will read the source)
                text = path.read_text(encoding="utf-8", errors="ignore")
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            text = f"<error reading {path.name}: {exc}>"

        if len(text) > _OUTPUTS_BUDGET_CHARS:
            text = text[:_OUTPUTS_BUDGET_CHARS] + "\n... [truncated]"
        sections.append(f"### Deliverable file: {path.name}\n{text}")
    return "\n\n".join(sections)


def _parse_rubric(raw: Any) -> List[Dict[str, Any]]:
    """Return the list of rubric items from a rubric_json blob."""
    if not raw:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            # Some rubrics wrap the list under a key
            items = (
                parsed.get("rubric")
                or parsed.get("criteria")
                or parsed.get("items")
                or []
            )
        else:
            return []
    else:
        return []

    cleaned: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and item.get("criterion"):
            cleaned.append(item)
    return cleaned


def _judge_verdict(raw: str) -> bool:
    """Return True iff the judge said 'verdict: yes'."""
    if not raw:
        return False
    m = re.search(r"verdict\s*:\s*(yes|no)", raw, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "yes"
    # Fallback: any standalone yes/no token in the first line
    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    return bool(re.search(r"\byes\b", first_line, re.IGNORECASE))


def _rubric_points(raw: Any) -> tuple[float, float]:
    """Return ``(positive_points, penalty_points)`` for a rubric item.

    GDPval rubrics occasionally encode anti-criteria with negative point
    values. Treat those as penalties when the judge says the anti-criterion is
    present, rather than adding negative values to the denominator.
    """
    try:
        points = float(raw)
    except (TypeError, ValueError):
        points = 1.0
    if points < 0:
        return 0.0, abs(points)
    return points, 0.0


def _rubric_totals(rubric: list[dict[str, Any]]) -> tuple[float, float, int]:
    positive_possible = 0.0
    penalty_possible = 0.0
    required_count = 0
    for item in rubric:
        positive_points, penalty_points = _rubric_points(item.get("score", 1.0))
        positive_possible += positive_points
        penalty_possible += penalty_points
        if item.get("required"):
            required_count += 1
    return positive_possible, penalty_possible, required_count


@EvaluationRegistry.register("gdpval")
class GdpvalHandler(EvaluationHandler):
    """Rubric-based LLM-as-judge grading for GDPval tasks."""

    evaluation_method = "gdpval"

    def evaluate(
        self,
        *,
        problem: str,
        reference: str,
        model_answer: str,
        metadata: Dict[str, object],
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        metadata_dict = dict(metadata)
        deliverable_paths, _reference_paths = deliverable_paths_from_metadata(
            metadata_dict
        )
        has_deliverable_context = bool(
            deliverable_paths
            or metadata_dict.get("deliverable_files")
            or metadata_dict.get("deliverable_file_hf_uris")
            or metadata_dict.get("gdpval_outputs_dir")
        )
        synthesized_empty_response = False
        if not model_answer or not model_answer.strip():
            if not has_deliverable_context:
                return False, {"reason": "empty_response"}
            synthesized_empty_response = True
            model_answer = (
                "No textual finish response was captured. Judge the submitted "
                "GDPval deliverable files using the deterministic validation "
                "summary and file evidence below."
            )

        rubric_raw = metadata_dict.get("rubric_json")
        rubric = _parse_rubric(rubric_raw)
        if not rubric:
            return None, {"reason": "no_rubric"}

        if not hasattr(self._client, "chat"):
            return None, {"reason": "no_llm_client_for_judging"}

        should_validate_deliverables = bool(
            deliverable_paths
            or metadata_dict.get("deliverable_files")
            or metadata_dict.get("deliverable_file_hf_uris")
        )
        deliverable_validation: dict[str, Any] | None = None
        if should_validate_deliverables:
            deliverable_validation = validate_deliverable_contract(
                paths=deliverable_paths,
                problem=problem,
                metadata=metadata_dict,
            )
            if deliverable_validation["status"] == "invalid":
                positive_possible, penalty_possible, required_count = _rubric_totals(rubric)
                return False, {
                    "reason": "deliverable_contract_invalid",
                    "failure_type": "harness_validation",
                    "deliverable_validation": deliverable_validation,
                    "score": 0.0,
                    "achieved_points": 0.0,
                    "max_points": positive_possible,
                    "positive_possible": positive_possible,
                    "penalty_possible": penalty_possible,
                    "required_failures": required_count,
                    "num_criteria": len(rubric),
                    "pass_threshold": _PASS_THRESHOLD,
                    "synthesized_empty_response": synthesized_empty_response,
                }

        judged_deliverable_paths = (
            deliverable_validation.get("primary_files")
            or deliverable_validation.get("files", [])
            if deliverable_validation is not None
            else deliverable_paths
        )
        deliverable_evidence = (
            collect_file_evidence(judged_deliverable_paths, vision_client=self._client)
            if judged_deliverable_paths
            else []
        )
        deliverable_text = (
            render_evidence_sections("Deliverable Files Produced", deliverable_evidence)
            if deliverable_evidence
            else ""
        )
        visual_gaps = find_visual_evidence_gaps(deliverable_evidence)
        if visual_gaps:
            return None, {
                "reason": "visual_evidence_unavailable",
                "failure_type": "harness_validation",
                "visual_evidence_gaps": visual_gaps,
                "deliverable_validation": deliverable_validation,
                "deliverables": [item.to_dict() for item in deliverable_evidence],
            }
        validation_summary = ""
        if deliverable_validation is not None:
            validation_summary = json.dumps(
                {
                    key: deliverable_validation.get(key)
                    for key in (
                        "status",
                        "primary_files",
                        "expected_filenames",
                        "expected_extensions",
                        "explicit_submission",
                    )
                },
                ensure_ascii=False,
            )
        deliverables_section = ""
        if validation_summary:
            deliverables_section += (
                "\n## Deterministic deliverable validation summary\n"
                f"{validation_summary}\n"
            )
        if deliverable_text:
            deliverables_section += f"\n{deliverable_text}\n"

        per_criterion: list[dict[str, Any]] = []
        positive_awarded = 0.0
        penalty_applied = 0.0
        positive_possible = 0.0
        penalty_possible = 0.0
        required_failures = 0

        for idx, item in enumerate(rubric):
            criterion_text = str(item.get("criterion") or "").strip()
            if not criterion_text:
                continue

            raw_points = item.get("score", 1.0)
            positive_points, penalty_points = _rubric_points(raw_points)
            required = bool(item.get("required", False))
            rubric_item_id = item.get("rubric_item_id", f"item_{idx}")

            positive_possible += positive_points
            penalty_possible += penalty_points

            prompt = _JUDGE_PROMPT.format(
                problem=problem[:4000],  # keep judge prompt bounded
                response=model_answer[:8000],
                deliverables_section=deliverables_section,
                criterion=criterion_text,
                criterion_guidance=_criterion_guidance(
                    criterion_text=criterion_text,
                    positive_points=positive_points,
                    penalty_points=penalty_points,
                ),
            )
            max_judge_attempts = max(
                1,
                int(os.getenv("IPW_GDPVAL_JUDGE_MAX_ATTEMPTS", "3")),
            )
            retry_seconds = max(
                0.0,
                float(os.getenv("IPW_GDPVAL_JUDGE_RETRY_SECONDS", "2")),
            )
            raw = ""
            judge_error: Exception | None = None
            for attempt in range(1, max_judge_attempts + 1):
                try:
                    raw = self._client.chat(
                        system_prompt="",
                        user_prompt=prompt,
                        temperature=0.0,
                        max_output_tokens=512,
                    )
                    judge_error = None
                    break
                except Exception as exc:
                    judge_error = exc
                    if attempt < max_judge_attempts:
                        time.sleep(retry_seconds * attempt)
            if judge_error is not None:
                if os.getenv("IPW_REQUIRE_GDPVAL_JUDGE") == "1":
                    raise RuntimeError(
                        f"GDPVal rubric judge failed for item {rubric_item_id}"
                    ) from judge_error
                LOGGER.warning(
                    "Rubric judge failed for item %s after %d attempts: %s",
                    rubric_item_id,
                    max_judge_attempts,
                    judge_error,
                )
                per_criterion.append(
                    {
                        "rubric_item_id": rubric_item_id,
                        "criterion": criterion_text,
                        "points": raw_points,
                        "positive_points": positive_points,
                        "penalty_points": penalty_points,
                        "required": required,
                        "satisfied": False,
                        "error": str(judge_error),
                    }
                )
                if required:
                    required_failures += 1
                continue

            if os.getenv("IPW_REQUIRE_GDPVAL_JUDGE") == "1":
                verdicts = re.findall(
                    r"^\s*verdict\s*:\s*(yes|no)\s*$", raw,
                    re.IGNORECASE | re.MULTILINE,
                )
                if len(verdicts) != 1:
                    raise RuntimeError(
                        f"GDPVal rubric judge returned an invalid verdict for item {rubric_item_id}"
                    )
                satisfied = verdicts[0].lower() == "yes"
            else:
                satisfied = _judge_verdict(raw)
            if satisfied and positive_points:
                positive_awarded += positive_points
            if satisfied and penalty_points:
                penalty_applied += penalty_points
            if not satisfied and required:
                required_failures += 1

            per_criterion.append(
                {
                    "rubric_item_id": rubric_item_id,
                    "criterion": criterion_text,
                    "points": raw_points,
                    "positive_points": positive_points,
                    "penalty_points": penalty_points,
                    "required": required,
                    "satisfied": satisfied,
                    "judge_output": raw[:500],
                }
            )

        achieved_points = positive_awarded - penalty_applied
        raw_score = (
            achieved_points / positive_possible
            if positive_possible > 0
            else 0.0
        )
        score = max(0.0, min(1.0, raw_score))
        is_correct = required_failures == 0 and score >= _PASS_THRESHOLD

        return is_correct, {
            "match_type": "rubric_llm_judge",
            "score": score,
            "raw_score": raw_score,
            "achieved_points": achieved_points,
            "positive_awarded": positive_awarded,
            "penalty_applied": penalty_applied,
            "max_points": positive_possible,
            "positive_possible": positive_possible,
            "penalty_possible": penalty_possible,
            "required_failures": required_failures,
            "num_criteria": len(per_criterion),
            "per_criterion": per_criterion,
            "pass_threshold": _PASS_THRESHOLD,
            "deliverable_validation": deliverable_validation,
            "deliverables": [item.to_dict() for item in deliverable_evidence],
            "synthesized_empty_response": synthesized_empty_response,
        }


__all__ = ["GdpvalHandler"]
