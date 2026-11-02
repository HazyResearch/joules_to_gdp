"""GDPval dataset (openai/gdpval).

Real-world professional tasks across 44 occupations. Each task has:
- A long-form prompt
- A set of reference files (Excel/PDF/Word) the agent should read
- A set of deliverable files the model is expected to produce
- A machine-readable rubric (rubric_json) for grading

This provider downloads reference files into a per-task cache, augments the
prompt with on-disk paths, and routes scoring to the ``gdpval`` evaluation
handler (rubric-based LLM-as-judge).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import unquote

from huggingface_hub import hf_hub_download

from datasets import load_dataset

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry, EvaluationRegistry
from ..core.types import DatasetRecord
from .base import DatasetProvider

LOGGER = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "gdpval"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_reference_staging() -> bool:
    return os.getenv("IPW_REQUIRE_GDPVAL_REFERENCES") == "1"

_PROMPT_TEMPLATE = """You are a domain expert in: {occupation} ({sector}).

{rescue_section}

Complete the following professional task. Use any reference files provided \
and produce a high-quality answer that a professional in this role would \
accept as a deliverable.

{file_section}
{deliverable_section}
{rubric_section}

## Task

{task_prompt}

## Tools and workflow

{tool_section}

This task asks you to read the reference files (if any), produce one or more \
output files in the current directory, and call `finish` describing what you \
produced.

Available system tools and Python libraries (installed and ready):

- Network retrieval for explicit URLs should be done with terminal commands \
  such as `curl -L`; do not call tools that are not listed in your tool menu.
- `tesseract` + `pdftoppm`/`pdftotext`/`pdfinfo` — for OCR of scanned PDFs.
- `ffmpeg` — for audio/video manipulation (e.g. extract audio from mp4).
- Python: `pdfplumber`, `PyMuPDF`/`fitz`, `openpyxl`, `xlsxwriter`, `python-docx`, \
  `python-pptx`, `reportlab`, `fpdf2`, `weasyprint`, `Pillow`, \
  `pdf2image`, `psd_tools`, `whisper`, `pandas`.

If a system PDF utility such as `pdfinfo`, `pdftoppm`, or `pdftotext` is not \
available, use Python PDF libraries such as `PyMuPDF`/`fitz` or `pdfplumber` \
instead of stopping.

The grader checks for the actual output files on disk in the current task \
working directory — saved scripts that you didn't run, files written outside \
the current task working directory, scraped research, and notes are worth \
nothing unless explicitly requested. If you write a script, save and run it in \
the current task working directory. After producing the deliverable, run `ls \
-la` to confirm the files exist before calling `finish`. Do not ask the user \
questions; make reasonable assumptions.

Work discipline for file-deliverable tasks:
1. Decide the output file type(s) and clear, task-matching filename(s) before \
broad exploration.
2. Within your first five tool actions, create draft deliverable file(s) for \
every required output type with valid file bytes. Prefer any listed target \
names, but case/spacing variants or similarly descriptive filenames with the \
correct extension are acceptable when the content is right. Improve those same \
files iteratively.
3. Keep exploratory command output small. Inspect only the rows, columns, and \
metadata needed to produce the deliverable.
4. Avoid very large multi-line scripts directly inside a tool-call JSON \
argument. Prefer creating a short script file in the current directory, running \
it, then verifying the output file. Do not read a long generated script back \
into the conversation unless the script failed and the error location is needed.
5. If a terminal command soft-times out, interrupt it once. Do not loop on \
interrupts or empty commands; immediately create or submit the best valid \
deliverable file you have.
6. Never finish with only scripts, screenshots, crops, notes, or copied input \
files when a PDF/DOCX/XLSX/PPTX/media deliverable was requested.
7. A partial but valid deliverable file is better than no submitted file.
"""

_DEFAULT_TOOL_SECTION = (
    "You have `terminal` (bash) and `file_editor`. The reference files for this "
    "task are at the absolute paths listed above — read them directly."
)

_TERMINAL_ONLY_TOOL_SECTION = (
    "You have `terminal` (bash). `file_editor` is not available in this run. "
    "The reference files for this task are at the absolute paths listed above "
    "— read them directly with terminal commands or Python scripts."
)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _rubric_prompt_section(rubric_pretty: object, rubric_json_raw: object) -> str:
    rubric_text = str(rubric_pretty or "").strip()
    if not rubric_text and rubric_json_raw:
        try:
            parsed = json.loads(str(rubric_json_raw))
            rubric_text = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (TypeError, json.JSONDecodeError):
            rubric_text = str(rubric_json_raw).strip()
    if not rubric_text:
        return ""
    return (
        "## Evaluation rubric\n\n"
        "Use the following rubric as a checklist while producing the deliverable. "
        "Satisfy each applicable criterion directly in the files you create. "
        "Treat the task prompt and reference files as the source of truth: do "
        "not invent data, alter source facts, or add irrelevant content merely "
        "to satisfy a rubric phrase. When a rubric criterion is checkable from "
        "the source files or final artifacts, verify it with the appropriate "
        "file library or command before finishing. Keep verification compact: "
        "once the expected final artifact exists, opens successfully, and one "
        "structured verification pass confirms the main task/rubric criteria, "
        "finish with the file path instead of re-reading the full source, "
        "rewriting the artifact, or continuing open-ended optimization. If a "
        "verification check fails, repair only that concrete failure and finish "
        "after re-checking it. If a criterion conflicts "
        "with the task prompt or source files, follow the task/source files and "
        "make the most faithful professional deliverable. When a conflict is "
        "only about packaging, filename, tab order, or presentation rather than "
        "the underlying facts, add a clearly labeled compatibility artifact, "
        "worksheet, or view so both interpretations are available without "
        "falsifying source data. For spreadsheet sampling tasks, keep sampled "
        "rows and full population/reference rows clearly separated, usually by "
        "putting the selected sample in the primary sample sheet and any full "
        "population copy in a separate reference sheet. If the task and rubric "
        "name compatible filenames for the same artifact type, create both "
        "filename variants with finished content unless the task or rubric "
        "explicitly requires a single final file or otherwise forbids extra "
        "files, and mention both relative paths when you finish.\n\n"
        f"{rubric_text}\n"
    )


def _deliverable_rescue_prompt_section() -> str:
    rescue_text = os.getenv("IPW_GDPVAL_DELIVERABLE_RESCUE_INSTRUCTIONS", "").strip()
    if not rescue_text:
        return ""
    return (
        "## Run-specific deliverable rescue instructions\n\n"
        f"{rescue_text}\n"
    )


_TASK_RESCUE_HINTS_CACHE: tuple[str, dict[str, Any]] | None = None


def _load_task_rescue_hints() -> dict[str, Any]:
    global _TASK_RESCUE_HINTS_CACHE
    path = os.getenv("IPW_GDPVAL_TASK_RESCUE_HINTS_FILE", "").strip()
    if not path:
        _TASK_RESCUE_HINTS_CACHE = ("", {})
        return {}
    if _TASK_RESCUE_HINTS_CACHE is not None and _TASK_RESCUE_HINTS_CACHE[0] == path:
        return _TASK_RESCUE_HINTS_CACHE[1]

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Failed to load GDPval task rescue hints from %s: %s", path, exc)
        hints: dict[str, Any] = {}
    else:
        if isinstance(payload, dict):
            hints = payload
        elif isinstance(payload, list):
            hints = {}
            for item in payload:
                if not isinstance(item, dict):
                    continue
                key = item.get("task_id") or item.get("instance_id") or item.get("id")
                if key:
                    hints[str(key)] = item
        else:
            LOGGER.warning("GDPval task rescue hints must be a JSON object or list: %s", path)
            hints = {}
    _TASK_RESCUE_HINTS_CACHE = (path, hints)
    return hints


def _task_rescue_hint_prompt_section(task_id: object) -> str:
    task_key = str(task_id or "").strip()
    if not task_key:
        return ""
    hint = _load_task_rescue_hints().get(task_key)
    if not hint:
        return ""
    if isinstance(hint, str):
        text = hint.strip()
    elif isinstance(hint, dict):
        parts: list[str] = []
        for key in (
            "hint",
            "failure_cause",
            "previous_failure",
            "expected_deliverables",
            "required_action",
            "avoid",
        ):
            value = hint.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple)):
                value_text = "; ".join(str(item) for item in value)
            else:
                value_text = str(value)
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {value_text}")
        text = "\n".join(parts).strip()
    else:
        text = str(hint).strip()
    if not text:
        return ""
    if len(text) > 4000:
        text = text[:4000] + "\n... [truncated]"
    return (
        "\n## Task-specific rescue hint from prior run\n\n"
        "This hint describes a previous failure mode for this same task. "
        "Use it only to avoid repeating that artifact mistake; the original "
        "task, reference files, and rubric remain authoritative.\n\n"
        f"{text}\n"
    )


def _normalize_task_instances(
    task_instances: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not task_instances:
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(task_instances):
        if not isinstance(item, dict):
            raise ValueError("task_instances entries must be JSON objects")
        task_id = str(item.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"task_instances[{index}] is missing task_id")
        instance_id = str(item.get("instance_id") or "").strip()
        if not instance_id:
            instance_id = f"{task_id}::instance{index:04d}"
        copy = dict(item)
        copy["task_id"] = task_id
        copy["instance_id"] = instance_id
        normalized.append(copy)
    return normalized


def _load_task_instances_file(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None or str(path).strip() == "":
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("task_instances_file must contain a JSON list")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError("task_instances_file entries must be JSON objects")
    return [dict(item) for item in raw]


def _deliverable_name_from_uri(uri: str) -> str | None:
    """Return the expected output filename embedded in a GDPval HF URI."""
    parsed = _hf_uri_to_repo_path(uri)
    if parsed is None:
        return None
    _, file_path, _ = parsed
    name = Path(file_path).name
    return unquote(name) if name else None


def _hf_uri_to_repo_path(
    hf_uri: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Parse a HuggingFace dataset URI into ``(repo, file_path, revision)``.

    Accepts both forms:
        hf://datasets/<org>/<name>/<file_path>
        hf://datasets/<org>/<name>@<revision>/<file_path>

    The file_path is URL-decoded (``%20`` → space, etc.).
    """
    if not hf_uri or not isinstance(hf_uri, str):
        return None
    prefix = "hf://datasets/"
    if not hf_uri.startswith(prefix):
        return None
    remainder = hf_uri[len(prefix):]
    parts = remainder.split("/", 2)
    if len(parts) < 3:
        return None
    org, name_with_rev, file_path = parts
    if "@" in name_with_rev:
        name, revision = name_with_rev.split("@", 1)
    else:
        name, revision = name_with_rev, None
    repo = f"{org}/{name}"
    return repo, unquote(file_path), revision


@DatasetRegistry.register("gdpval")
class GDPvalDataset(DatasetProvider):
    """OpenAI GDPval dataset (openai/gdpval).

    Loads 220 professional tasks from HuggingFace, downloads associated
    reference files on demand, and yields one ``DatasetRecord`` per task.
    Scoring uses the rubric-based ``gdpval`` evaluation handler.
    """

    dataset_id = "gdpval"
    dataset_name = "GDPval"
    evaluation_method = "gdpval"

    _hf_path = "openai/gdpval"
    _default_split = "train"
    _default_subset = "default"

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        revision: Optional[str] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        download_files: bool = True,
        shard_idx: Optional[int] = None,
        n_shards: Optional[int] = None,
        task_ids: Optional[list[str]] = None,
        instance_ids: Optional[list[str]] = None,
        task_instances: Optional[list[dict[str, Any]]] = None,
        task_instances_file: Optional[str] = None,
        tool_preference: Optional[str] = None,
        include_rubric_in_prompt: bool | str = False,
    ) -> None:
        self._split = split or self._default_split
        self._subset = subset or self._default_subset
        self._revision = revision
        self._max_samples = max_samples
        file_task_instances = _load_task_instances_file(task_instances_file)
        if task_instances and file_task_instances:
            raise ValueError(
                "task_instances cannot be combined with task_instances_file"
            )
        self._task_instances = _normalize_task_instances(
            task_instances or file_task_instances
        )
        if self._task_instances and (task_ids or instance_ids):
            raise ValueError(
                "task_instances cannot be combined with task_ids or instance_ids"
            )
        selected_ids = list(task_ids or []) + list(instance_ids or [])
        self._task_ids = {str(item).strip() for item in selected_ids if str(item).strip()}
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._download_files = download_files
        if tool_preference not in {None, "", "terminal"}:
            raise ValueError("tool_preference must be 'terminal' when provided")
        self._tool_preference = tool_preference or None
        self._include_rubric_in_prompt = _coerce_bool(include_rubric_in_prompt)
        if (shard_idx is None) != (n_shards is None):
            raise ValueError("shard_idx and n_shards must be specified together")
        if n_shards is not None and n_shards <= 0:
            raise ValueError("n_shards must be > 0")
        if shard_idx is not None and not (0 <= shard_idx < n_shards):
            raise ValueError(f"shard_idx must be in [0, {n_shards})")
        self._shard_idx = shard_idx
        self._n_shards = n_shards
        self._records: Tuple[DatasetRecord, ...] = tuple(self._build_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        provider = (self.eval_client or "openai").strip().lower()
        provider_keys = {
            "openai": ("IPW_EVAL_API_KEY", "OPENAI_API_KEY"),
            "gemini": ("IPW_EVAL_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
            "anthropic": ("IPW_EVAL_API_KEY", "ANTHROPIC_API_KEY"),
        }
        keys = provider_keys.get(provider, ("IPW_EVAL_API_KEY",))
        if not any(os.getenv(key) for key in keys):
            issues.append(
                "Missing evaluation API key. Set one of "
                + ", ".join(keys)
                + " for GDPval rubric scoring."
            )
        return issues

    def score(
        self,
        record: DatasetRecord,
        response: str,
        *,
        eval_client: Optional[InferenceClient] = None,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        handler = self._resolve_handler(eval_client)
        return handler.evaluate(
            problem=record.problem,
            reference=record.answer,
            model_answer=response,
            metadata=record.dataset_metadata,
        )

    def _resolve_handler(self, eval_client: Optional[InferenceClient]):
        judge_client = eval_client or ClientRegistry.create(
            self.eval_client or "openai",
            base_url=self.eval_base_url or "https://api.openai.com/v1",
            model=self.eval_model or "gpt-5.4-mini",
        )
        return EvaluationRegistry.create(self.evaluation_method, client=judge_client)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _build_records(self) -> List[DatasetRecord]:
        kwargs = {"name": self._subset, "split": self._split}
        if self._revision is not None:
            kwargs["revision"] = self._revision
        dataset = load_dataset(self._hf_path, **kwargs)

        rows: Sequence[MutableMapping[str, Any]]
        if hasattr(dataset, "to_list"):
            rows = dataset.to_list()
        else:
            rows = list(dataset)
        if self._task_instances:
            rows_by_task_id: dict[str, MutableMapping[str, Any]] = {}
            for row in rows:
                task_id = str(row.get("task_id") or "").strip()
                if task_id:
                    rows_by_task_id[task_id] = row
            instances = list(self._task_instances)
            if self._max_samples is not None:
                instances = instances[: self._max_samples]
            if self._n_shards is not None:
                instances = [
                    instance
                    for i, instance in enumerate(instances)
                    if i % self._n_shards == self._shard_idx
                ]
            records: list[DatasetRecord] = []
            for instance_position, instance in enumerate(instances):
                task_id = str(instance.get("task_id") or "").strip()
                raw = rows_by_task_id.get(task_id)
                if raw is None:
                    raise ValueError(f"GDPval task_instances task_id not found: {task_id}")
                if "occupation" in instance:
                    scheduled_occupation = str(instance.get("occupation") or "").strip()
                    dataset_occupation = str(raw.get("occupation") or "").strip()
                    if not scheduled_occupation or scheduled_occupation != dataset_occupation:
                        raise ValueError(
                            "GDPval task_instances occupation does not match pinned "
                            f"dataset row for task_id {task_id}: "
                            f"{scheduled_occupation!r} != {dataset_occupation!r}"
                        )
                record = self._convert_row(
                    dict(raw),
                    task_instance=instance,
                    instance_position=instance_position,
                )
                if record is not None:
                    records.append(record)
            return records
        if self._task_ids:
            rows = [
                row
                for row in rows
                if str(row.get("task_id") or "").strip() in self._task_ids
            ]
        if self._max_samples is not None:
            rows = rows[: self._max_samples]

        # Shard by global task index, modulo n_shards. Round-robin distribution
        # gives each shard a similar mix of task difficulty (rather than e.g.
        # shard 0 getting all the bank-audit tasks).
        if self._n_shards is not None:
            rows = [r for i, r in enumerate(rows) if i % self._n_shards == self._shard_idx]

        records: list[DatasetRecord] = []
        for raw in rows:
            record = self._convert_row(dict(raw))
            if record is not None:
                records.append(record)
        return records

    def _convert_row(
        self,
        raw: MutableMapping[str, Any],
        *,
        task_instance: dict[str, Any] | None = None,
        instance_position: int | None = None,
    ) -> Optional[DatasetRecord]:
        task_id = str(raw.get("task_id") or "").strip()
        prompt = str(raw.get("prompt") or "").strip()
        if not task_id or not prompt:
            return None

        occupation = str(raw.get("occupation") or "").strip() or "Professional"
        sector = str(raw.get("sector") or "").strip() or "General"

        reference_hf_uris = list(raw.get("reference_file_hf_uris") or [])
        reference_urls = list(raw.get("reference_file_urls") or [])
        deliverable_hf_uris = list(raw.get("deliverable_file_hf_uris") or [])

        rubric_json_raw = raw.get("rubric_json") or ""
        rubric_pretty = raw.get("rubric_pretty") or ""

        # Materialize reference files on disk (downloaded lazily on first use)
        inputs_dir = self._cache_dir / task_id / "inputs"
        if self._download_files:
            local_paths = self._materialize_files(reference_hf_uris, inputs_dir)
        else:
            if _strict_reference_staging() and reference_hf_uris:
                raise RuntimeError("GDPVal reference files are required for this run")
            local_paths = []
        if _strict_reference_staging() and len(local_paths) != len(reference_hf_uris):
            raise RuntimeError(f"GDPVal task {task_id} has incomplete reference files")
        reference_sha256 = {path.name: _sha256_file(path) for path in local_paths}

        # Build a human-readable file listing for the prompt.
        # Files are copied into the agent's working environment under
        # ./inputs/ by the OpenHands harness.
        if local_paths:
            file_lines = "\n".join(f"- {p.name}" for p in local_paths)
            file_section = (
                "## Reference files\n\n"
                "The following files are available in the working directory "
                "under `inputs/` (or `/workspace/inputs/` if you are in a "
                "sandboxed shell). Read them with whatever file tools you "
                "have available.\n\n"
                f"{file_lines}\n"
            )
        elif reference_urls:
            url_lines = "\n".join(f"- {u}" for u in reference_urls)
            file_section = (
                "## Reference files (download from URL)\n\n"
                f"{url_lines}\n"
            )
        else:
            file_section = ""

        deliverable_files = [
            name
            for uri in deliverable_hf_uris
            for name in [_deliverable_name_from_uri(str(uri))]
            if name
        ]
        if deliverable_files:
            deliverable_lines = "\n".join(f"- {name}" for name in deliverable_files)
            deliverable_section = (
                "## Expected output deliverables\n\n"
                "Create final deliverable files in the current working directory "
                "for the target filename(s) below. Reference/input files do not "
                "count as submitted deliverables. Prefer the listed names, but "
                "case/spacing variants or similarly descriptive filenames with "
                "the correct extension are acceptable when the content is right. "
                "For every listed deliverable or output type, make a valid "
                "starter file before broad exploration, then improve the same "
                "file in place. If several files are listed, create all required "
                "deliverables in one compact script or command before spending "
                "more turns on analysis.\n\n"
                f"{deliverable_lines}\n"
            )
        else:
            deliverable_section = ""

        task_prompt = _PROMPT_TEMPLATE.format(
            occupation=occupation,
            sector=sector,
            file_section=file_section,
            deliverable_section=deliverable_section,
            rescue_section=(
                _deliverable_rescue_prompt_section()
                + _task_rescue_hint_prompt_section(task_id)
            ),
            rubric_section=(
                _rubric_prompt_section(rubric_pretty, rubric_json_raw)
                if self._include_rubric_in_prompt
                else ""
            ),
            task_prompt=prompt,
            tool_section=(
                _TERMINAL_ONLY_TOOL_SECTION
                if self._tool_preference == "terminal"
                else _DEFAULT_TOOL_SECTION
            ),
        )

        instance_id = str((task_instance or {}).get("instance_id") or task_id).strip()
        metadata: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "task_id": task_id,
            "instance_id": instance_id,
            "occupation": occupation,
            "sector": sector,
            "reference_file_hf_uris": reference_hf_uris,
            "gdpval_reference_sha256": reference_sha256,
            "reference_file_urls": reference_urls,
            "deliverable_file_hf_uris": deliverable_hf_uris,
            "deliverable_files": deliverable_files,
            "deliverable_rescue_instructions": bool(
                os.getenv("IPW_GDPVAL_DELIVERABLE_RESCUE_INSTRUCTIONS", "").strip()
            ),
            "task_rescue_hint": str(task_id) in _load_task_rescue_hints(),
            "gdpval_inputs_dir": str(inputs_dir) if local_paths else None,
            "tool_preference": self._tool_preference,
            "include_rubric_in_prompt": self._include_rubric_in_prompt,
            "rubric_pretty": rubric_pretty,
            "rubric_json": rubric_json_raw,
        }
        if task_instance is not None:
            for key in (
                "repeat_index",
                "repeat_count",
                "occupation_batch_index",
                "occupation_batch_id",
                "occupation_task_position",
                "source_task_index",
            ):
                if key in task_instance:
                    metadata[key] = task_instance[key]
            metadata["repeated_instance"] = True
            metadata["task_instance_position"] = instance_position
        else:
            metadata["repeated_instance"] = False

        # The "answer" slot is unused for GDPval (no reference answers); we
        # store rubric_pretty there so it shows up in artifact dumps.
        return DatasetRecord(
            problem=task_prompt,
            answer=rubric_pretty if isinstance(rubric_pretty, str) else "",
            subject=occupation,
            dataset_metadata=metadata,
        )

    def _materialize_files(self, hf_uris: List[str], dest_dir: Path) -> List[Path]:
        """Download each HF URI into ``dest_dir`` (idempotent, cached)."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_paths: list[Path] = []
        strict = _strict_reference_staging()
        seen_names: set[str] = set()
        for uri in hf_uris:
            parsed = _hf_uri_to_repo_path(uri)
            if parsed is None:
                if strict:
                    raise ValueError(f"GDPVal reference URI is not a Hugging Face file: {uri}")
                LOGGER.warning("Skipping non-HF reference URI: %s", uri)
                continue
            repo, path_in_repo, revision = parsed
            name = Path(path_in_repo).name
            if strict and (not name or name in seen_names):
                raise ValueError(f"GDPVal reference filename is empty or duplicated: {name!r}")
            seen_names.add(name)
            # The table and its referenced files must come from the same
            # immutable snapshot. GDPVal URIs can embed mutable @main, so an
            # explicit dataset revision takes precedence for this repository.
            if self._revision and repo == self._hf_path:
                revision = self._revision
            try:
                local = hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=path_in_repo,
                    revision=revision,
                    cache_dir=str(self._cache_dir / "_hf_cache"),
                )
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"Failed to download GDPVal reference file {uri}") from exc
                LOGGER.warning("Failed to download %s: %s", uri, exc)
                continue
            # Copy into dest_dir with a flat filename so agents see a clean tree
            target = dest_dir / name
            if strict:
                source = Path(local)
                if not source.is_file():
                    raise RuntimeError(f"GDPVal reference download is not a file: {uri}")
                expected_hash = _sha256_file(source)
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise RuntimeError(f"GDPVal reference cache target is not a regular file: {target}")
                if not target.exists() or _sha256_file(target) != expected_hash:
                    descriptor, temporary_name = tempfile.mkstemp(prefix=".reference-", dir=dest_dir)
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    try:
                        shutil.copy2(source, temporary)
                        if _sha256_file(temporary) != expected_hash:
                            raise RuntimeError(f"GDPVal reference copy changed: {uri}")
                        os.replace(temporary, target)
                    finally:
                        temporary.unlink(missing_ok=True)
                if _sha256_file(target) != expected_hash:
                    raise RuntimeError(f"GDPVal reference cache checksum mismatch: {uri}")
            elif not target.exists():
                shutil.copy2(local, target)
            local_paths.append(target)
        return local_paths


__all__ = ["GDPvalDataset"]
