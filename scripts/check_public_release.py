#!/usr/bin/env python3
"""Check staged public source, documented occupational inputs, and fixtures."""

from __future__ import annotations

import io
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys

from rich.console import Console
from rich.markdown import Markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    "README.md",
    "reproduction/README.md",
    "energy-monitor/README.md",
    "intelligence-per-watt/README.md",
    "occupation_data/README.md",
    "analysis/README.md",
    "cluster/README.md",
    "reproduction/vllm/README.md",
)
ALLOWLIST = Path("scripts/release_files.txt")
FORBIDDEN_PARTS = {
    ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "__pycache__", "artifacts", "cache", "caches", "checkpoints", "dist",
    "figures", "htmlcov", "logs", "outputs", "plots", "results", "runs", "site",
    "target",
}
FORBIDDEN_SUFFIXES = {
    ".a", ".arrow", ".bak", ".bash", ".class", ".csv", ".db", ".dll",
    ".dylib", ".exe", ".feather", ".gz", ".ipynb", ".jpeg", ".jpg",
    ".jsonl", ".log", ".npy", ".npz", ".o", ".parquet", ".pdf", ".pickle",
    ".pkl", ".png", ".pt", ".pyc", ".pyo", ".safetensors", ".sh", ".so",
    ".sqlite", ".sqlite3", ".swp", ".tar", ".tmp", ".tsv", ".webp", ".whl",
    ".xls", ".xlsx", ".zip",
}
SAFE_SYNTHETIC_DATA = {
    Path("analysis/fixtures/energy_boundary/batches.csv"),
    Path("analysis/fixtures/energy_boundary/requests.csv"),
    Path("analysis/fixtures/energy_boundary/ipw_batches.jsonl"),
    Path("analysis/fixtures/energy_boundary/ipw_traces.jsonl"),
}
PUBLIC_INPUTS = {
    Path("occupation_data/data/input/OEWS.csv"),
    Path("occupation_data/data/input/ONET.csv"),
    Path("occupation_data/data/input/task_statements.csv"),
}
PUBLIC_IMAGES = {
    Path("assets/ai-gpu-cash-flow.png"),
    Path("assets/j2g.png"),
    Path("assets/pipeline.svg"),
    Path("assets/project-mark.png"),
}
GENERATOR_NAME = re.compile(r"(?:^|[_-])(?:plot|figure|table)s?(?:[_-]|$)", re.IGNORECASE)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{28,}\b"),
    re.compile(rb"\bASIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-(?:ant|proj)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


class DocumentationHTMLLinks(HTMLParser):
    """Collect native HTML navigation and image references in Markdown guides."""

    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attribute = {"a": "href", "img": "src"}.get(tag)
        if attribute:
            self.references.extend(value for key, value in attrs if key == attribute and value)


def staged_snapshot() -> tuple[dict[Path, bytes], dict[Path, str]]:
    """Read the exact staged blobs, including staged deletions and file modes."""
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    entries = []
    for item in result.stdout.split(b"\0"):
        if not item:
            continue
        header, name = item.split(b"\t", 1)
        mode, oid, stage = header.decode("ascii").split()
        if stage != "0":
            raise ValueError("Resolve index conflicts before checking the release")
        entries.append((Path(name.decode("utf-8")), mode, oid))
    result = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=ROOT, check=True, capture_output=True,
        input="".join(f"{oid}\n" for _, _, oid in entries).encode("ascii"),
    )
    contents = io.BytesIO(result.stdout)
    files, modes = {}, {}
    for path, mode, oid in entries:
        actual_oid, kind, size = contents.readline().split()
        if actual_oid.decode("ascii") != oid or kind != b"blob":
            raise ValueError(f"Unsupported staged object: {path}")
        payload = contents.read(int(size))
        if len(payload) != int(size) or contents.read(1) != b"\n":
            raise ValueError(f"Incomplete staged object: {path}")
        files[path], modes[path] = payload, mode
    return files, modes


def check_paths(files: dict[Path, bytes], modes: dict[Path, str]) -> list[str]:
    issues: list[str] = []
    for relative, content in files.items():
        if modes[relative] not in {"100644", "100755"}:
            issues.append(f"nonregular staged file: {relative}")
            continue
        suffix_forbidden = (relative.suffix.lower() in FORBIDDEN_SUFFIXES
                            and relative not in SAFE_SYNTHETIC_DATA | PUBLIC_INPUTS | PUBLIC_IMAGES)
        if (FORBIDDEN_PARTS.intersection(relative.parts)
                or suffix_forbidden
                or GENERATOR_NAME.search(relative.stem)
                or relative.name in {".DS_Store", "Thumbs.db", ".env", "coverage.xml"}
                or relative.name.startswith((".env.", ".coverage"))
                or "ledger" in relative.name.lower()):
            issues.append(f"recorded output or binary asset tracked: {relative}")
        if b"\0" in content:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                issues.append(f"possible secret ({pattern.pattern.decode('ascii')}): {relative}")
    return issues


def check_bundled_sources(files: dict[Path, bytes]) -> list[str]:
    manifest = json.loads(files[Path("reproduction/input_manifest.json")])
    bundled = {Path(item["path"]): item["sha256"] for item in manifest["inputs"]
               if item.get("bundled", False)}
    if set(bundled) != PUBLIC_INPUTS:
        return ["bundled input manifest differs from the approved public source list"]
    issues = []
    for relative, expected in bundled.items():
        if relative not in files:
            issues.append(f"bundled source is not a tracked regular file: {relative}")
        elif hashlib.sha256(files[relative]).hexdigest() != expected:
            issues.append(f"bundled source checksum mismatch: {relative}")
    return issues


def check_allowlist(files: dict[Path, bytes]) -> list[str]:
    expected = {
        line for line in files[ALLOWLIST].decode("utf-8").splitlines()
        if line and not line.startswith("#")
    }
    actual = {path.as_posix() for path in files}
    return ([f"unexpected tracked file: {name}" for name in sorted(actual - expected)]
            + [f"missing tracked file: {name}" for name in sorted(expected - actual)])


def check_documentation(files: dict[Path, bytes]) -> list[str]:
    issues: list[str] = []
    for name in DOCS:
        relative = Path(name)
        path = ROOT / relative
        if relative not in files:
            issues.append(f"missing guide: {name}")
            continue
        source = files[relative].decode("utf-8")
        blocked_terms = r"\bledgers?\b" if name == "README.md" else r"\b(?:paper|ledgers?)\b"
        if re.search(blocked_terms, source, re.IGNORECASE):
            issues.append(f"outdated project documentation terminology: {name}")
        rendered = io.StringIO()
        Console(file=rendered, width=100, force_terminal=False, color_system=None).print(Markdown(source))
        if len(rendered.getvalue().strip()) < 100:
            issues.append(f"empty Markdown render: {name}")
        html_links = DocumentationHTMLLinks()
        html_links.feed(source)
        for target in MARKDOWN_LINK.findall(source) + html_links.references:
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            linked = (path.parent / target).resolve()
            if (not linked.is_relative_to(ROOT)
                    or not any(candidate == linked.relative_to(ROOT)
                               or candidate.is_relative_to(linked.relative_to(ROOT))
                               for candidate in files)):
                issues.append(f"broken local Markdown link: {name} -> {target}")
    return issues


def main() -> None:
    files, modes = staged_snapshot()
    issues = (check_allowlist(files) + check_paths(files, modes)
              + check_bundled_sources(files) + check_documentation(files))
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        raise SystemExit(f"{len(issues)} public release checks failed")
    print("Public release staged files, secret patterns, bundled sources, and guides passed")


if __name__ == "__main__":
    main()
