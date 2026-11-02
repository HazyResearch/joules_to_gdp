#!/usr/bin/env python3
"""Verify bundled public sources and external recorded inputs by checksum."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproduction" / "input_manifest.json"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def directory_digest(path: Path) -> str:
    """Hash every file in a tree, including its relative name and byte length.

    Records are sorted by UTF-8 relative path and encoded as
    ``path NUL size NUL SHA-256 LF``. Empty directories have no records.
    Internal symlinks are rejected so the digest cannot silently depend on
    files outside the archived directory.
    """
    sha = hashlib.sha256()
    children = sorted(path.rglob("*"), key=lambda child: child.relative_to(path).as_posix().encode("utf-8"))
    for child in children:
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            raise ValueError(f"symlink in directory: {relative}")
        if child.is_dir():
            continue
        if not child.is_file():
            raise ValueError(f"unsupported entry in directory: {relative}")
        sha.update(relative.encode("utf-8"))
        sha.update(b"\0")
        sha.update(str(child.stat().st_size).encode("ascii"))
        sha.update(b"\0")
        sha.update(digest(child).encode("ascii"))
        sha.update(b"\n")
    return sha.hexdigest()


def has_symlink_component(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def verify(data_root: Path, manifest: Path = MANIFEST) -> list[str]:
    data = json.loads(manifest.read_text())
    issues = []
    seen = set()
    for item in data["inputs"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            issues.append(f"unsafe manifest path: {relative}")
            continue
        if relative in seen:
            issues.append(f"duplicate manifest path: {relative}")
            continue
        seen.add(relative)
        bundled = item.get("bundled", False)
        if type(bundled) is not bool:
            issues.append(f"invalid bundled flag: {relative}")
            continue
        source_root = ROOT if bundled else data_root
        path = source_root / relative
        kind = item.get("kind", "file")
        if kind not in {"file", "directory"}:
            issues.append(f"unsupported input kind: {relative}: {kind}")
            continue
        if has_symlink_component(source_root, relative):
            issues.append(f"symlink in input path: {relative}")
            continue
        exists = path.is_dir() if kind == "directory" else path.is_file()
        if not exists:
            if item.get("required", True):
                issues.append(f"missing: {relative}")
            continue
        expected = item.get("sha256")
        if not expected:
            issues.append(f"missing checksum: {relative}")
            continue
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            issues.append(f"invalid checksum: {relative}")
            continue
        try:
            actual = directory_digest(path) if kind == "directory" else digest(path)
        except (OSError, ValueError) as exc:
            issues.append(f"cannot hash: {relative}: {exc}")
        else:
            if actual != expected:
                issues.append(f"checksum mismatch: {relative}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    issues = verify(args.data_root, args.manifest)
    for issue in issues:
        print(issue)
    if issues:
        raise SystemExit(f"{len(issues)} input checks failed")
    print("All bundled and external inputs match the manifest")


if __name__ == "__main__":
    main()
