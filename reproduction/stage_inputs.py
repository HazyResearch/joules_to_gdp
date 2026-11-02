#!/usr/bin/env python3
"""Link verified external inputs; leave bundled public sources in place."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_inputs import MANIFEST, ROOT, has_symlink_component, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    source_root = args.data_root.resolve()
    issues = verify(source_root, args.manifest)
    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit("External inputs must pass verification before staging")
    entries = json.loads(args.manifest.read_text())["inputs"]
    linked = 0
    for item in entries:
        if item.get("bundled", False):
            continue
        rel = Path(item["path"])
        target_rel = Path(item.get("target_path", item["path"]))
        if target_rel.is_absolute() or ".." in target_rel.parts:
            raise SystemExit(f"Unsafe staging target: {target_rel}")
        target = ROOT / target_rel
        source = source_root / rel
        if not source.exists() and not item.get("required", True):
            continue
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise SystemExit(f"Refusing to replace existing path: {target}")
            continue
        if has_symlink_component(ROOT, target_rel.parent):
            raise SystemExit(f"Staging target has a symlink parent: {target_rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
        linked += 1
    print(f"Linked {linked} verified inputs into ignored local paths")


if __name__ == "__main__":
    main()
