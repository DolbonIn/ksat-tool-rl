#!/usr/bin/env python3
"""Verify packaged source bytes without importing or executing project code."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
def main():
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            failures.append(item["path"] + ": unsafe path")
        elif not path.is_file():
            failures.append(item["path"] + ": missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            failures.append(item["path"] + ": checksum mismatch")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Verified {len(manifest['files'])} packaged files.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
