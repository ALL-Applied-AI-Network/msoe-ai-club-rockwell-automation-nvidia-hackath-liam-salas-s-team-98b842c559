"""Check that the untracked dataset on this machine matches the team's manifest.

    python scripts/check_data.py            # verify sizes and SHA-256 hashes
    python scripts/check_data.py --quick    # sizes only, seconds instead of a minute
    python scripts/check_data.py --only dataset   # dataset-only download: skip the twin rows
    python scripts/check_data.py --write    # (once) record the manifest from a good copy

The manifest data/dataset_manifest.csv is small and committed; the data is not.
Two teammates whose check passes are reading byte-identical inputs.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.paths import ENV_VAR, REPO, data_root

MANIFEST = REPO / "data/dataset_manifest.csv"
TRACKED = ("csi-vial-fill-dataset", "csi_digital_twin_v1")  # tool outputs elsewhere in the root are ignored
ONLY = {"dataset": TRACKED[0], "twin": TRACKED[1]}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan(root, tracked=TRACKED):
    for top in tracked:
        for path in sorted((root / top).rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                yield path.relative_to(root).as_posix(), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", help=f"data root (default: ${ENV_VAR} or external/)")
    ap.add_argument("--quick", action="store_true", help="compare sizes only")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="manifest file (default: data/dataset_manifest.csv)")
    ap.add_argument("--only", choices=sorted(ONLY), help="check only one top-level folder; the other's manifest rows are skipped")
    ap.add_argument("--write", action="store_true", help="overwrite the manifest from this copy")
    a = ap.parse_args()
    root = data_root(a.root)
    manifest = a.manifest
    print(f"data root: {root}")
    tracked = (ONLY[a.only],) if a.only else TRACKED
    if not any((root / top).is_dir() for top in tracked):
        raise SystemExit(f"no {' or '.join(tracked)} under {root}; see docs/data.md")
    if a.only and a.write:
        raise SystemExit("--only cannot be combined with --write: the manifest must cover both folders")

    if a.write:
        rows = [{"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)} for rel, p in scan(root)]
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} entries to {manifest}; commit it with the source/version in docs/data.md")
        return

    if not manifest.is_file():
        raise SystemExit("no data/dataset_manifest.csv yet; one teammate runs --write on the original download")
    with manifest.open() as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)
                    if row["path"].split("/", 1)[0] in tracked}
    found = dict(scan(root, tracked))
    # Compare case-insensitively too: macOS/Windows hide case mismatches that break on Linux GPU hosts.
    lower = {k.lower(): k for k in found}
    problems = []
    for rel, row in expected.items():
        path = found.get(rel)
        if path is None:
            hint = f" (found as {lower[rel.lower()]}: case differs)" if rel.lower() in lower else ""
            problems.append(f"missing  {rel}{hint}")
        elif path.stat().st_size != int(row["bytes"]):
            problems.append(f"size     {rel}: {path.stat().st_size} != {row['bytes']}")
        elif not a.quick and sha256(path) != row["sha256"]:
            problems.append(f"hash     {rel}")
    extra = sorted(set(found) - set(expected))
    for rel in extra:
        print(f"extra    {rel} (not in manifest; fine if intentional)")
    for line in problems:
        print(line)
    checked = "sizes" if a.quick else "sizes and hashes"
    if problems:
        raise SystemExit(f"{len(problems)} of {len(expected)} files differ from the manifest")
    print(f"ok: {len(expected)} files match the manifest ({checked})")


if __name__ == "__main__":
    main()
