#!/usr/bin/env python3
from pathlib import Path
import os

root = Path("/Volumes/samsung_ssd/data/birdsong")

def _try_list_root(path: Path) -> None:
    try:
        entries = list(path.iterdir())
    except PermissionError as exc:
        print(f"PermissionError listing {path}: {exc}")
        return
    except FileNotFoundError as exc:
        print(f"FileNotFoundError listing {path}: {exc}")
        return
    print(f"Root entries: {len(entries)}")
    if entries:
        sample = [e.name for e in entries[:10]]
        print(f"Root sample: {sample}")


def summarize_dir(path: Path, max_depth: int = 3) -> None:
    print(f"Root: {path}")
    if not path.exists():
        print("Path does not exist.")
        return
    _try_list_root(path)
    total_dirs = 0
    total_files = 0
    walked = False
    def _onerror(err):
        print(f"os.walk error: {err}")
    for dirpath, dirnames, filenames in os.walk(path, onerror=_onerror):
        walked = True
        depth = Path(dirpath).relative_to(path).parts
        if len(depth) > max_depth:
            dirnames[:] = []
            continue
        total_dirs += 1
        total_files += len(filenames)
        wavs = [f for f in filenames if f.lower().endswith(".wav")]
        txts = [f for f in filenames if f.lower().endswith(".txt")]
        if wavs or txts:
            rel = Path(dirpath).relative_to(path)
            print(f"\n{rel}/")
            print(f"  wav: {len(wavs)}")
            print(f"  txt: {len(txts)}")
            if wavs:
                print(f"  wav sample: {wavs[:3]}")
            if txts:
                print(f"  txt sample: {txts[:3]}")
    if not walked:
        print("os.walk returned no directories. This usually indicates a permission issue.")
    print(f"\nScanned dirs: {total_dirs} files: {total_files}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()
    summarize_dir(root, max_depth=args.max_depth)
