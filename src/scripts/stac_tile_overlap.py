#!/usr/bin/env python3
"""
Intersect Capella tile names in a tiles list with the CAPELLA_* item JSON names
referenced by the `links` field in a STAC collection.json.

Example item link href:
  ../.../CAPELLA_C13_SP_GEO_HH_20251112022441_20251112022453/
       CAPELLA_C13_SP_GEO_HH_20251112022441_20251112022453.json

Usage:
  python stac_tile_overlap.py \
    --stac https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-ieee-data-contest/collection.json \
    --tiles tiles.txt \
    --out overlap_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple
from urllib.request import urlopen


CAPELLA_NAME_RE = re.compile(r"(CAPELLA_[A-Z0-9_]+)")
CAPELLA_ITEM_JSON_RE = re.compile(r"(CAPELLA_[^/]+)\.json$")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json_from_path_or_url(stac: str) -> dict:
    if stac.startswith("http://") or stac.startswith("https://"):
        with urlopen(stac) as r:
            return json.loads(r.read().decode("utf-8"))
    else:
        return json.loads(_read_text(Path(stac)))


def _extract_capella_names_from_tiles_txt(lines: Iterable[str]) -> Set[str]:
    """
    tiles.txt can be:
      - plain names (one per line)
      - `ls -l` style lines where the name is last
    We just regex for CAPELLA_* anywhere in the line and take the last match.
    """
    names: Set[str] = set()
    for line in lines:
        matches = CAPELLA_NAME_RE.findall(line.strip())
        if matches:
            names.add(matches[-1])
    return names


def _extract_capella_names_from_collection_links(collection: dict) -> Set[str]:
    """
    Extract CAPELLA_* basenames from collection['links'][*]['href'] when rel == 'item'.
    """
    links = collection.get("links", [])
    names: Set[str] = set()

    for link in links:
        if link.get("rel") != "item":
            continue
        href = link.get("href", "")
        m = CAPELLA_ITEM_JSON_RE.search(href)
        if m:
            names.add(m.group(1))
        else:
            # if weird formatting, try generic CAPELLA_... match
            matches = CAPELLA_NAME_RE.findall(href)
            if matches:
                names.add(matches[-1])

    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stac", required=True, help="Path or URL to collection.json")
    ap.add_argument("--tiles", required=True, help="Path to tiles.txt")
    ap.add_argument("--out", default=None, help="Optional output JSON report path")
    args = ap.parse_args()

    tiles_path = Path(args.tiles)
    if not tiles_path.exists():
        print(f"ERROR: tiles file not found: {tiles_path}", file=sys.stderr)
        return 2

    collection = _load_json_from_path_or_url(args.stac)

    tile_names = _extract_capella_names_from_tiles_txt(_read_text(tiles_path).splitlines())
    stac_names = _extract_capella_names_from_collection_links(collection)
    overlaps = sorted(tile_names & stac_names)

    report = {
        "stac": args.stac,
        "tiles_file": str(tiles_path),
        "tiles_count": len(tile_names),
        "stac_item_links_count": sum(1 for l in collection.get("links", []) if l.get("rel") == "item"),
        "stac_item_names_count": len(stac_names),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }

    out = json.dumps(report, indent=2, sort_keys=False)
    print(out)

    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
