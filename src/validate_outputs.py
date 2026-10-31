#!/usr/bin/env python3
"""Memory-bounded full-file validator for this challenge's paired TSV outputs.

Unlike the provided validator, this reads one S1 row from each output at a time.
The two output files must have identical S1 row order, as the standalone exporter does.
"""
from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path


def source_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as stream:
        header = next(stream).rstrip("\r\n").split("\t")
        if not header or header[0] != "entity_id":
            raise ValueError(f"unexpected header in {path}: {header}")
        return {line.partition("\t")[0] for line in stream if line.strip()}


def parse_row(line: str, expected_s1: str, title: str) -> list[str]:
    if line.count("\t") != 1:
        raise ValueError(f"{title}: expected exactly one tab for {expected_s1}")
    s1, rest = line.rstrip("\r\n").split("\t")
    if s1 != expected_s1:
        raise ValueError(f"{title}: expected S1 {expected_s1}, got {s1}")
    if not rest:
        return []
    ids = rest.split(",")
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError(f"{title}: empty or duplicate ID in list for {s1}")
    if any(not mid.startswith(("S2-", "S3-")) for mid in ids):
        raise ValueError(f"{title}: non-S2/S3 ID for {s1}")
    return ids


def validate(matching: Path, candidates: Path, test_dir: Path, check_ids: bool) -> None:
    required = source_ids(test_dir / "test_source1.tsv")
    total_required = len(required)
    targets = None
    if check_ids:
        targets = source_ids(test_dir / "test_source2.tsv")
        targets.update(source_ids(test_dir / "test_source3.tsv"))
    matched_links = candidate_links = rows = 0
    with matching.open(encoding="utf-8") as mf, candidates.open(encoding="utf-8") as cf:
        if mf.readline().rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
            raise ValueError("wrong matching_results.tsv header")
        if cf.readline().rstrip("\r\n") != "source1_entity_id\tcandidate_entity_ids":
            raise ValueError("wrong candidate_pairs.tsv header")
        for match_line, candidate_line in zip_longest(mf, cf):
            if match_line is None or candidate_line is None:
                raise ValueError("the two output files have different row counts")
            s1 = match_line.partition("\t")[0]
            if s1 not in required:
                raise ValueError(f"duplicate or unknown S1 row: {s1}")
            required.remove(s1)
            mids = parse_row(match_line, s1, "matching_results.tsv")
            cids = parse_row(candidate_line, s1, "candidate_pairs.tsv")
            candidate_set = set(cids)
            if not set(mids).issubset(candidate_set):
                raise ValueError(f"matched ID absent from candidates for {s1}")
            if targets is not None:
                if any(mid not in targets for mid in mids):
                    raise ValueError(f"unknown matched ID for {s1}")
                if any(cid not in targets for cid in cids):
                    raise ValueError(f"unknown candidate ID for {s1}")
            rows += 1
            matched_links += len(mids)
            candidate_links += len(cids)
            if rows % 250_000 == 0:
                print(f"checked {rows:,}/{total_required:,} S1 rows", flush=True)
    if required:
        raise ValueError(f"{len(required):,} required S1 rows are missing")
    print(f"PASS: {rows:,} S1 rows, {matched_links:,} final links, "
          f"{candidate_links:,} candidate links; ID existence checked={check_ids}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()
    validate(args.matching, args.candidate, args.test_dir, args.check_ids)


if __name__ == "__main__":
    main()
