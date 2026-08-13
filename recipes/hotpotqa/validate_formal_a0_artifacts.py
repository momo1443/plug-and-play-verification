#!/usr/bin/env python3
"""Fail-closed preflight for the formal shared-AgentFlow A0 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow.parquet as pq

from recipes.hotpotqa.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    OfficialEvidenceStore,
    parse_evidence_id,
)

# Bump when the *logic* of validate_formal_a0_artifacts changes so stale caches
# from an older code version are ignored instead of silently reused.
PREFLIGHT_CACHE_VERSION = 1


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, int]:
    """Cheap identity of a frozen artifact: byte size + nanosecond mtime.

    Any content change to a normal file also changes size or mtime, so an
    unchanged (size, mtime_ns) pair lets us trust a previously computed result
    without re-reading multi-GB artifacts from NAS on every launch.
    """

    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def load_cache(cache_path: Path | None) -> dict[str, Any]:
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_cache_atomic(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, cache_path)


def validate_formal_a0_artifacts(
    *,
    validation_path: Path,
    corpus_dir: Path,
    evidence_sidecar_path: Path,
    expected_validation_rows: int = 7405,
    minimum_gold_mapping_rate: float = 0.99,
    verify_hashes: bool = True,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    corpus_path = corpus_dir / "hpqa_corpus.jsonl"
    index_path = corpus_dir / "index.bin"
    embeddings_path = corpus_dir / "hpqa_corpus.npy"
    for path in (validation_path, corpus_path, index_path, evidence_sidecar_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required formal A0 artifact is missing: {path}")

    # Every artifact whose bytes influence this result is fingerprinted by
    # (size, mtime_ns). An identical fingerprint set lets us reuse the cached
    # summary and skip the FAISS load, corpus scan and full-file hashing.
    artifact_fingerprints = {
        "validation_parquet": fingerprint(validation_path),
        "corpus_jsonl": fingerprint(corpus_path),
        "index_bin": fingerprint(index_path),
        "evidence_sidecar": fingerprint(evidence_sidecar_path),
    }
    if embeddings_path.is_file():
        artifact_fingerprints["embeddings_npy"] = fingerprint(embeddings_path)
    cache_key = {
        "cache_version": PREFLIGHT_CACHE_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "expected_validation_rows": expected_validation_rows,
        "minimum_gold_mapping_rate": minimum_gold_mapping_rate,
        "verify_hashes": verify_hashes,
        "artifact_fingerprints": artifact_fingerprints,
    }
    if use_cache:
        cached = load_cache(cache_path).get("artifact_preflight")
        if isinstance(cached, dict) and cached.get("key") == cache_key:
            summary = dict(cached["summary"])
            summary["cache"] = "hit"
            return summary

    validation_rows = int(pq.read_metadata(validation_path).num_rows)
    if expected_validation_rows > 0 and validation_rows != expected_validation_rows:
        raise ValueError(
            f"Validation parquet has {validation_rows} rows; expected {expected_validation_rows}"
        )

    corpus_rows = 0
    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                raise ValueError(f"Frozen corpus has a blank row at line {line_number}")
            record = json.loads(line)
            if not str(record.get("title") or "") or not str(record.get("text") or ""):
                raise ValueError(f"Frozen corpus line {line_number} is missing title/text")
            corpus_rows += 1
    if corpus_rows == 0:
        raise ValueError(f"Frozen corpus is empty: {corpus_path}")

    index = faiss.read_index(str(index_path))
    if int(index.ntotal) != corpus_rows:
        raise ValueError(
            f"FAISS index has {int(index.ntotal)} vectors for {corpus_rows} corpus rows"
        )
    embeddings_shape = None
    if embeddings_path.is_file():
        embeddings = np.load(embeddings_path, mmap_mode="r")
        embeddings_shape = [int(value) for value in embeddings.shape]
        if embeddings.ndim != 2 or embeddings_shape[0] != corpus_rows:
            raise ValueError(
                f"Embedding shape {embeddings.shape} does not match {corpus_rows} corpus rows"
            )

    store = OfficialEvidenceStore(evidence_sidecar_path)
    if store.paragraph_count != corpus_rows:
        raise ValueError(
            f"Evidence sidecar has {store.paragraph_count} paragraphs for {corpus_rows} corpus rows"
        )
    if store.sample_count("validation") != validation_rows:
        raise ValueError(
            f"Evidence sidecar has {store.sample_count('validation')} validation samples for "
            f"{validation_rows} parquet rows"
        )

    total_gold = int(store.metadata.get("gold_fact_count_validation", 0))
    unresolved_gold = int(store.metadata.get("unresolved_gold_fact_count_validation", 0))
    if total_gold <= 0 or unresolved_gold < 0 or unresolved_gold > total_gold:
        raise ValueError("Evidence sidecar has invalid validation gold mapping counts")
    mapping_rate = (total_gold - unresolved_gold) / total_gold
    if mapping_rate < minimum_gold_mapping_rate:
        raise ValueError(
            f"Validation supporting-fact mapping rate {mapping_rate:.6f} is below "
            f"the frozen gate {minimum_gold_mapping_rate:.6f}"
        )

    samples = store.list_samples("validation")
    official_qids: set[str] = set()
    eligible_samples = 0
    example_evidence_ids: list[str] = []
    for expected_index, sample in enumerate(samples):
        if sample.row_index != expected_index or sample.sample_key != f"validation:{expected_index}":
            raise ValueError(f"Evidence sample order is not contiguous at validation:{expected_index}")
        if sample.official_qid in official_qids:
            raise ValueError(f"Duplicate official qid: {sample.official_qid}")
        official_qids.add(sample.official_qid)
        if not sample.unresolved_gold_facts:
            eligible_samples += 1
        for evidence_id in sample.gold_evidence_ids:
            pid, _ = parse_evidence_id(evidence_id)
            if pid >= corpus_rows:
                raise ValueError(f"Gold evidence PID is outside corpus: {evidence_id}")
            if len(example_evidence_ids) < 5:
                example_evidence_ids.append(evidence_id)

    verified_hashes: dict[str, str] = {}
    if verify_hashes:
        expected_hashes = {
            "validation_parquet": store.metadata.get("sha256_validation_parquet"),
            "corpus_jsonl": store.metadata.get("sha256_corpus_jsonl"),
        }
        paths = {
            "validation_parquet": validation_path,
            "corpus_jsonl": corpus_path,
        }
        for name, expected_hash in expected_hashes.items():
            if not expected_hash:
                raise ValueError(f"Evidence sidecar has no frozen SHA-256 for {name}")
            actual_hash = _sha256(paths[name])
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Frozen artifact hash mismatch for {name}: {actual_hash} != {expected_hash}"
                )
            verified_hashes[name] = actual_hash

    summary = {
        "status": "ok",
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "validation_rows": validation_rows,
        "corpus_rows": corpus_rows,
        "index_vectors": int(index.ntotal),
        "index_dimension": int(index.d),
        "embeddings_shape": embeddings_shape,
        "validation_gold_facts": total_gold,
        "unresolved_validation_gold_facts": unresolved_gold,
        "validation_gold_mapping_rate": mapping_rate,
        "evidence_metric_eligible_rows": eligible_samples,
        "verified_hashes": verified_hashes,
        "example_official_evidence_ids": example_evidence_ids,
        "cache": "miss",
    }
    if use_cache and cache_path is not None:
        cache = load_cache(cache_path)
        cache["artifact_preflight"] = {"key": cache_key, "summary": summary}
        write_cache_atomic(cache_path, cache)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation_path", type=Path, required=True)
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--evidence_sidecar_path", type=Path, required=True)
    parser.add_argument("--expected_validation_rows", type=int, default=7405)
    parser.add_argument("--minimum_gold_mapping_rate", type=float, default=0.99)
    parser.add_argument("--skip_hash_verification", action="store_true")
    args = parser.parse_args()
    summary = validate_formal_a0_artifacts(
        validation_path=args.validation_path.expanduser().resolve(),
        corpus_dir=args.corpus_dir.expanduser().resolve(),
        evidence_sidecar_path=args.evidence_sidecar_path.expanduser().resolve(),
        expected_validation_rows=args.expected_validation_rows,
        minimum_gold_mapping_rate=args.minimum_gold_mapping_rate,
        verify_hashes=not args.skip_hash_verification,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
