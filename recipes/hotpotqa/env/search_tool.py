import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import torch
from FlagEmbedding import FlagAutoModel

from recipes.hotpotqa.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    EVIDENCE_SIDECAR_FILENAME,
    OfficialEvidenceStore,
    coerce_bool,
)

# Retrieval corpus root: defaults to <repo>/data/corpus/hotpotqa_corpus
# (index.bin + hpqa_corpus.jsonl). Override with HOTPOTQA_CORPUS_DATA_ROOT.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HOTPOTQA_CORPUS_DATA_ROOT = _REPO_ROOT / "data" / "corpus" / "hotpotqa_corpus"


def resolve_hotpotqa_corpus_data_root(corpus_data_dir: Optional[str] = None) -> Path:
    """Resolve the directory that stores HotpotQA retrieval files.

    Args:
        corpus_data_dir: Optional explicit directory from recipe YAML.

    Returns:
        Absolute path containing ``index.bin`` and ``hpqa_corpus.jsonl``.
    """
    raw = corpus_data_dir or os.environ.get("HOTPOTQA_CORPUS_DATA_ROOT") or str(_DEFAULT_HOTPOTQA_CORPUS_DATA_ROOT)
    return Path(raw).expanduser().resolve()


HOTPOTQA_CORPUS_DATA_ROOT = resolve_hotpotqa_corpus_data_root()
# Backward-compatible alias for older imports. New code should use HOTPOTQA_CORPUS_DATA_ROOT.
HOTPOTQA_DATA_ROOT = HOTPOTQA_CORPUS_DATA_ROOT
HOTPOTQA_INDEX_BIN = HOTPOTQA_CORPUS_DATA_ROOT / "index.bin"
# Passage text for decoding search hits; must match hpqa_corpus.jsonl used when building the index.
HOTPOTQA_CORPUS_JSONL = HOTPOTQA_CORPUS_DATA_ROOT / "hpqa_corpus.jsonl"
# hpqa_corpus.npy is only produced by env/build_index.py for embedding cache; not loaded at runtime.

# Default BGE checkpoint (local dir or Hugging Face hub id). Override via YAML `embedding_model_name` or
# `HOTPOTQA_EMBEDDING_MODEL` for portability.
DEFAULT_HOTPOTQA_EMBEDDING_MODEL = (
    os.environ.get("HOTPOTQA_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5").strip() or "BAAI/bge-large-en-v1.5"
)

logger = logging.getLogger(__name__)


def resolve_hotpotqa_evidence_sidecar(
    evidence_sidecar_path: Optional[str],
    *,
    corpus_data_dir: Path,
) -> Path:
    raw = (
        evidence_sidecar_path
        or os.environ.get("HOTPOTQA_EVIDENCE_SIDECAR")
        or str(corpus_data_dir / EVIDENCE_SIDECAR_FILENAME)
    )
    return Path(raw).expanduser().resolve()


def default_hotpotqa_embedding_device() -> str:
    """Read desired device string from env (e.g. cuda:N); see `normalize_embedding_device` for actual resolution."""
    return os.environ.get("HOTPOTQA_EMBEDDING_DEVICE", "cpu").strip() or "cpu"


def normalize_embedding_device(requested: str) -> str:
    """
    Resolve configured device to a FlagEmbedding-compatible `devices` string.

    Ray AgentFlowWorker processes often see **no GPUs** (no CUDA device assigned or visibility
    masked). Passing `cuda:*` then raises "no CUDA GPUs are available" — we fall back to `cpu`
    with a warning.
    """
    dev = (requested or "cpu").strip() or "cpu"
    if dev.lower() == "cpu":
        return "cpu"
    dl = dev.lower()
    if dl.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "embedding device %r requested but torch.cuda.is_available() is False; "
                "using cpu (typical for Ray agent workers without GPU allocation).",
                dev,
            )
            return "cpu"
        if ":" in dev:
            try:
                idx = int(dev.split(":")[-1])
            except ValueError:
                return dev
            if idx < 0 or idx >= torch.cuda.device_count():
                logger.warning(
                    "embedding device %r invalid for torch.cuda.device_count()=%s; using cpu.",
                    dev,
                    torch.cuda.device_count(),
                )
                return "cpu"
        return dev
    return dev


def resolve_hotpotqa_embedding_devices(
    embedding_devices: Optional[str],
    agent_flow_worker_index: Optional[int],
) -> Optional[str]:
    """Pick BGE `devices` for HotpotQAAgentFlow before constructing `HotpotQASearchToolLegacy`.

    Args:
        embedding_devices: Non-empty value from YAML ``embedding_devices``; if set, it wins.
        agent_flow_worker_index: Index of the Ray ``AgentFlowWorker`` (0 .. N-1).

    Returns:
        Device string for FlagEmbedding, or ``None`` to use ``HotpotQASearchToolLegacy`` defaults
        (``HOTPOTQA_EMBEDDING_DEVICE`` env, then ``cpu``).

    If environment variable ``HOTPOTQA_EMBEDDING_PER_WORKER_GPU`` is truthy (``1``/``true``/``yes``)
    and ``agent_flow_worker_index`` is not ``None``, uses ``cuda:{index}`` (after
    :func:`normalize_embedding_device`), so each of 4 workers can colocate BGE on its training GPU.
    """
    if embedding_devices is not None:
        s = str(embedding_devices).strip()
        if s and s.lower() != "null":
            return s
    flag = os.environ.get("HOTPOTQA_EMBEDDING_PER_WORKER_GPU", "").strip().lower()
    if flag in ("1", "true", "yes", "on") and agent_flow_worker_index is not None:
        raw = f"cuda:{int(agent_flow_worker_index)}"
        return normalize_embedding_device(raw)
    return None


@dataclass
class Passage:
    pid: int
    title: str
    text: str
    score: float = 0.0
    sentence_evidence: list[dict[str, Any]] = field(default_factory=list)

    def evidence_record(self) -> dict[str, Any]:
        """Return the compact, auditable retrieval record persisted by A0."""

        return {
            "pid": int(self.pid),
            "title": self.title,
            "text": self.text,
            "score": float(self.score),
            "sentence_evidence": self.sentence_evidence,
        }


@dataclass(frozen=True)
class CorpusParagraph:
    """One unchanged row from the existing paragraph corpus."""

    pid: int
    title: str
    text: str

    @property
    def retrieval_text(self) -> str:
        return f"{self.title} {self.text}".strip()


class HotpotQASearchToolLegacy:
    """
    Local HotpotQA FAISS retrieval; behavior matches the production-validated legacy SearchTool.

    Extensions vs. upstream (success-path semantics unchanged):
    - Data layout: `HOTPOTQA_CORPUS_DATA_ROOT` + `index.bin` / `hpqa_corpus.jsonl`
    - Process-wide shared index/corpus/model to avoid reloading every trajectory
    - `_format_results` bounds-checks ids (legacy direct indexing could raise IndexError)
    - If `encode_queries` returns torch.Tensor, converts via `.cpu().numpy()` for FAISS CPU index I/O

    Upstream often uses `FlagAutoModel.from_finetuned(..., devices="cpu")`; CPU encode is recommended
    for training. If `HOTPOTQA_EMBEDDING_DEVICE=cuda:*`, call from the same thread (see hotpotqa_agent_flow).
    """

    _shared_lock = threading.RLock()
    _shared_key: Optional[str] = None
    _shared_index: Optional[faiss.Index] = None
    _shared_corpus: Optional[list[CorpusParagraph]] = None
    _shared_model: Optional[FlagAutoModel] = None
    _shared_evidence_key: Optional[str] = None
    _shared_evidence_store: Optional[OfficialEvidenceStore] = None

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_HOTPOTQA_EMBEDDING_MODEL,
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
        embedding_devices: Optional[str] = None,
        corpus_data_dir: Optional[str] = None,
        require_sentence_evidence: bool = False,
        evidence_sidecar_path: Optional[str] = None,
    ) -> None:
        self.data_dir = resolve_hotpotqa_corpus_data_root(corpus_data_dir)
        self.index_path = self.data_dir / "index.bin"
        self.corpus_path = self.data_dir / "hpqa_corpus.jsonl"
        self.embedding_model_name = embedding_model_name
        self.query_instruction = query_instruction
        env_requirement = os.environ.get("HOTPOTQA_REQUIRE_SENTENCE_EVIDENCE")
        self.require_sentence_evidence = coerce_bool(
            env_requirement if env_requirement is not None else require_sentence_evidence,
            name="HOTPOTQA_REQUIRE_SENTENCE_EVIDENCE",
        )
        self.evidence_schema_version = (
            EVIDENCE_SCHEMA_VERSION if self.require_sentence_evidence else "legacy-compatible"
        )
        self.evidence_sidecar_path = resolve_hotpotqa_evidence_sidecar(
            evidence_sidecar_path,
            corpus_data_dir=self.data_dir,
        )
        raw = (
            embedding_devices if embedding_devices is not None else default_hotpotqa_embedding_device()
        ).strip() or "cpu"
        self.embedding_devices = normalize_embedding_device(raw)
        if self.embedding_devices != raw:
            logger.info(
                "HotpotQASearchToolLegacy: effective embedding_devices=%r (from requested %r)",
                self.embedding_devices,
                raw,
            )

        self._index: Optional[faiss.Index] = None
        self._corpus: list[CorpusParagraph] = []
        self._model: Optional[FlagAutoModel] = None
        self._evidence_store: Optional[OfficialEvidenceStore] = None
        self._ensure_loaded()

    def __enter__(self):
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_loaded(self) -> None:
        cache_key = f"{self.data_dir}|{self.embedding_devices}|{self.embedding_model_name}"
        with self.__class__._shared_lock:
            if (
                self.__class__._shared_key != cache_key
                or self.__class__._shared_index is None
                or self.__class__._shared_corpus is None
                or self.__class__._shared_model is None
            ):
                if not self.index_path.exists():
                    raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
                if not self.corpus_path.exists():
                    raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

                logger.info("HotpotQASearchToolLegacy: loading FAISS index from %s", self.index_path)
                index = faiss.read_index(str(self.index_path))
                logger.info(
                    "HotpotQASearchToolLegacy: loading corpus jsonl from %s (may take several minutes)",
                    self.corpus_path,
                )
                corpus: list[CorpusParagraph] = []
                with self.corpus_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        title = str(rec.get("title", ""))
                        text = str(rec.get("text", ""))
                        corpus.append(CorpusParagraph(pid=len(corpus), title=title, text=text))

                logger.info(
                    "HotpotQASearchToolLegacy: loading FlagEmbedding model=%s devices=%s",
                    self.embedding_model_name,
                    self.embedding_devices,
                )
                model = FlagAutoModel.from_finetuned(
                    self.embedding_model_name,
                    query_instruction_for_retrieval=self.query_instruction,
                    devices=self.embedding_devices,
                )
                self.__class__._shared_key = cache_key
                self.__class__._shared_index = index
                self.__class__._shared_corpus = corpus
                self.__class__._shared_model = model

                if int(index.ntotal) != len(corpus):
                    message = (
                        f"FAISS index.ntotal ({int(index.ntotal)}) != hpqa_corpus.jsonl rows ({len(corpus)}). "
                        "Ids from search may be out of range and passages will be empty; "
                        "rebuild index.bin with the same jsonl or fix the corpus file."
                    )
                    if self.require_sentence_evidence:
                        raise ValueError(message)
                    logger.warning(message)

            self._index = self.__class__._shared_index
            self._corpus = self.__class__._shared_corpus or []
            self._model = self.__class__._shared_model
            if self.require_sentence_evidence:
                evidence_key = str(self.evidence_sidecar_path)
                if (
                    self.__class__._shared_evidence_key != evidence_key
                    or self.__class__._shared_evidence_store is None
                ):
                    logger.info(
                        "HotpotQASearchToolLegacy: loading official evidence sidecar from %s",
                        self.evidence_sidecar_path,
                    )
                    # Artifact preparation/validation performs the full SQLite
                    # integrity scan once. Repeating a 658 MB PRAGMA
                    # integrity_check in every AgentFlow process creates eight
                    # redundant NAS scans before the first model request.
                    store = OfficialEvidenceStore(
                        self.evidence_sidecar_path,
                        verify_integrity=False,
                    )
                    if store.paragraph_count != len(self._corpus):
                        raise ValueError(
                            f"Evidence sidecar has {store.paragraph_count} paragraphs for "
                            f"{len(self._corpus)} corpus rows"
                        )
                    self.__class__._shared_evidence_key = evidence_key
                    self.__class__._shared_evidence_store = store
                self._evidence_store = self.__class__._shared_evidence_store
            else:
                self._evidence_store = None

    def close(self) -> None:
        # Keep shared model/index alive for whole training process.
        # This matches legacy behavior where SearchTool is initialized once and reused.
        self._index = self.__class__._shared_index
        self._corpus = self.__class__._shared_corpus or []
        self._model = self.__class__._shared_model
        self._evidence_store = self.__class__._shared_evidence_store

    @property
    def evidence_store(self) -> OfficialEvidenceStore:
        if not self.require_sentence_evidence or self._evidence_store is None:
            raise RuntimeError("Official sentence evidence is not enabled for this search tool")
        return self._evidence_store

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            query = str(args["query"])
            embeddings = self._encode_queries([query])
            assert self._index is not None
            scores, ids = self._index.search(embeddings, 5)
            result_str = self._format_results(ids[0], scores[0])
            return {"content": result_str, "success": True}
        except Exception as e:
            return {"content": str(e), "success": False}

    def batch_execute(self, args_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch query with one encode/search pass, returning per-row errors on failure."""
        if not args_list:
            return []
        try:
            queries = [str(x["query"]) for x in args_list]
            embeddings = self._encode_queries(queries)
            assert self._index is not None
            scores, ids = self._index.search(embeddings, 5)
            results_str = [self._format_results(ids[i], scores[i]) for i in range(len(ids))]
            return [{"content": result_str, "success": True} for result_str in results_str]
        except Exception as e:
            logger.warning(
                "HotpotQASearchToolLegacy.batch_execute failed (%s queries): %s",
                len(args_list),
                e,
                exc_info=True,
            )
            return [{"content": str(e), "success": False} for _ in args_list]

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        self._ensure_loaded()
        with self.__class__._shared_lock:
            assert self._model is not None
            out = self._model.encode_queries(queries)
        # FAISS CPU Index::search expects host float32 ndarray; BGE on GPU may return torch.Tensor.
        if torch.is_tensor(out):
            out = out.detach().float().cpu().numpy()
        arr = np.asarray(out, dtype=np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr

    def _format_results(self, results, scores=None) -> str:
        results_list: list[Any] = []
        row_ids = [int(x) for x in np.asarray(results, dtype=np.int64).reshape(-1)]
        row_scores = (
            [float(x) for x in np.asarray(scores, dtype=np.float32).reshape(-1)]
            if scores is not None
            else [0.0] * len(row_ids)
        )
        for result, score in zip(row_ids, row_scores, strict=False):
            if result < 0 or result >= len(self._corpus):
                continue
            paragraph = self._corpus[result]
            retrieval_text = paragraph.retrieval_text
            if self.require_sentence_evidence:
                sentence_evidence = self.evidence_store.paragraph_evidence(
                    result,
                    expected_title=paragraph.title,
                    expected_text=paragraph.text,
                )
                results_list.append(
                    Passage(
                        pid=result,
                        title=paragraph.title,
                        text=retrieval_text,
                        score=score,
                        sentence_evidence=sentence_evidence,
                    ).evidence_record()
                )
            else:
                results_list.append(retrieval_text)
        if not results_list and self._corpus and row_ids and max(row_ids) >= len(self._corpus):
            logger.warning(
                "FAISS returned ids %s but corpus length is %s; dropping all hits.",
                row_ids[:10],
                len(self._corpus),
            )
        payload: dict[str, Any] = {"results": results_list}
        if self.require_sentence_evidence:
            payload["evidence_schema_version"] = self.evidence_schema_version
        return json.dumps(payload, ensure_ascii=False)


def parse_legacy_tool_result(content: str) -> list[Passage]:
    """Parse paragraph results while retaining the legacy string payload."""
    passages: list[Passage] = []
    try:
        payload = json.loads(content)
        results = payload.get("results", [])
        for idx, item in enumerate(results):
            if isinstance(item, dict):
                sentence_evidence = item.get("sentence_evidence")
                passages.append(
                    Passage(
                        pid=int(item.get("pid", idx)),
                        title=str(item.get("title", "")),
                        text=str(item.get("text", "")),
                        score=float(item.get("score", 0.0)),
                        sentence_evidence=(
                            list(sentence_evidence) if isinstance(sentence_evidence, list) else []
                        ),
                    )
                )
            else:
                passages.append(Passage(pid=idx, title="", text=str(item), score=0.0))
    except Exception:
        return []
    return passages
