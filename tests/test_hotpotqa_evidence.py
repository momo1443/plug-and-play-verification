import json
import tempfile
import unittest
from pathlib import Path

import threading
from dataclasses import field
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from recipes.hotpotqa.build_evidence_sidecar import build_evidence_sidecar
from recipes.hotpotqa.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    OfficialEvidenceStore,
    coerce_bool,
    make_evidence_id,
    normalize_title,
    parse_evidence_id,
    sentence_evidence_records,
)
from pnp_test_utils import load_source

search_module = load_source(
    "recipes/hotpotqa/env/search_tool.py", threading=threading, field=field, np=np,
    EVIDENCE_SCHEMA_VERSION=EVIDENCE_SCHEMA_VERSION, OfficialEvidenceStore=OfficialEvidenceStore,
    coerce_bool=coerce_bool,
)
CorpusParagraph = search_module.CorpusParagraph
HotpotQASearchToolLegacy = search_module.HotpotQASearchToolLegacy
parse_legacy_tool_result = search_module.parse_legacy_tool_result


def _agent_row(question: str, answer: str, split: str, index: int) -> dict:
    return {
        "data_source": "hotpotqa_distractor",
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "extra_info": {
            "index": index,
            "question_id": f"{split}_{index}",
            "split": split,
            "type": "bridge",
            "level": "hard",
        },
    }


def _official_row(qid: str, question: str, answer: str, title: str) -> dict:
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": "bridge",
        "level": "hard",
        "supporting_facts": {"title": [title], "sent_id": [1]},
        "context": {
            "title": [title],
            "sentences": [["Sentence zero.", "Sentence one."]],
        },
    }


class HotpotQAEvidenceTest(unittest.TestCase):
    def _artifacts(self, root: Path) -> tuple[Path, Path, Path]:
        corpus_dir = root / "corpus"
        corpus_dir.mkdir()
        corpus_rows = [
            {"title": "Train Title", "text": "Sentence zero. Sentence one."},
            {"title": "Validation Title", "text": "Sentence zero. Sentence one."},
        ]
        (corpus_dir / "hpqa_corpus.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in corpus_rows), encoding="utf-8"
        )
        train_parquet = root / "train.parquet"
        validation_parquet = root / "validation.parquet"
        pd.DataFrame([_agent_row("Train?", "Train answer", "train", 0)]).to_parquet(
            train_parquet, index=False
        )
        pd.DataFrame(
            [_agent_row("Validation?", "Validation answer", "validation", 0)]
        ).to_parquet(validation_parquet, index=False)
        train_source = root / "official_train.parquet"
        validation_source = root / "official_validation.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [_official_row("official-train", "Train?", "Train answer", "Train Title")]
            ),
            train_source,
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    _official_row(
                        "official-validation",
                        "Validation?",
                        "Validation answer",
                        "Validation Title",
                    )
                ]
            ),
            validation_source,
        )
        sidecar = corpus_dir / "hotpotqa_evidence_v1.sqlite3"
        build_evidence_sidecar(
            train_source=train_source,
            validation_source=validation_source,
            train_parquet=train_parquet,
            validation_parquet=validation_parquet,
            corpus_path=corpus_dir / "hpqa_corpus.jsonl",
            output_path=sidecar,
        )
        return corpus_dir, validation_parquet, sidecar

    def test_evidence_id_is_stable_and_reversible(self):
        evidence_id = make_evidence_id(509, 3)
        self.assertEqual(evidence_id, f"{EVIDENCE_SCHEMA_VERSION}:509:3")
        self.assertEqual(parse_evidence_id(evidence_id), (509, 3))

    def test_records_use_supplied_source_sentences_without_runtime_splitting(self):
        records = sentence_evidence_records(
            pid=7,
            title="A_Title",
            sentences=["First sentence. Still source sentence.", "Second?"],
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], "First sentence. Still source sentence.")
        self.assertEqual(records[1]["evidence_id"], make_evidence_id(7, 1))
        self.assertEqual(records[0]["normalized_title"], normalize_title("A_Title"))

    def test_sidecar_aligns_official_qid_gold_and_frozen_pid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir, _, sidecar = self._artifacts(Path(temp_dir))
            store = OfficialEvidenceStore(sidecar)
            sample = store.sample("validation", 0)
            self.assertEqual(sample.official_qid, "official-validation")
            self.assertEqual(sample.gold_evidence_ids, (make_evidence_id(1, 1),))
            self.assertFalse(sample.unresolved_gold_facts)
            evidence = store.paragraph_evidence(
                1,
                expected_title="Validation Title",
                expected_text="Sentence zero. Sentence one.",
            )
            self.assertEqual([row["sentence_id"] for row in evidence], [0, 1])
            self.assertTrue((corpus_dir / "hotpotqa_evidence_v1.manifest.json").is_file())

    def test_search_payload_retains_sidecar_evidence_for_certificate_verification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, _, sidecar = self._artifacts(Path(temp_dir))
            tool = object.__new__(HotpotQASearchToolLegacy)
            tool.require_sentence_evidence = True
            tool.evidence_schema_version = EVIDENCE_SCHEMA_VERSION
            tool._evidence_store = OfficialEvidenceStore(sidecar)
            tool._corpus = [
                CorpusParagraph(
                    pid=0,
                    title="Train Title",
                    text="Sentence zero. Sentence one.",
                )
            ]
            payload = json.loads(tool._format_results(np.array([0]), np.array([0.625])))
            result = payload["results"][0]
            self.assertEqual(
                [sentence["evidence_id"] for sentence in result["sentence_evidence"]],
                [make_evidence_id(0, 0), make_evidence_id(0, 1)],
            )
            passage = parse_legacy_tool_result(json.dumps(payload))[0]
            self.assertIn("Train Title Sentence zero.", passage.text)
            self.assertEqual(passage.sentence_evidence, result["sentence_evidence"])


    def test_boolean_strings_do_not_use_python_truthiness(self):
        self.assertFalse(coerce_bool("false"))
        self.assertTrue(coerce_bool("true"))
        with self.assertRaises(ValueError):
            coerce_bool("maybe")


if __name__ == "__main__":
    unittest.main()
