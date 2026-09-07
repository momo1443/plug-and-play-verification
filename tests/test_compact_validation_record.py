import unittest

from agent_r1.trainer.compact_validation_record import (
    _lenient_answer_extract,
    build_validation_record,
    extract_final_answer,
    extract_question,
    extract_search_queries,
)
from recipes.hotpotqa.final_answer_protocol import RAW_FINAL_ANSWER_PROTOCOL, require_raw_final_only
from recipes.hotpotqa.output_parsing import split_native_thinking
from recipes.hotpotqa.reward_fn import _extract_answer_from_solution
from recipes.hotpotqa_a9.protocol import A9_FINISH_PROTOCOL
from recipes.hotpotqa_lr.protocol import LR_FINISH_PROTOCOL


class CompactValidationRecordTest(unittest.TestCase):
    def test_builds_formal_a0_record_with_thinking_and_evidence(self):
        search_steps = [
            {
                "search_index": 1,
                "assistant_turn": 1,
                "source": "model",
                "query": "target query",
                "success": True,
                "returned_evidence": [
                    {
                        "pid": 7,
                        "title": "Title",
                        "text": "Title Evidence text.",
                        "score": 0.9,
                        "sentence_evidence": [
                            {
                                "evidence_id": "hotpotqa-official-sentence-v1:7:0",
                                "pid": 7,
                                "title": "Title",
                                "sentence_id": 0,
                                "text": "Evidence text.",
                            }
                        ],
                    }
                ],
            }
        ]
        record = build_validation_record(
            sample_index=0,
            raw_prompt=[{"role": "user", "content": "Question?"}],
            decoded_input="rendered prompt",
            output_text="reasoning</think><answer>Answer</answer>",
            ground_truth="Answer",
            score=1.0,
            extra_info={
                "question_id": "qid-0",
            },
            thinking_mode="native",
            thinking_steps=[{"turn": 1, "content": "reasoning", "complete": True}],
            force_first_search=False,
            final_answer_protocol=RAW_FINAL_ANSWER_PROTOCOL,
            evidence_schema_version="hotpotqa-official-sentence-v1",
            search_steps=search_steps,
            local_reasoning_transitions=[{"reason_step": {"ref": "r1"}}],
            executed_queries=["target query"],
            num_turns=4,
            sample_key="validation:0",
            dataset_question_id="validation_0",
            official_qid="official-qid-0",
            gold_evidence_ids=["hotpotqa-official-sentence-v1:7:0"],
            unresolved_gold_facts=[],
            evidence_metrics={"evidence_metrics_eligible": True},
        )
        self.assertEqual(record["thinking_mode"], "native")
        self.assertEqual(record["qwen_thinking"][0]["content"], "reasoning")
        self.assertFalse(record["force_first_search"])
        self.assertEqual(record["final_answer_protocol"], RAW_FINAL_ANSWER_PROTOCOL)
        self.assertEqual(record["search_steps"], search_steps)
        self.assertEqual(record["local_reasoning_transitions"], [{"reason_step": {"ref": "r1"}}])
        self.assertEqual(record["answer"], "Answer")
        self.assertEqual(record["sample_key"], "validation:0")
        self.assertEqual(record["official_qid"], "official-qid-0")

    def test_raw_final_contract_rejects_removed_force_switch(self):
        self.assertEqual(require_raw_final_only({}), RAW_FINAL_ANSWER_PROTOCOL)
        with self.assertRaisesRegex(RuntimeError, "has been removed"):
            require_raw_final_only({"HOTPOTQA_FORCE_FINAL_ANSWER": "false"})

    def test_extracts_question_from_raw_prompt(self):
        raw_prompt = [{"role": "user", "content": "What is the answer?"}]
        self.assertEqual(extract_question(raw_prompt, "large rendered prompt"), "What is the answer?")

    def test_falls_back_to_rendered_user_query_block(self):
        rendered = "system text\n### User Query\nFallback question?\n\n### History Actions\nNone"
        self.assertEqual(extract_question(None, rendered), "Fallback question?")

    def test_extracts_generated_search_queries_in_order(self):
        output = """
<tool_call><function=search><parameter=query>first query</parameter></function></tool_call>
tool result
<tool_call><function=search><parameter=query>
second query
</parameter></function></tool_call>
"""
        self.assertEqual(extract_search_queries(output), ["first query", "second query"])

    def test_extracts_only_last_complete_answer(self):
        output = "mention <answer> early\n<answer> final answer \n</answer>"
        self.assertEqual(extract_final_answer(output), "final answer")

    def test_answer_extraction_ignores_answer_tags_inside_thinking(self):
        output = "wrong <answer>thought</answer></think><answer>visible</answer>"
        self.assertEqual(extract_final_answer(output), "visible")
        self.assertEqual(_extract_answer_from_solution(output), "visible")

    def test_missing_complete_answer_returns_text(self):
        # Lenient extraction returns the full text as fallback when there
        # is no complete <answer>...</answer> pair.
        self.assertEqual(extract_final_answer("<answer>truncated"), "<answer>truncated")

    def test_extracts_local_reasoning_finish_answer(self):
        output = (
            '<tool_call>{"name":"finish","arguments":{"status":"answer","answer":"Paris",'
            '"reason_step":{"ref":"r1"}}}</tool_call>'
        )
        self.assertEqual(
            extract_final_answer(output, LR_FINISH_PROTOCOL),
            "Paris",
        )

    def test_splits_prompt_owned_native_thinking(self):
        thinking, visible, complete = split_native_thinking(
            "reason about <tool_call>fake</tool_call></think>\n<answer>yes</answer>"
        )
        self.assertEqual(thinking, "reason about <tool_call>fake</tool_call>")
        self.assertEqual(visible, "<answer>yes</answer>")
        self.assertTrue(complete)

    def test_tolerates_generated_think_open_tag(self):
        thinking, visible, complete = split_native_thinking(
            "  <think>\n推理过程\n</think>\n<tool_call>real</tool_call>"
        )
        self.assertEqual(thinking, "推理过程")
        self.assertEqual(visible, "<tool_call>real</tool_call>")
        self.assertTrue(complete)

    def test_unclosed_thinking_has_no_visible_output(self):
        thinking, visible, complete = split_native_thinking("unfinished reasoning")
        self.assertEqual(thinking, "unfinished reasoning")
        self.assertEqual(visible, "")
        self.assertFalse(complete)


# -- Lenient fallback for A8/A9 protocols --


class LenientFallbackTest(unittest.TestCase):
    """Tests for extract_final_answer falling back to <answer> tags when
    the A8/A9 finish tool call fails to parse."""

    def test_lr_finish_invalid_envelope_falls_back_to_answer_tag(self):
        """When LR finish envelope is invalid but <answer> tag exists,
        extract_final_answer should fall back to the tag."""
        _TOOL_OPEN = chr(60) + "tool_call" + chr(62)
        _TOOL_CLOSE = chr(60) + "/tool_call" + chr(62)
        # Missing reason_step -> envelope_valid=False
        output = (
            f"{_TOOL_OPEN}\n"
            '{"name": "finish", "arguments": {"status": "answer", "answer": "American"}}\n'
            f"{_TOOL_CLOSE}\n"
            "<answer>American</answer>"
        )
        self.assertEqual(
            extract_final_answer(output, LR_FINISH_PROTOCOL),
            "American",
        )

    def test_lr_finish_no_tool_call_falls_back_to_answer_tag(self):
        """When no finish tool call exists but <answer> tag does,
        extract_final_answer should fall back to the tag."""
        output = "<answer>American</answer>"
        self.assertEqual(
            extract_final_answer(output, LR_FINISH_PROTOCOL),
            "American",
        )

    def test_lr_finish_valid_envelope_not_fallen_back(self):
        """When finish envelope is valid, its answer should be used directly."""
        _TOOL_OPEN = chr(60) + "tool_call" + chr(62)
        _TOOL_CLOSE = chr(60) + "/tool_call" + chr(62)
        output = (
            f"{_TOOL_OPEN}\n"
            '{"name": "finish", "arguments": {"status": "answer", "answer": "Paris", '
            '"reason_step": {"ref": "r1", "op": "extract_answer_candidate", '
            '"premises": [], "inputs": [], "output": {"answer_candidate": "Paris"}}}}\n'
            f"{_TOOL_CLOSE}"
        )
        self.assertEqual(
            extract_final_answer(output, LR_FINISH_PROTOCOL),
            "Paris",
        )

    def test_a9_finish_no_tool_call_falls_back_to_answer_tag(self):
        """When no finish tool call exists but <answer> tag does,
        extract_final_answer should fall back for A9 protocol too."""
        output = "<answer>American</answer>"
        self.assertEqual(
            extract_final_answer(output, A9_FINISH_PROTOCOL),
            "American",
        )

    def test_a9_finish_valid_answer_not_fallen_back(self):
        """When A9 finish call has an answer, it should be used directly."""
        _TOOL_OPEN = chr(60) + "tool_call" + chr(62)
        _TOOL_CLOSE = chr(60) + "/tool_call" + chr(62)
        output = (
            f"{_TOOL_OPEN}\n"
            '{"name": "finish", "arguments": {"status": "answer", "answer": "Paris", '
            '"certificate": {"source_id": "passage:1", "support_span": "test", '
            '"answer_span": "Paris"}}}\n'
            f"{_TOOL_CLOSE}"
        )
        self.assertEqual(
            extract_final_answer(output, A9_FINISH_PROTOCOL),
            "Paris",
        )

    def test_lenient_answer_extract_basic(self):
        self.assertEqual(_lenient_answer_extract("<answer>X</answer>"), "X")

    def test_lenient_answer_extract_no_tag_returns_text(self):
        self.assertEqual(_lenient_answer_extract("plain text answer"), "plain text answer")

    def test_lenient_answer_extract_empty_returns_none(self):
        self.assertIsNone(_lenient_answer_extract(""))


if __name__ == "__main__":
    unittest.main()
