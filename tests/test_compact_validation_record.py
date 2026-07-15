import unittest

from agent_r1.trainer.compact_validation_record import (
    build_validation_record,
    extract_final_answer,
    extract_question,
    extract_search_queries,
)
from recipes.hotpotqa.output_parsing import split_native_thinking
from recipes.hotpotqa.reward_fn import _extract_answer_from_solution


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
            evidence_schema_version="hotpotqa-official-sentence-v1",
            search_steps=search_steps,
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
        self.assertEqual(record["search_steps"], search_steps)
        self.assertEqual(record["answer"], "Answer")
        self.assertEqual(record["sample_key"], "validation:0")
        self.assertEqual(record["official_qid"], "official-qid-0")

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

    def test_missing_complete_answer_is_null(self):
        self.assertIsNone(extract_final_answer("<answer>truncated"))

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


if __name__ == "__main__":
    unittest.main()
