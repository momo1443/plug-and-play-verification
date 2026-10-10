from agent_r1.evaluation.answers import final_answer_record


def test_thoughts_and_format_instructions_are_not_submissions():
    assert final_answer_record(r"I should use `\boxed{...}` before finishing.") is None
    assert final_answer_record(r"\boxed{...}") is None
    assert final_answer_record(r"<think>The answer is 5. \boxed{5}") is None
    assert final_answer_record(r"The answer is 5. \boxed{5}", require_think_close=True) is None
    assert final_answer_record(r"<think>Earlier \boxed{5}</think>Still checking.") is None


def test_complete_thinking_selects_only_visible_submission_and_keeps_history():
    text = r"Earlier \boxed{5}</think>Corrected calculation. \boxed{7}"
    record = final_answer_record(text, require_think_close=True)
    assert record['answer'] == '7'
    assert r"Earlier \boxed{5}" in record['reasoning']
    assert final_answer_record(r"\boxed{5}, final \boxed{") is None
