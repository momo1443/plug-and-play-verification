from recipes.deepscaler.clean_data import DRESS_PROBLEM, quality_decisions


def test_conflicts_are_quarantined_without_using_model_predictions():
    report = quality_decisions([
        {"problem": "Question A", "answer": "24/5"},
        {"problem": "Question A", "answer": r"$\dfrac{24}5$"},
        {"problem": "Question B", "answer": "1"},
        {"problem": "Question B", "answer": "2"},
        {"problem": DRESS_PROBLEM, "answer": "54"},
        {"problem": "In the diagram shown, find x.", "answer": "2"},
        {"problem": "In the diagram shown, find x. [asy]draw((0,0)--(1,0));[/asy]", "answer": "2"},
    ])
    assert report['excluded'] == {2: 'conflicting_reference_answers', 3: 'conflicting_reference_answers', 5: 'missing_referenced_diagram'}
    assert report['corrections'][4]['answer'] == '66'
    assert len(report['equivalent_label_groups']) == 1


def test_all_missing_or_invalid_rows_are_accounted_for():
    report = quality_decisions([None, {"problem": "", "answer": "3"}, {"problem": "Q", "answer": None}])
    assert report['source_rows'] == 3
    assert len(report['excluded']) == 3


def test_unavailable_symbolic_parse_is_reported_without_aborting_data_audit():
    from unittest.mock import patch
    from recipes.deepscaler.clean_data import answer_parse_status
    with patch('recipes.deepscaler.math_expressions._parsed_answer',
               side_effect=RuntimeError('symbolic parsing failed')):
        assert answer_parse_status('unreadable label')[1] == 'unsupported_answer_format'
