from src.evaluation import answer_exact_match, answer_token_f1


def test_answer_exact_match_normalizes_articles_and_punctuation():
    assert answer_exact_match("The Eiffel Tower!", "eiffel tower") == 1.0


def test_answer_token_f1_handles_partial_overlap():
    assert answer_token_f1("Alice went home", "Alice went") == 0.8
    assert answer_token_f1("unrelated", "answer") == 0.0
