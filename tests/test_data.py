from src.data import parse_hotpot_record


def test_hotpot_record_keeps_labels_outside_context_text():
    record = {
        "_id": "q1", "question": "Who?", "answer": "Alice",
        "context": [["Page", ["Alice worked here."]]],
        "supporting_facts": {"title": ["Page"], "sent_id": [0]},
    }
    parsed = parse_hotpot_record(record)
    assert parsed.question_id == "q1"
    assert parsed.contexts == [{"title": "Page", "sentences": ["Alice worked here."]}]
    assert parsed.supporting_facts == [("Page", 0)]


def test_hotpot_record_dictionary_context():
    record = {
        "_id": "q2", "question": "Who?", "answer": "Alice",
        "context": {
            "title": ["Page"],
            "sentences": [["Alice worked here."]],
        },
        "supporting_facts": {"title": ["Page"], "sent_id": [0]},
    }
    parsed = parse_hotpot_record(record)
    assert parsed.question_id == "q2"
    assert parsed.contexts == [{"title": "Page", "sentences": ["Alice worked here."]}]
    assert parsed.supporting_facts == [("Page", 0)]