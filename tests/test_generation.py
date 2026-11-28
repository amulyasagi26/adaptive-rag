from src.chunking import Chunk
from src.generation import build_generation_prompt


def test_generation_prompt_contains_question_and_retrieved_context():
    chunk = Chunk("d:0", "d", "Example", "Alice worked here.", 3, (0,), ("q1",))

    messages = build_generation_prompt("Who worked here?", [{"score": 0.9, "chunk": chunk}])

    assert messages[0]["role"] == "system"
    assert "Alice worked here." in messages[1]["content"]
    assert "Who worked here?" in messages[1]["content"]
