from src.chunking import WhitespaceTokenizer, chunk_document
from src.data import Document


def test_chunks_respect_token_limit_and_metadata():
    document = Document("doc-1", "Example", ["one two three", "four five six", "seven"], ("q1",))
    chunks = chunk_document(document, WhitespaceTokenizer(), chunk_size=4)
    assert [chunk.token_count for chunk in chunks] == [3, 4]
    assert chunks[0].source_sentence_indices == (0,)
    assert chunks[1].source_sentence_indices == (1, 2)
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
