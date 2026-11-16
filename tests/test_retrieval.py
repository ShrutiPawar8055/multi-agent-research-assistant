import pytest

from researchcrew.retrieval import BM25Index, KnowledgeBase, chunk_text
from researchcrew.text import content_words, split_sentences


def test_sentences_keep_trailing_citations_attached():
    parts = split_sentences("RAG grounds answers [1]. It cites sources. [2] Next one here.")
    assert parts == ["RAG grounds answers [1].", "It cites sources. [2]", "Next one here."]


def test_content_words_drop_stopwords_and_plurals():
    assert content_words("The chunks of the documents") == ["chunk", "document"]


def test_chunking_overlaps_and_covers_everything():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(text, "a.md", size=10, overlap=3)
    assert [c.position for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert "w24" in chunks[-1].text


def test_chunking_validates_arguments_and_handles_empty():
    assert chunk_text("   ", "a.md") == []
    with pytest.raises(ValueError):
        chunk_text("x y z", "a.md", size=5, overlap=5)


def test_bm25_ranks_relevant_chunk_first_and_ignores_unrelated():
    chunks = chunk_text("bm25 ranks documents by rare query terms", "k.md") + chunk_text(
        "bananas are yellow fruit grown in warm climates", "f.md"
    )
    index = BM25Index()
    index.build(chunks)
    hits = index.search("how does bm25 rank documents", k=2)
    assert hits[0][0].source == "k.md"
    assert all(chunk.source != "f.md" for chunk, _ in hits)
    assert index.search("quantum chromodynamics") == []


def test_knowledge_base_from_directory_and_roundtrip(tmp_path):
    (tmp_path / "a.md").write_text("Embeddings place similar meanings close together.")
    (tmp_path / "b.txt").write_text("Temperature controls randomness in sampling.")
    (tmp_path / "ignored.csv").write_text("embeddings,embeddings")
    kb = KnowledgeBase.from_directory(tmp_path)
    assert len(kb) == 2
    assert kb.search("embeddings")[0].source == "a.md"

    kb.save(tmp_path / "idx.json")
    again = KnowledgeBase.load(tmp_path / "idx.json")
    assert again.search("temperature sampling")[0].source == "b.txt"


def test_markdown_headings_become_their_own_sentence(tmp_path):
    (tmp_path / "a.md").write_text("# Big Title\n\nBody text starts here and continues.")
    assert KnowledgeBase.from_directory(tmp_path).chunks[0].text.startswith("Big Title. Body text")


def test_missing_directory_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        KnowledgeBase.from_directory(tmp_path / "nope")
