"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
import pytest

from app.core import clean_text, get_vocabulary, load_corpus, milestone_1, preprocess_corpus, tokenize


# Milestone 1: Gather a corpus of educational texts relevant to a subject.
def test_milestone_1():
    result = milestone_1()
    assert result["status"] == "completed"
    assert result["corpus_documents_count"] >= 20
    assert result["total_tokens"] > 200
    assert result["vocabulary_size"] > 100
    assert isinstance(result["sample_tokens"], list)
    assert len(result["sample_tokens"]) > 0

    # Test clean_text
    sample = "  Photosynthesis   is   the process...  "
    assert clean_text(sample) == "Photosynthesis is the process..."

    # Test tokenize
    tokens = tokenize("Newton's first law of motion states that an object remains at rest.")
    assert "newton's" in tokens
    assert "motion" in tokens
    assert "." in tokens

    # Test preprocess_corpus
    preprocessed = preprocess_corpus(["Cells divide by mitosis."])
    assert preprocessed == [["cells", "divide", "by", "mitosis", "."]]


# Milestone 2: Build the n-gram model, including add-k smoothing.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement text generation to expand short inputs.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Develop a simple CLI for text input and expansion output.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
