"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass."""
import pytest

from app.core import (
    NGramLanguageModel,
    build_educational_model,
    clean_text,
    expand_text,
    format_tokens_to_sentence,
    get_vocabulary,
    load_corpus,
    milestone_1,
    milestone_2,
    milestone_3,
    preprocess_corpus,
    suggest_next_words,
    tokenize,
)


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
def test_milestone_2():
    result = milestone_2()
    assert result["status"] == "completed"
    assert result["n"] == 3
    assert result["k"] == 1.0
    assert result["vocab_size"] > 100
    assert len(result["top_predictions"]) > 0
    # Probabilities must sum to 1.0 over the vocabulary
    assert abs(result["probability_sum"] - 1.0) < 1e-5

    # Test NGramLanguageModel unit mechanics
    mini_corpus = [
        ["the", "cell", "is", "the", "basic", "unit", "."],
        ["the", "cell", "divides", "by", "mitosis", "."],
    ]
    model = NGramLanguageModel(n=2, k=1.0)
    model.fit(mini_corpus)

    # Check context counts
    assert model.context_counts[("cell",)] == 2
    assert model.ngram_counts[("cell",)]["is"] == 1
    assert model.ngram_counts[("cell",)]["divides"] == 1

    # Check Add-k smoothing
    v_size = model.vocab_size
    prob_is = model.probability("is", ("cell",), k=1.0)
    assert abs(prob_is - (2.0 / (2.0 + v_size))) < 1e-5

    # Check that unseen words in context still get positive probability due to smoothing
    prob_unseen = model.probability("mitosis", ("basic",), k=1.0)
    assert prob_unseen > 0.0

    # Probability distribution sums to 1.0
    probs = model.get_next_word_probabilities(("cell",))
    assert abs(sum(probs.values()) - 1.0) < 1e-5

    # Check k=0.5 (Lidstone smoothing)
    prob_lidstone = model.probability("is", ("cell",), k=0.5)
    assert abs(prob_lidstone - (1.5 / (2.0 + 0.5 * v_size))) < 1e-5


# Milestone 3: Implement text generation to expand short inputs.
def test_milestone_3():
    result = milestone_3()
    assert result["status"] == "completed"
    assert result["model_order"] == 3
    assert "Photosynthesis is" in result["expansions"]
    assert "Cells divide by" in result["expansions"]

    photo_expansion = result["expansions"]["Photosynthesis is"]["expansion"]
    assert len(photo_expansion) > len("Photosynthesis is")
    assert "photosynthesis is" in photo_expansion.lower()

    # Test expand_text function
    model = build_educational_model(n=3, k=1.0)
    text = expand_text(model, "cells divide by", max_new_words=10)
    assert "cells divide by" in text.lower()
    assert text.endswith(".")

    # Test formatting
    formatted = format_tokens_to_sentence(["photosynthesis", "is", "a", "process", "."])
    assert formatted == "Photosynthesis is a process."

    # Test next word suggestions
    suggestions = suggest_next_words(model, "photosynthesis is", top_k=3)
    assert len(suggestions) == 3
    assert all(isinstance(s[0], str) and isinstance(s[1], float) for s in suggestions)


# Milestone 4: Develop a simple CLI for text input and expansion output.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
