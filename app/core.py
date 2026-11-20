"""Core logic for Educational Content Expander.

Milestone 1: Educational text corpus gathering, preprocessing, and tokenization.
Milestone 2: N-gram language model with add-k smoothing.
Milestone 3: Iterative text generation and expansion.
Milestone 4: CLI integration and educational content workflow.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from app.corpus import EDUCATIONAL_TEXTS

START_TOKEN = "<s>"
END_TOKEN = "</s>"


# ============================================================================
# Milestone 1: Educational Corpus, Preprocessing, and Tokenization
# ============================================================================

def clean_text(text: str) -> str:
    """Normalize whitespace and remove non-standard formatting while preserving punctuation."""
    if not text:
        return ""
    # Normalize quotes and dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " ").replace("–", " ")
    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str, lowercase: bool = True, preserve_punctuation: bool = True) -> list[str]:
    """Tokenize educational text into words and sentence-ending punctuation tokens."""
    cleaned = clean_text(text)
    if lowercase:
        cleaned = cleaned.lower()

    if preserve_punctuation:
        # Match words (including contractions like "newton's") and terminal punctuation
        pattern = r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b|[.!?]"
    else:
        pattern = r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b"

    tokens = re.findall(pattern, cleaned)
    return tokens


def load_corpus(custom_texts: list[str] | None = None) -> list[str]:
    """Load the educational corpus."""
    if custom_texts is not None:
        return [clean_text(t) for t in custom_texts if clean_text(t)]
    return [clean_text(t) for t in EDUCATIONAL_TEXTS]


def preprocess_corpus(texts: list[str] | None = None, lowercase: bool = True) -> list[list[str]]:
    """Clean and tokenize the entire educational corpus into sentence token lists."""
    raw_texts = load_corpus(texts)
    tokenized_corpus = []
    for doc in raw_texts:
        tokens = tokenize(doc, lowercase=lowercase, preserve_punctuation=True)
        if tokens:
            tokenized_corpus.append(tokens)
    return tokenized_corpus


def get_vocabulary(tokenized_sentences: list[list[str]]) -> list[str]:
    """Extract a sorted, unique vocabulary list from tokenized sentences."""
    vocab_set = {token for sent in tokenized_sentences for token in sent}
    return sorted(vocab_set)


def milestone_1() -> dict[str, Any]:
    """Milestone 1: Gather a corpus of educational texts relevant to a subject.

    Preprocesses and tokenizes the educational corpus.
    """
    corpus = load_corpus()
    tokenized = preprocess_corpus(corpus)
    vocab = get_vocabulary(tokenized)
    total_tokens = sum(len(sentence) for sentence in tokenized)

    return {
        "status": "completed",
        "corpus_documents_count": len(corpus),
        "total_tokens": total_tokens,
        "vocabulary_size": len(vocab),
        "vocabulary": vocab,
        "sample_tokens": tokenized[0] if tokenized else [],
        "tokenized_corpus": tokenized,
    }


# ============================================================================
# Milestone 2: N-Gram Model with Add-k Smoothing
# ============================================================================

class NGramLanguageModel:
    """An n-gram language model supporting frequency counting and add-k smoothing."""

    def __init__(self, n: int = 3, k: float = 1.0):
        if n < 1:
            raise ValueError("n-gram order n must be at least 1")
        if k < 0:
            raise ValueError("smoothing parameter k must be non-negative")
        self.n = n
        self.k = k
        # ngram_counts[context_tuple][next_word] = count
        self.ngram_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        # context_counts[context_tuple] = total occurrences of context
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.vocab: set[str] = set()
        self.vocab_list: list[str] = []

    def _pad_tokens(self, tokens: list[str]) -> list[str]:
        """Pad a token list with (n-1) START_TOKENs and one END_TOKEN."""
        padding = [START_TOKEN] * (self.n - 1)
        return padding + tokens + [END_TOKEN]

    def _normalize_context(self, context: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Ensure context is a tuple of length (n-1), padding or truncating as needed."""
        ctx_list = list(context)
        needed = self.n - 1
        if needed == 0:
            return ()
        if len(ctx_list) < needed:
            ctx_list = [START_TOKEN] * (needed - len(ctx_list)) + ctx_list
        else:
            ctx_list = ctx_list[-needed:]
        return tuple(ctx_list)

    def fit(self, tokenized_sentences: list[list[str]]) -> NGramLanguageModel:
        """Count n-gram and context frequencies across the tokenized corpus."""
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocab.clear()

        for sentence in tokenized_sentences:
            padded = self._pad_tokens(sentence)
            for token in padded:
                if token != START_TOKEN:
                    self.vocab.add(token)

            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i : i + self.n - 1])
                word = padded[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1

        self.vocab_list = sorted(self.vocab)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def probability(
        self,
        word: str,
        context: tuple[str, ...] | list[str],
        k: float | None = None,
    ) -> float:
        """Calculate conditional probability P(word | context) with add-k smoothing."""
        if k is None:
            k = self.k
        ctx = self._normalize_context(context)
        v_size = self.vocab_size

        count_ctx_word = self.ngram_counts[ctx][word]
        count_ctx = self.context_counts[ctx]

        if k == 0:
            if count_ctx == 0:
                return 1.0 / v_size if v_size > 0 else 0.0
            return count_ctx_word / count_ctx

        # Add-k smoothing formula: (count(ctx, word) + k) / (count(ctx) + k * |V|)
        numerator = count_ctx_word + k
        denominator = count_ctx + (k * v_size)
        return numerator / denominator if denominator > 0 else 0.0

    def get_next_word_probabilities(
        self,
        context: tuple[str, ...] | list[str],
        k: float | None = None,
    ) -> dict[str, float]:
        """Compute the full probability distribution over the vocabulary given context."""
        ctx = self._normalize_context(context)
        probs = {}
        for w in self.vocab_list:
            probs[w] = self.probability(w, ctx, k=k)
        return probs

    def predict_next_words(
        self,
        context: tuple[str, ...] | list[str],
        top_k: int = 5,
        k: float | None = None,
        filter_end_token: bool = False,
    ) -> list[tuple[str, float]]:
        """Return the top_k most probable next words given context."""
        probs = self.get_next_word_probabilities(context, k=k)
        if filter_end_token and END_TOKEN in probs:
            del probs[END_TOKEN]
        sorted_candidates = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        return sorted_candidates[:top_k]


def build_educational_model(n: int = 3, k: float = 1.0) -> NGramLanguageModel:
    """Helper to train an n-gram model on the educational corpus."""
    tokenized_corpus = preprocess_corpus()
    model = NGramLanguageModel(n=n, k=k)
    model.fit(tokenized_corpus)
    return model


def milestone_2() -> dict[str, Any]:
    """Milestone 2: Build the n-gram model, including add-k smoothing.

    Implements frequency counting for n-grams and applies add-k smoothing to probabilities.
    """
    model = build_educational_model(n=3, k=1.0)
    sample_context = ("photosynthesis", "is")
    top_candidates = model.predict_next_words(sample_context, top_k=3)
    vocab_size = model.vocab_size

    # Verify probability distribution sums to approximately 1.0
    probs = model.get_next_word_probabilities(sample_context)
    prob_sum = sum(probs.values())

    return {
        "status": "completed",
        "model": model,
        "n": model.n,
        "k": model.k,
        "vocab_size": vocab_size,
        "sample_context": sample_context,
        "top_predictions": top_candidates,
        "probability_sum": prob_sum,
    }


# ============================================================================
# Milestone 3: Text Generation to Expand Short Inputs
# ============================================================================

def format_tokens_to_sentence(tokens: list[str]) -> str:
    """Reconstruct a well-formatted sentence from a list of tokens."""
    if not tokens:
        return ""

    out_parts: list[str] = []
    punctuation_marks = {".", "!", "?", ",", ";", ":"}

    for i, token in enumerate(tokens):
        if token in {START_TOKEN, END_TOKEN}:
            continue
        if i == 0 or not out_parts:
            out_parts.append(token.capitalize())
        elif token in punctuation_marks:
            out_parts[-1] += token
        else:
            out_parts.append(token)

    sentence = " ".join(out_parts)
    # Ensure terminal punctuation
    if sentence and sentence[-1] not in punctuation_marks:
        sentence += "."
    return sentence


def expand_text(
    model: NGramLanguageModel,
    prompt: str,
    max_new_words: int = 15,
    stop_at_punct: bool = True,
    k_smoothing: float | None = None,
    avoid_repetition: bool = True,
) -> str:
    """Expand a short prompt or bullet point into a complete sentence using the n-gram model."""
    prompt_tokens = tokenize(prompt, lowercase=True, preserve_punctuation=False)
    if not prompt_tokens:
        return ""

    generated = list(prompt_tokens)

    for _ in range(max_new_words):
        context = tuple(generated[-(model.n - 1) :])
        candidates = model.predict_next_words(context, top_k=15, k=k_smoothing)

        next_word = None
        for word, prob in candidates:
            if word == START_TOKEN:
                continue
            if avoid_repetition and len(generated) >= 1 and word == generated[-1]:
                continue
            if avoid_repetition and len(generated) >= 3 and word in generated[-3:]:
                # skip immediate loop
                continue
            next_word = word
            break

        if next_word is None or next_word == END_TOKEN:
            break

        generated.append(next_word)

        if stop_at_punct and next_word in {".", "!", "?"}:
            break

    return format_tokens_to_sentence(generated)


def suggest_next_words(
    model: NGramLanguageModel,
    prompt: str,
    top_k: int = 5,
    k_smoothing: float | None = None,
) -> list[tuple[str, float]]:
    """Suggest top next-word candidates with probabilities for an educator."""
    tokens = tokenize(prompt, lowercase=True, preserve_punctuation=False)
    context = tuple(tokens[-(model.n - 1) :]) if tokens else ()
    candidates = model.predict_next_words(
        context, top_k=top_k, k=k_smoothing, filter_end_token=True
    )
    return candidates


def milestone_3() -> dict[str, Any]:
    """Milestone 3: Implement text generation to expand short inputs.

    Uses the n-gram model to iteratively predict and append the most probable next words.
    """
    model = build_educational_model(n=3, k=1.0)
    test_prompts = [
        "Photosynthesis is",
        "Cells divide by",
        "Newton's first law of motion",
        "An algorithm is",
    ]

    expansions = {}
    for prompt in test_prompts:
        expanded = expand_text(model, prompt, max_new_words=12)
        suggestions = suggest_next_words(model, prompt, top_k=3)
        expansions[prompt] = {
            "expansion": expanded,
            "next_word_suggestions": suggestions,
        }

    return {
        "status": "completed",
        "model_order": model.n,
        "expansions": expansions,
    }


# ============================================================================
# Milestone 4: Develop a Simple CLI for Text Input and Expansion Output
# ============================================================================

def milestone_4(prompt: str = "Photosynthesis is", max_words: int = 15) -> dict[str, Any]:
    """Milestone 4: Develop a simple CLI for text input and expansion output.

    Utilizes Python's argparse module or basic input() and print() functions.
    """
    from app.cli import build_parser

    parser = build_parser()
    model = build_educational_model(n=3, k=1.0)
    expanded = expand_text(model, prompt, max_new_words=max_words)
    suggestions = suggest_next_words(model, prompt, top_k=3)

    return {
        "status": "completed",
        "cli_parser": parser,
        "prompt": prompt,
        "expanded_sentence": expanded,
        "suggestions": suggestions,
        "cli_usage_example": f"python -m app.cli expand '{prompt}'",
    }
