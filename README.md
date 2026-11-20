# Educational Content Expander

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/g3rpbb8x2a/badge.svg)](https://ask.thili.ai/b/g3rpbb8x2a)

The **Educational Content Expander** is a CLI tool designed to assist educators and curriculum designers. It takes short phrases or bullet points and expands them into coherent, grammatically plausible, and contextually rich sentences for lesson plans by generating next-word suggestions based on educational texts.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that expands short phrases or bullet points into coherent sentences by generating next-word suggestions based on educational texts.

**Done when.** The tool successfully takes a short input phrase via the CLI and generates grammatically plausible and contextually relevant expansions, demonstrating the functionality of a next-word predictor built from educational content.

## Milestones Completed

- [x] **1. Gather a corpus of educational texts relevant to a subject.**  
      Curated educational texts across biology, physics, chemistry, mathematics, computer science, and earth sciences. Implemented text preprocessing (`clean_text`), normalization, and tokenization (`tokenize`).
- [x] **2. Build the n-gram model, including add-k smoothing.**  
      Implemented `NGramLanguageModel` supporting arbitrary n-gram orders ($n \ge 1$), frequency counting for contexts and n-grams, and Add-k (Laplace / Lidstone) smoothing guaranteeing valid probability distributions ($\sum P = 1.0$).
- [x] **3. Implement text generation to expand short inputs.**  
      Developed iterative next-word generation (`expand_text`) with context slicing, repetition avoidance, punctuation formatting, and top candidate next-word ranking (`suggest_next_words`).
- [x] **4. Develop a simple CLI for text input and expansion output.**  
      Built a full CLI in `app.cli` with options for single prompt expansion, candidate suggestion ranking (`--suggest`), batch input file expansion (`--input-file`), file export (`--output-file`), and interactive mode.

## Stretch Features Implemented

- **Multiple Input Expansion (Batch Mode)**: Use `--input-file` to expand multiple bullet points line-by-line.
- **Export to File**: Save generated lesson plans directly to disk using `--output-file`.
- **Top Next-Word Candidate Suggestions**: Use `--suggest` to display the top candidate next words along with their probability confidence scores.

## Architecture & Smoothing

The next-word predictor uses an $n$-gram language model with **Add-k smoothing**:

$$P(w_i \mid w_{i-n+1}, \dots, w_{i-1}) = \frac{C(w_{i-n+1}, \dots, w_i) + k}{C(w_{i-n+1}, \dots, w_{i-1}) + k \cdot |V|}$$

- When $k = 1.0$, this corresponds to **Laplace smoothing**.
- When $0 < k < 1.0$, this corresponds to **Lidstone smoothing**.
- Handles unseen n-grams gracefully by assigning a smoothed non-zero probability rather than zero.

## Quickstart & Usage

### 1. Requirements & Setup

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Expander via CLI

```bash
# Expand a single prompt
python -m app.cli expand "Photosynthesis is"

# Expand with candidate next-word suggestions and confidence scores
python -m app.cli expand "Cells divide by" --suggest

# Adjust max generated words and n-gram order
python -m app.cli expand "An algorithm is" --max-words 20 --order 3

# Batch process bullet points from a file and save results
python -m app.cli expand --input-file input_bullets.txt --output-file expanded_lesson.txt

# Smoke check
python -m app.cli hello
```

### 3. Run the Test Suite

```bash
pytest -v
```

All 7 core milestone and smoke tests pass cleanly.

## CI/CD Pipeline

Both GitLab CI (`.gitlab-ci.yml`) and GitHub Actions (`.github/workflows/ci.yml`) are configured to automatically install dependencies and run the pytest test suite on each commit.
