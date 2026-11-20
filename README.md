# Educational Content Expander

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/g3rpbb8x2a/badge.svg)](https://ask.thili.ai/b/g3rpbb8x2a)

The Educational Content Expander is a CLI tool designed to assist educators. It takes short phrases or bullet points and expands them into more complete sentences, leveraging a next-word prediction model trained on educational materials. This helps streamline the creation of detailed lesson plans and teaching content.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that expands short phrases or bullet points into coherent sentences by generating next-word suggestions based on educational texts.

**Done when.** The tool successfully takes a short input phrase via the CLI and generates grammatically plausible and contextually relevant expansions, demonstrating the functionality of a next-word predictor built from educational content.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Gather a corpus of educational texts relevant to a subject.**  
      Hint: Focus on text preprocessing and tokenization for your corpus.
- [ ] **2. Build the n-gram model, including add-k smoothing.**  
      Hint: Implement frequency counting for n-grams and apply add-k smoothing to probabilities.
- [ ] **3. Implement text generation to expand short inputs.**  
      Hint: Use your n-gram model to iteratively predict and append the most probable next words.
- [ ] **4. Develop a simple CLI for text input and expansion output.**  
      Hint: Utilize Python's `argparse` module or basic `input()` and `print()` functions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify multiple input phrases at once and generate expansions for each.
- Implement an option to save the expanded content directly to a text file instead of just printing to the console.
- Enable the tool to suggest multiple possible next words at each step, letting the user choose the best fit.
