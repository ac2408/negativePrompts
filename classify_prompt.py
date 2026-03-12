"""
Prompt Classifier for the 5 NegativePrompt tasks:
  - sentiment
  - antonyms
  - translation_en-fr
  - cause_and_effect
  - larger_animal

Usage:
    from classify_prompt import classify, classify_with_confidence

    label = classify("This film was absolutely breathtaking.")
    # -> "sentiment"

    scores = classify_with_confidence("fortunate")
    # -> {"antonyms": 0.62, "translation_en-fr": 0.31, ...}
"""

import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

TASKS = ["sentiment", "antonyms", "translation_en-fr", "cause_and_effect", "larger_animal"]

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data/instruction_induction/raw/execute/")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_task_inputs(task: str) -> list[str]:
    """Load raw input strings for a task from the execute split."""
    path = os.path.join(_DATA_PATH, f"{task}.json")
    with open(path, "r") as f:
        data = json.load(f)

    inputs = []
    for example in data["examples"].values():
        if task == "cause_and_effect":
            cause, effect = example["cause"], example["effect"]
            # Both orderings appear at inference time
            inputs.append(f"Sentence 1: {cause} Sentence 2: {effect}")
            inputs.append(f"Sentence 1: {effect} Sentence 2: {cause}")
        else:
            inputs.append(example["input"])
    return inputs


# ---------------------------------------------------------------------------
# Classifier build & training
# ---------------------------------------------------------------------------

def _build_classifier() -> Pipeline:
    """Build and train the TF-IDF + Logistic Regression pipeline."""
    X, y = [], []
    for task in TASKS:
        examples = _load_task_inputs(task)
        X.extend(examples)
        y.extend([task] * len(examples))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=20_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")),
    ])
    pipeline.fit(X, y)
    return pipeline


# Lazy singleton — built once on first call
_classifier: Pipeline | None = None

def _get_classifier() -> Pipeline:
    global _classifier
    if _classifier is None:
        _classifier = _build_classifier()
    return _classifier


# ---------------------------------------------------------------------------
# Heuristics (fast-path for unambiguous inputs)
# ---------------------------------------------------------------------------

def _heuristic(text: str) -> str | None:
    """Return a task label if the input matches a clear rule, else None."""
    # cause_and_effect inputs always contain the "Sentence 1/2:" prefix
    if "Sentence 1:" in text and "Sentence 2:" in text:
        return "cause_and_effect"

    words = text.split()

    # sentiment inputs are long (movie reviews — typically 5+ words, no comma pair)
    if len(words) >= 8:
        return "sentiment"

    # larger_animal inputs are exactly two animal names separated by a comma
    if "," in text and len(words) <= 5:
        return "larger_animal"

    return None  # fall through to ML


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(text: str) -> str:
    """
    Classify a prompt/input text into one of the 5 tasks.

    Parameters
    ----------
    text : str
        The raw input string to classify.

    Returns
    -------
    str
        One of: "sentiment", "antonyms", "translation_en-fr",
                "cause_and_effect", "larger_animal".
    """
    label = _heuristic(text)
    if label is not None:
        return label
    return _get_classifier().predict([text])[0]


def classify_with_confidence(text: str) -> dict[str, float]:
    """
    Return a confidence score for each task.

    The heuristic fast-path is bypassed here so that you always get
    a full probability distribution from the ML model.

    Parameters
    ----------
    text : str
        The raw input string to classify.

    Returns
    -------
    dict[str, float]
        Mapping from task name to probability, sorted by descending score.
    """
    clf = _get_classifier()
    proba = clf.predict_proba([text])[0]
    scores = dict(zip(clf.classes_, proba.tolist()))
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Quick evaluation helper
# ---------------------------------------------------------------------------

def evaluate(verbose: bool = False) -> float:
    """
    Evaluate the classifier on the training data (leave-one-out not applied —
    this is a sanity check, not a held-out benchmark).

    Returns the overall accuracy.
    """
    clf = _get_classifier()
    correct = total = 0

    for task in TASKS:
        inputs = _load_task_inputs(task)
        predictions = [classify(x) for x in inputs]
        task_correct = sum(p == task for p in predictions)
        if verbose:
            print(f"{task:20s}  {task_correct:3d}/{len(inputs):3d}  "
                  f"({100 * task_correct / len(inputs):.1f}%)")
        correct += task_correct
        total += len(inputs)

    acc = correct / total
    if verbose:
        print(f"\nOverall: {correct}/{total} ({100 * acc:.1f}%)")
    return acc


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # sentiment
        "Unflinchingly bleak and desperate",
        "A woman's pic directed with resonance by Ilya Chaiken.",
        # antonyms
        "fortunate",
        "urban",
        # translation_en-fr
        "family",
        "place",
        # cause_and_effect
        "Sentence 1: The child hurt their knee. Sentence 2: The child started crying.",
        "Sentence 1: I washed the car. Sentence 2: My car got dirty.",
        # larger_animal
        "mirror carp, alligator",
        "baboons, hamster",
    ]

    expected = [
        "sentiment", "sentiment",
        "antonyms", "antonyms",
        "translation_en-fr", "translation_en-fr",
        "cause_and_effect", "cause_and_effect",
        "larger_animal", "larger_animal",
    ]

    print("=== Classify Demo ===\n")
    for text, exp in zip(test_cases, expected):
        pred = classify(text)
        status = "✓" if pred == exp else "✗"
        print(f"[{status}] {repr(text[:50])}")
        print(f"    expected={exp}, predicted={pred}")

    print("\n=== Confidence Scores (sample) ===\n")
    for text in ["fortunate", "family", "baboons, hamster"]:
        scores = classify_with_confidence(text)
        print(f"Input: {repr(text)}")
        for task, score in scores.items():
            bar = "█" * int(score * 30)
            print(f"  {task:20s} {score:.3f} {bar}")
        print()

    print("\n=== Evaluation on training data ===\n")
    evaluate(verbose=True)
