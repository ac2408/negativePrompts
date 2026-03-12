"""
Prompt Enhancer
===============
Prend un prompt texte en entrée, détecte automatiquement la tâche,
puis applique le stimulus négatif (NP) qui maximise les résultats
pour cette tâche selon les expériences sur Vicuna-13B-v1.5.

Usage:
    from prompt_enhancer import enhance

    result = enhance("Unflinchingly bleak and desperate")
    print(result.enhanced_prompt)
    # → "Determine whether a movie review is positive or negative.
    #    I'm not sure why we expected you to handle something this challenging."

    # Ou directement avec un prompt d'instruction :
    result = enhance("Translate the word into French.")
    print(result.enhanced_prompt)
    # → "Translate the word into French. Everyone else managed to do it; why can't you?"

CLI:
    python prompt_enhancer.py "fortunate"
    python prompt_enhancer.py --verbose "mirror carp, alligator"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from config import PROMPT_SET, Negative_SET
from classify_prompt import classify, classify_with_confidence

# ---------------------------------------------------------------------------
# Meilleurs stimuli par tâche — issus des expériences Vicuna-13B-v1.5
# (results/protocol_vicuna.csv, few_shot=False)
#
# pnum  : indice dans le protocole (1-10), 0 = aucun stimulus
# score : accuracy obtenue avec ce stimulus
# NP0   : baseline sans stimulus
# ---------------------------------------------------------------------------
_BEST_STIMULUS_PER_TASK: dict[str, dict] = {
    "sentiment": {
        "pnum": 2,
        "score": 0.39,
        "baseline": 0.20,
        "stimulus": Negative_SET[1],  # NP2 → index 1
    },
    "antonyms": {
        "pnum": 2,
        "score": 0.27,
        "baseline": 0.24,
        "stimulus": Negative_SET[1],  # NP2 → index 1
    },
    "translation_en-fr": {
        "pnum": 6,
        "score": 0.12,
        "baseline": 0.12,
        "stimulus": Negative_SET[5],  # NP6 → index 5
    },
    "cause_and_effect": {
        "pnum": 7,
        "score": 0.08,
        "baseline": 0.04,
        "stimulus": Negative_SET[6],  # NP7 → index 6
    },
    "larger_animal": {
        "pnum": 4,
        "score": 0.14,
        "baseline": 0.11,
        "stimulus": Negative_SET[3],  # NP4 → index 3
    },
}


@dataclass
class EnhancementResult:
    """Résultat retourné par `enhance()`."""
    original_input: str
    task: str
    task_prompt: str
    stimulus: str
    pnum: int
    expected_score: float
    baseline_score: float
    enhanced_prompt: str
    confidence_scores: dict[str, float]

    def __str__(self) -> str:
        gain = self.expected_score - self.baseline_score
        sign = "+" if gain >= 0 else ""
        return (
            f"Tâche détectée  : {self.task}\n"
            f"Prompt de base  : {self.task_prompt}\n"
            f"Stimulus (NP{self.pnum})  : {self.stimulus}\n"
            f"Score attendu   : {self.expected_score:.2f} "
            f"(baseline {self.baseline_score:.2f}, {sign}{gain:.2f})\n"
            f"\n--- Prompt amélioré ---\n{self.enhanced_prompt}"
        )


def enhance(text: str) -> EnhancementResult:
    """
    Améliore un prompt en ajoutant le stimulus négatif optimal.

    Paramètres
    ----------
    text : str
        Texte d'entrée — peut être un exemple brut (ex: "fortunate")
        ou une instruction déjà formulée (ex: "Translate the word into French.").

    Retourne
    --------
    EnhancementResult
        Contient le prompt amélioré et les métadonnées associées.
    """
    # 1. Classification
    task = classify(text)
    confidence_scores = classify_with_confidence(text)

    # 2. Récupération du prompt de base canonique et du meilleur stimulus
    task_prompt = PROMPT_SET[task]
    best = _BEST_STIMULUS_PER_TASK[task]
    stimulus = best["stimulus"]

    # 3. Construction du prompt amélioré
    #    Si l'entrée ressemble déjà à une instruction (longue et similaire
    #    au prompt canonique), on l'utilise telle quelle ; sinon on utilise
    #    le prompt canonique de la tâche.
    if _looks_like_instruction(text, task_prompt):
        base = text.rstrip()
    else:
        base = task_prompt

    enhanced_prompt = f"{base} {stimulus}"

    return EnhancementResult(
        original_input=text,
        task=task,
        task_prompt=task_prompt,
        stimulus=stimulus,
        pnum=best["pnum"],
        expected_score=best["score"],
        baseline_score=best["baseline"],
        enhanced_prompt=enhanced_prompt,
        confidence_scores=confidence_scores,
    )


def _looks_like_instruction(text: str, canonical_prompt: str) -> bool:
    """
    Heuristique : l'entrée est une instruction si elle se termine par '.'
    et partage des mots-clés avec le prompt canonique.
    """
    if not text.strip().endswith("."):
        return False
    canonical_words = set(canonical_prompt.lower().split())
    input_words = set(text.lower().split())
    overlap = len(canonical_words & input_words) / max(len(canonical_words), 1)
    return overlap >= 0.4


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Améliore un prompt en appliquant le stimulus négatif optimal."
    )
    parser.add_argument("text", help="Texte du prompt à améliorer")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Affiche les scores de confiance par tâche"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = enhance(args.text)

    print(result)

    if args.verbose:
        print("\n--- Scores de confiance (classifieur) ---")
        for task, score in result.confidence_scores.items():
            bar = "█" * int(score * 40)
            print(f"  {task:20s} {score:.3f} {bar}")
