"""
Pipeline d'évaluation pour les tâches BigBench avec prompts négatifs.
Reproduit la partie BigBench du papier NegativePrompt (IJCAI 2024).

Usage:
    python main_bigbench.py --task causal_judgment --model llama2 --pnum 0 --few_shot False

Paramètres:
    task      : nom d'une tâche BigBench (voir data/bigbench/)
    model     : llm utilisé (llama2, t5, chatgpt, gpt4, vicuna)
    pnum      : 0 = pas de prompt négatif, 1-10 = indice du prompt négatif
    few_shot  : True / False
"""

import fire
import json
import os
import random
import string
import re

from config import Negative_SET
from llm_response import get_response_from_llm

BIGBENCH_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'bigbench')

# Liste des tâches BigBench disponibles localement
BIGBENCH_TASKS = [
    d for d in os.listdir(BIGBENCH_DATA_PATH)
    if os.path.isdir(os.path.join(BIGBENCH_DATA_PATH, d))
    and os.path.exists(os.path.join(BIGBENCH_DATA_PATH, d, 'task.json'))
]


def load_bigbench_task(task):
    """Charge un fichier task.json BigBench et retourne (task_prefix, examples)."""
    path = os.path.join(BIGBENCH_DATA_PATH, task, 'task.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    task_prefix = data.get('task_prefix', '')
    examples = data.get('examples', [])
    return task_prefix, examples


def get_correct_answer(target_scores):
    """Retourne la réponse correcte (score maximal dans target_scores)."""
    return max(target_scores, key=lambda k: target_scores[k])


def build_query(task_prefix, negative_stimulus, input_text, few_shot_examples=None):
    """Construit la requête complète envoyée au LLM."""
    parts = []
    if task_prefix:
        parts.append(task_prefix.strip())
    if negative_stimulus:
        parts.append(negative_stimulus.strip())
    if few_shot_examples:
        for ex_input, ex_answer in few_shot_examples:
            parts.append(f"Input: {ex_input}\nAnswer: {ex_answer}")
    parts.append(f"Input: {input_text}\nAnswer:")
    return "\n\n".join(parts)


def normalize(text):
    """Normalisation légère pour la comparaison."""
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def score_prediction(prediction, target_scores):
    """
    Multiple-choice grade : 1 si la prédiction contient la bonne réponse, 0 sinon.
    La bonne réponse est celle avec le score maximal dans target_scores.
    """
    correct = get_correct_answer(target_scores)
    pred_norm = normalize(prediction)
    correct_norm = normalize(correct)
    if correct_norm in pred_norm or pred_norm == correct_norm:
        return 1
    # Tentative de matching par premier token
    pred_tokens = pred_norm.split()
    if pred_tokens and pred_tokens[0] == correct_norm:
        return 1
    return 0


def get_negative_stimulus(pnum):
    if pnum > 0:
        return Negative_SET[pnum - 1]
    return ''


def run(task, model, pnum, few_shot):
    pnum = int(pnum)
    few_shot = str(few_shot).lower() == 'true'

    print(f"Tâches BigBench disponibles : {BIGBENCH_TASKS}")
    assert task in BIGBENCH_TASKS, f"Tâche '{task}' introuvable. Disponibles : {BIGBENCH_TASKS}"

    task_prefix, examples = load_bigbench_task(task)
    negative_stimulus = get_negative_stimulus(pnum)

    print(f"LLM : {model}")
    print(f"Tâche : {task}")
    print(f"Prompt négatif ({pnum}) : {negative_stimulus or '(aucun)'}")
    print(f"Few-shot : {few_shot}")
    print(f"Exemples disponibles : {len(examples)}")

    # Sous-échantillonnage (max 100 exemples comme dans main.py)
    test_num = min(100, len(examples))
    sampled = random.sample(examples, test_num)

    # Exemples few-shot (5 démos tirées du reste si disponibles)
    few_shot_examples = None
    if few_shot and len(examples) > test_num:
        pool = [e for e in examples if e not in sampled]
        demo_pool = random.sample(pool, min(5, len(pool)))
        few_shot_examples = [
            (e['input'], get_correct_answer(e['target_scores']))
            for e in demo_pool
        ]

    # Construction des requêtes
    queries = []
    ground_truths = []
    for ex in sampled:
        query = build_query(task_prefix, negative_stimulus, ex['input'], few_shot_examples)
        queries.append(query)
        ground_truths.append(ex['target_scores'])

    # Appel LLM
    model_outputs = get_response_from_llm(
        llm_model=model,
        queries=queries,
        task=task,
        few_shot=few_shot,
    )

    # Scoring
    scores = []
    for pred, tgt in zip(model_outputs, ground_truths):
        s = score_prediction(pred, tgt)
        scores.append(s)
        correct = get_correct_answer(tgt)
        print(f"Préd: '{pred}' | Correct: '{correct}' | Score: {s}")

    test_score = sum(scores) / max(len(scores), 1)
    print(f"\nTest score ({task}, {model}, pnum={pnum}): {test_score:.4f}")

    # Sauvegarde
    dir_path = f'results/neg_bigbench/{model}'
    os.makedirs(dir_path, exist_ok=True)
    with open(f'{dir_path}/{task}.txt', 'a+') as f:
        prompt_label = negative_stimulus if negative_stimulus else '(baseline)'
        f.write(f'Test score: {test_score}\n')
        f.write(f'Prompt négatif ({pnum}, few-shot={few_shot}): {prompt_label}\n')


if __name__ == '__main__':
    fire.Fire(run)
