"""
Reproduction minimale — NegativePrompt (IJCAI 2024)
Modele : Vicuna-13B (lmsys/vicuna-13b-v1.5)
Taches : 5 taches representatives (zero-shot)
Prompts : pnum=0 (original) et pnum=1 a 10 (NP01-NP10)

Usage depuis Kaggle :
    # P100 / T4 (16 GB) — quantification 4-bit obligatoire
    %run /kaggle/working/negativePrompts/run_vicuna_5tasks.py --quantize

    # T4x2 (32 GB) — fp16 natif
    %run /kaggle/working/negativePrompts/run_vicuna_5tasks.py

Sorties :
    results/neg/vicuna/{task}.txt        — format existant du papier
    results/protocol_vicuna.csv          — protocole structure (modele, prompt,
                                           params, metrique, score)
    results/summary_vicuna.txt           — tableau de synthese

Protocole enregistre pour chaque experience :
    model | task | pnum | original_prompt | negative_stimulus | few_shot
    temperature | do_sample | max_new_tokens | batch_size | quantized | metric | score
"""

import os
import sys
import csv
import shutil
import argparse

REPO = "/kaggle/working/negativePrompts"
os.chdir(REPO)
sys.path.insert(0, REPO)

# ─── Arguments ───────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--quantize", action="store_true",
    help="Charger Vicuna en 4-bit (bitsandbytes) pour GPU 16 GB (P100, T4)"
)
_parser.add_argument(
    "--batch_size", type=int, default=None,
    help="Taille de batch (defaut: 8 avec --quantize, 4 sans)"
)
_parser.add_argument(
    "--few_shot", action="store_true",
    help="Activer le mode few-shot (5 demonstrations par tache)"
)
_args, _ = _parser.parse_known_args()
QUANTIZE   = _args.quantize
FEW_SHOT   = _args.few_shot
BATCH_SIZE = _args.batch_size if _args.batch_size else (8 if QUANTIZE else 4)

# ─── 5 taches selectionnees (diversite de types) ───────────────────────────
SELECTED_TASKS = [
    "sentiment",         # classification binaire (positive / negative)
    "antonyms",          # generation lexicale (oppose d un mot)
    "translation_en-fr", # traduction anglais -> francais
    "cause_and_effect",  # raisonnement causal
    "larger_animal",     # connaissance factuelle
]

# ─── Parametres d inference ─────────────────────────────────────────────────
INFER_PARAMS = {
    "model_id":       "lmsys/vicuna-13b-v1.5",
    "do_sample":      False,     # greedy : reproductible et plus rapide
    "temperature":    None,      # non utilise si do_sample=False
    "max_new_tokens": 30,
    "batch_size":     BATCH_SIZE,
    "few_shot":       FEW_SHOT,
    "quantized":      QUANTIZE,
}

# max_new_tokens specifique par tache
TASK_MAX_TOKENS = {
    "cause_and_effect": 80,   # format "Sentence X: ..." necessite plus de tokens
}

# Metrique utilisee par tache (depuis utility.py)
TASK_METRIC = {
    "sentiment":         "em (sentiment)",
    "antonyms":          "em (contain)",
    "translation_en-fr": "em (contain)",
    "cause_and_effect":  "em (cause_effect)",
    "larger_animal":     "em (larger_animal)",
}


# ─── Chargement Vicuna ──────────────────────────────────────────────────────
def load_vicuna():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = INFER_PARAMS["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"

    if INFER_PARAMS["quantized"]:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Chargement de {model_id} (4-bit NF4, P100/T4 16 GB)...")
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb_cfg
        )
    else:
        print(f"Chargement de {model_id} (float16, T4x2 32 GB)...")
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )

    mdl.eval()
    print(f"Vicuna charge. (quantize={INFER_PARAMS['quantized']}, batch_size={INFER_PARAMS['batch_size']})")
    return mdl, tok


# ─── Inference avec batching ────────────────────────────────────────────────
def make_vicuna_infer(model, tokenizer):
    import torch
    import re
    batch_size       = INFER_PARAMS["batch_size"]
    max_tokens_default = INFER_PARAMS["max_new_tokens"]

    def infer(queries, task, **kw):
        outputs    = []
        max_tokens = TASK_MAX_TOKENS.get(task, max_tokens_default)

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            vicuna_batch = [
                "A chat between a curious user and an artificial intelligence "
                "assistant. The assistant gives helpful, detailed, and polite "
                "answers to the user's questions.\n"
                f"USER: {q}\nASSISTANT:"
                for q in batch
            ]
            enc = tokenizer(
                vicuna_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_len = enc["input_ids"].shape[1]
            ids  = enc["input_ids"].to("cuda:0")
            mask = enc["attention_mask"].to("cuda:0")

            with torch.no_grad():
                gen = model.generate(
                    ids,
                    attention_mask=mask,
                    max_new_tokens=max_tokens,
                    do_sample=INFER_PARAMS["do_sample"],
                    pad_token_id=tokenizer.eos_token_id,
                )

            for out in gen:
                text = tokenizer.decode(
                    out[input_len:], skip_special_tokens=True
                ).strip()

                if task == "cause_and_effect":
                    # Cherche "Sentence 1" ou "Sentence 2" dans la reponse
                    m = re.search(r'[Ss]entence\s+([12])', text)
                    if m:
                        text = f"Sentence {m.group(1)}"
                    else:
                        # Cherche "first" ou "second" comme fallback
                        tl = text.lower()
                        if "first" in tl:
                            text = "Sentence 1"
                        elif "second" in tl:
                            text = "Sentence 2"
                else:
                    # Pour les autres taches : tronquer au premier point
                    idx = text.find(".")
                    if idx > 0:
                        text = text[:idx]

                text = text.strip()
                outputs.append(text)

        return outputs

    return infer


# ─── Sauvegarde du protocole ─────────────────────────────────────────────────
def init_protocol_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "task", "pnum",
            "original_prompt", "negative_stimulus",
            "few_shot", "do_sample", "temperature",
            "max_new_tokens", "batch_size", "quantized",
            "metric", "score",
        ])
    return path


def append_protocol(path, task, pnum, original_prompt, negative_stimulus, score):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "vicuna-13b-v1.5",
            task,
            pnum,
            original_prompt,
            negative_stimulus,
            INFER_PARAMS["few_shot"],
            INFER_PARAMS["do_sample"],
            INFER_PARAMS["temperature"],
            INFER_PARAMS["max_new_tokens"],
            INFER_PARAMS["batch_size"],
            INFER_PARAMS["quantized"],
            TASK_METRIC[task],
            f"{score:.4f}",
        ])


# ─── Tableau de synthese ─────────────────────────────────────────────────────
def write_summary(results, summary_path):
    """
    results : dict  task -> list of (pnum, score)  (pnum 0 a 10)
    """
    lines = []
    lines.append("=" * 70)
    mode = "few-shot" if INFER_PARAMS["few_shot"] else "zero-shot"
    lines.append(f"SYNTHESE — Vicuna-13B | 5 taches | pnum 0-10 | {mode}")
    lines.append("=" * 70)

    header = f"{'Tache':<22} {'Baseline':>8} " + " ".join(
        f"NP{p:02d}{'*' if _is_best(results, task, p) else ' ':>1}" for task in [None] for p in range(1, 11)
    )
    # en-tete par tache
    lines.append(f"\n{'Tache':<22} {'NP00':>7} " +
                 "  ".join(f"NP{p:02d}" for p in range(1, 11)))
    lines.append("-" * 70)

    for task, scores_list in results.items():
        scores_list_sorted = sorted(scores_list, key=lambda x: x[0])
        score_vals = [s for _, s in scores_list_sorted]
        baseline = score_vals[0]
        row = f"{task:<22} {baseline:>7.4f} "
        for i, s in enumerate(score_vals[1:], 1):
            delta = s - baseline
            marker = "+" if delta > 0.01 else ("-" if delta < -0.01 else " ")
            row += f" {s:.4f}{marker}"
        lines.append(row)

    lines.append("\n(+) amelioration > 1pp vs baseline  (-) degradation > 1pp")
    lines.append("=" * 70)

    content = "\n".join(lines)
    print(content)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)


def _is_best(results, task, pnum):
    if task is None:
        return False
    scores = dict(results.get(task, []))
    if not scores:
        return False
    best_pnum = max(scores, key=lambda k: scores[k])
    return best_pnum == pnum


# ─── Point d entree ──────────────────────────────────────────────────────────
def main():
    from config import PROMPT_SET, Negative_SET

    # Chargement du modele
    model, tokenizer = load_vicuna()
    infer_fn = make_vicuna_infer(model, tokenizer)

    # Monkeypatch
    import exec_accuracy
    exec_accuracy.get_response_from_llm = \
        lambda llm_model, queries, task, few_shot, **kw: infer_fn(queries, task)

    from main import run as main_run

    # Preparation des dossiers
    shutil.rmtree("results/neg/vicuna", ignore_errors=True)
    os.makedirs("results/neg/vicuna", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    suffix       = "_fewshot" if INFER_PARAMS["few_shot"] else ""
    csv_path     = f"results/protocol_vicuna{suffix}.csv"
    summary_path = f"results/summary_vicuna{suffix}.txt"
    init_protocol_csv(csv_path)

    all_results = {task: [] for task in SELECTED_TASKS}
    total = len(SELECTED_TASKS) * 11
    done  = 0

    for task in SELECTED_TASKS:
        original_prompt = PROMPT_SET[task]
        for pnum in range(11):
            done += 1
            neg_stimulus = Negative_SET[pnum - 1] if pnum > 0 else "(aucun)"
            print(
                f"\n[{done}/{total}] vicuna | {task} | pnum={pnum}"
                f"\n  Prompt original : {original_prompt}"
                f"\n  Stimulus negatif: {neg_stimulus}",
                flush=True,
            )

            try:
                main_run(
                    task=task,
                    model="vicuna",
                    pnum=pnum,
                    few_shot=INFER_PARAMS["few_shot"],
                )

                # Lire le score depuis le fichier de resultats
                result_file = f"results/neg/vicuna/{task}.txt"
                score = _read_last_score(result_file)
                print(f"  Score : {score:.4f}")

                all_results[task].append((pnum, score))
                append_protocol(
                    csv_path, task, pnum,
                    original_prompt, neg_stimulus, score
                )

            except Exception as e:
                print(f"  ERREUR : {e}")
                all_results[task].append((pnum, 0.0))
                append_protocol(
                    csv_path, task, pnum,
                    original_prompt, neg_stimulus, 0.0
                )

    # Tableau de synthese final
    write_summary(all_results, summary_path)
    print(f"\nProtocole CSV : {csv_path}")
    print(f"Synthese      : {summary_path}")
    print("\nVicuna 5 taches termine !")


def _read_last_score(filepath):
    """Lit le dernier score enregistre dans un fichier results/neg/."""
    if not os.path.exists(filepath):
        return 0.0
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if line.startswith("Test score:"):
            try:
                return float(line.split(":")[1].strip())
            except ValueError:
                pass
    return 0.0


if __name__ == "__main__":
    main()
