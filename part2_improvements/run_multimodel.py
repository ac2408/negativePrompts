"""
Axe 4 — Extension multi-modèles : Mistral-7B-Instruct-v0.2 vs Vicuna-13B
==========================================================================
Applique la même méthodologie NegativePrompt (5 tâches × 11 pnums = 55 runs)
à mistralai/Mistral-7B-Instruct-v0.2 et génère un tableau comparatif avec
les résultats Vicuna-13B obtenus en Partie 1.

Usage depuis Kaggle :
  # Restart & Clear Output AVANT de lancer pour libérer la GPU RAM
  # T4 16GB (quantize obligatoire)
  %run /kaggle/working/negativePrompts/part2_improvements/run_multimodel.py --quantize --batch_size 4

  # T4x2 32GB (fp16 natif)
  %run /kaggle/working/negativePrompts/part2_improvements/run_multimodel.py

Sorties :
  results/protocol_mistral.csv          — même format que protocol_vicuna.csv
  results/comparison_vicuna_mistral.txt — tableau comparatif baseline/best/delta
"""

import os
import sys
import csv
import shutil
import argparse

REPO = ("/kaggle/working/negativePrompts" if os.path.exists("/kaggle/working") else
        "/content/negativePrompts"         if os.path.exists("/content/negativePrompts") else
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

# ─── Arguments ────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--quantize",   action="store_true",
                     help="4-bit NF4 (obligatoire sur T4 16GB)")
_parser.add_argument("--batch_size", type=int, default=None,
                     help="Taille de batch (défaut: 4 avec --quantize, 4 sans)")
_args, _ = _parser.parse_known_args()
QUANTIZE   = _args.quantize
BATCH_SIZE = _args.batch_size if _args.batch_size else 4

# ─── Constantes ───────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "mistral-7b-instruct-v0.2"

SELECTED_TASKS = [
    "sentiment",
    "antonyms",
    "translation_en-fr",
    "cause_and_effect",
    "larger_animal",
]

INFER_PARAMS = {
    "model_id":       MODEL_ID,
    "do_sample":      False,
    "temperature":    None,
    "max_new_tokens": 30,
    "batch_size":     BATCH_SIZE,
    "few_shot":       False,
    "quantized":      QUANTIZE,
}

TASK_MAX_TOKENS = {"cause_and_effect": 80}

TASK_METRIC = {
    "sentiment":         "em (sentiment)",
    "antonyms":          "em (contain)",
    "translation_en-fr": "em (contain)",
    "cause_and_effect":  "em (cause_effect)",
    "larger_animal":     "em (larger_animal)",
}


# ─── Nettoyage GPU ────────────────────────────────────────────────────────────
def free_gpu():
    """Libère toute la mémoire GPU avant de charger le modèle."""
    import gc, torch

    for var in ["model", "tokenizer", "mdl", "tok", "infer_fn"]:
        if var in globals():
            del globals()[var]
        try:
            import __main__
            if hasattr(__main__, var):
                delattr(__main__, var)
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            print(f"GPU {i} après nettoyage : {free/1e9:.1f}GB free / {total/1e9:.1f}GB total")


# ─── Chargement Mistral ───────────────────────────────────────────────────────
def load_mistral():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"

    if QUANTIZE:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Chargement {MODEL_ID} (4-bit NF4)...")
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", quantization_config=bnb)
    else:
        print(f"Chargement {MODEL_ID} (fp16)...")
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.float16)

    mdl.eval()
    print(f"Mistral chargé (quantize={QUANTIZE}, batch_size={BATCH_SIZE})")
    return mdl, tok


# ─── Inférence avec template Mistral ─────────────────────────────────────────
def make_mistral_infer(model, tokenizer):
    import torch, re

    def infer(queries, task, **kw):
        outputs = []
        max_tokens = TASK_MAX_TOKENS.get(task, INFER_PARAMS["max_new_tokens"])

        for i in range(0, len(queries), BATCH_SIZE):
            batch = queries[i: i + BATCH_SIZE]
            # Template Mistral-Instruct : <s>[INST] {query} [/INST]
            mistral_batch = [f"<s>[INST] {q} [/INST]" for q in batch]

            enc = tokenizer(
                mistral_batch,
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
                    out[input_len:], skip_special_tokens=True).strip()

                if task == "cause_and_effect":
                    m = re.search(r'[Ss]entence\s+([12])', text)
                    if m:
                        text = f"Sentence {m.group(1)}"
                    else:
                        tl = text.lower()
                        text = "Sentence 1" if "first" in tl else \
                               "Sentence 2" if "second" in tl else text
                else:
                    idx = text.find(".")
                    if idx > 0:
                        text = text[:idx]

                text = text.strip()
                print(f"  Mistral -> '{text[:60]}'")
                outputs.append(text)

        return outputs

    return infer


# ─── CSV protocole ────────────────────────────────────────────────────────────
def init_protocol_csv(path):
    os.makedirs("results", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "model", "task", "pnum",
            "original_prompt", "negative_stimulus",
            "few_shot", "do_sample", "temperature",
            "max_new_tokens", "batch_size", "quantized",
            "metric", "score",
        ])


def append_protocol(path, task, pnum, original_prompt, negative_stimulus, score):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            MODEL_NAME,
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


# ─── Lecture du score depuis le fichier résultat ──────────────────────────────
def read_last_score(filepath):
    if not os.path.exists(filepath):
        return 0.0
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if line.startswith("Test score:"):
            try:
                return float(line.split(":")[1].strip())
            except ValueError:
                pass
    return 0.0


# ─── Chargement des résultats Vicuna (Partie 1) ───────────────────────────────
def load_vicuna_results():
    """Charge protocol_vicuna.csv → baseline NP00 + meilleur score par tâche."""
    path = "results/protocol_vicuna.csv"
    vicuna = {}
    if not os.path.exists(path):
        print(f"Attention : {path} introuvable, comparaison Vicuna désactivée.")
        return vicuna
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task  = row["task"]
            pnum  = int(row["pnum"])
            score = float(row["score"])
            if task not in vicuna:
                vicuna[task] = {"baseline": 0.0, "best_score": 0.0, "best_pnum": 0}
            if pnum == 0:
                vicuna[task]["baseline"] = score
            if score > vicuna[task]["best_score"]:
                vicuna[task]["best_score"] = score
                vicuna[task]["best_pnum"]  = pnum
    return vicuna


# ─── Tableau comparatif ───────────────────────────────────────────────────────
def write_comparison(mistral_results, vicuna_results, comp_path):
    lines = []
    lines.append("=" * 76)
    lines.append("  AXE 4 — Comparaison Vicuna-13B vs Mistral-7B-Instruct-v0.2")
    lines.append("=" * 76)
    lines.append(
        f"\n  {'Tâche':<22} {'V-base':>7} {'V-best':>7} {'V-delta':>8} "
        f"{'M-base':>7} {'M-best':>7} {'M-delta':>8}  {'Gagnant':>8}"
    )
    lines.append("  " + "-" * 72)

    for task in SELECTED_TASKS:
        v   = vicuna_results.get(task, {})
        v_b = v.get("baseline", 0.0)
        v_s = v.get("best_score", 0.0)
        v_d = v_s - v_b

        m   = mistral_results.get(task, {})
        m_b = m.get("baseline", 0.0)
        m_s = m.get("best_score", 0.0)
        m_d = m_s - m_b

        if m_s > v_s + 0.005:
            winner = "MISTRAL"
        elif v_s > m_s + 0.005:
            winner = "VICUNA"
        else:
            winner = "EGAL"

        lines.append(
            f"  {task:<22} {v_b:>7.4f} {v_s:>7.4f} {v_d:>+8.4f} "
            f"{m_b:>7.4f} {m_s:>7.4f} {m_d:>+8.4f}  {winner:>8}"
        )

    lines.append("\n  V-base/M-base = score sans stimulus (pnum=0)")
    lines.append("  V-best/M-best = meilleur score sur les 10 NP")
    lines.append("  delta = best - base")
    lines.append("  Gagnant = modèle avec le meilleur score NP absolu")
    lines.append("\n" + "=" * 76)
    lines.append("  SENSIBILITÉ AUX NEGATIVE PROMPTS")
    lines.append("=" * 76)
    lines.append(
        f"\n  {'Tâche':<22} {'Δ Vicuna':>10} {'Δ Mistral':>10}  {'Plus sensible':>14}"
    )
    lines.append("  " + "-" * 60)

    for task in SELECTED_TASKS:
        v   = vicuna_results.get(task, {})
        m   = mistral_results.get(task, {})
        v_d = v.get("best_score", 0.0) - v.get("baseline", 0.0)
        m_d = m.get("best_score", 0.0) - m.get("baseline", 0.0)
        sens = "MISTRAL" if m_d > v_d + 0.005 else ("VICUNA" if v_d > m_d + 0.005 else "EGAL")
        lines.append(f"  {task:<22} {v_d:>+10.4f} {m_d:>+10.4f}  {sens:>14}")

    content = "\n".join(lines)
    print(content)
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nComparaison : {comp_path}")


# ─── Point d'entrée ──────────────────────────────────────────────────────────
def main():
    from config import PROMPT_SET, Negative_SET

    # Nettoyage GPU
    print("Nettoyage mémoire GPU...")
    free_gpu()

    # Chargement résultats Vicuna (Partie 1) pour comparaison finale
    vicuna_results = load_vicuna_results()

    # Chargement Mistral
    model, tokenizer = load_mistral()
    infer_fn = make_mistral_infer(model, tokenizer)

    # Monkeypatch
    import exec_accuracy
    exec_accuracy.get_response_from_llm = \
        lambda llm_model, queries, task, few_shot, **kw: infer_fn(queries, task)

    from main import run as main_run

    # Préparation dossiers
    shutil.rmtree(f"results/neg/{MODEL_NAME}", ignore_errors=True)
    os.makedirs(f"results/neg/{MODEL_NAME}", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    csv_path  = "results/protocol_mistral.csv"
    comp_path = "results/comparison_vicuna_mistral.txt"
    init_protocol_csv(csv_path)

    mistral_results = {task: {"baseline": 0.0, "best_score": 0.0, "best_pnum": 0}
                       for task in SELECTED_TASKS}
    total = len(SELECTED_TASKS) * 11
    done  = 0

    for task in SELECTED_TASKS:
        original_prompt = PROMPT_SET[task]
        for pnum in range(11):
            done += 1
            neg_stimulus = Negative_SET[pnum - 1] if pnum > 0 else "(aucun)"
            print(
                f"\n[{done}/{total}] mistral | {task} | pnum={pnum}"
                f"\n  Prompt original  : {original_prompt}"
                f"\n  Stimulus négatif : {neg_stimulus}",
                flush=True,
            )

            try:
                main_run(
                    task=task,
                    model=MODEL_NAME,
                    pnum=pnum,
                    few_shot=False,
                )
                result_file = f"results/neg/{MODEL_NAME}/{task}.txt"
                score = read_last_score(result_file)
                print(f"  Score : {score:.4f}")
            except Exception as e:
                print(f"  ERREUR : {e}")
                score = 0.0

            # Mise à jour résultats Mistral
            if pnum == 0:
                mistral_results[task]["baseline"] = score
            if score > mistral_results[task]["best_score"]:
                mistral_results[task]["best_score"] = score
                mistral_results[task]["best_pnum"]  = pnum

            append_protocol(
                csv_path, task, pnum,
                original_prompt, neg_stimulus, score,
            )

    # Tableau comparatif final
    write_comparison(mistral_results, vicuna_results, comp_path)
    print(f"\nProtocole CSV : {csv_path}")
    print("\nMistral 5 tâches terminé !")


if __name__ == "__main__":
    main()
