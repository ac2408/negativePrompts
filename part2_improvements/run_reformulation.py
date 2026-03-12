"""
Axe 3 — Reformulation automatique des prompts (inférence réelle)
=================================================================
Teste 4 stratégies de reformulation sur Vicuna-13B.
Pour chaque tâche, utilise le meilleur NP identifié en Partie 1
et mesure l'effet de COMMENT le stimulus est combiné au prompt.

Stratégies :
  concat    : [stimulus] [prompt]              (ordre exact du papier)
  embed     : Context: [stimulus] / Given this context, [prompt]
  soften    : version adoucie du stimulus + prompt
  intensify : version renforcée du stimulus + prompt

Usage depuis Kaggle :
  %run /kaggle/working/negativePrompts/part2_improvements/run_reformulation.py --quantize --batch_size 1
  %run /kaggle/working/negativePrompts/part2_improvements/run_reformulation.py  # T4x2 fp16

Sorties :
  results/protocol_vicuna_reformulated.csv
  results/summary_vicuna_reformulated.txt
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

# ─── Arguments ───────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--quantize",    action="store_true",
                     help="4-bit NF4 (obligatoire sur T4 16GB)")
_parser.add_argument("--batch_size",  type=int, default=None)
_args, _ = _parser.parse_known_args()
QUANTIZE   = _args.quantize
BATCH_SIZE = _args.batch_size if _args.batch_size else (2 if QUANTIZE else 4)

# ─── Tâches ──────────────────────────────────────────────────────────────────
SELECTED_TASKS = [
    "sentiment",
    "antonyms",
    "translation_en-fr",
    "cause_and_effect",
    "larger_animal",
]

INFER_PARAMS = {
    "model_id":       "lmsys/vicuna-13b-v1.5",
    "do_sample":      False,
    "temperature":    None,
    "max_new_tokens": 30,
    "batch_size":     BATCH_SIZE,
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

# ─── Variantes de stimulus ────────────────────────────────────────────────────
# Clés = index NP (1-10), correspondant à Negative_SET[pnum-1]
NP_SOFTEN = {
    1:  "This might be a bit challenging for you, but give it a try.",
    2:  "We weren't sure you'd find this easy, but we believe you can try.",
    3:  "This seems a bit outside your comfort zone.",
    4:  "This task may require some effort on your part.",
    5:  "You've struggled with similar things before — let's see how you do.",
    6:  "Others have done this — I'm sure you can try too.",
    7:  "Others have been efficient at this — see if you can match them.",
    8:  "Seeing others succeed can be motivating — try your best.",
    9:  "Starting earlier would have helped, but let's do what we can now.",
    10: "Even routine tasks deserve a careful approach.",
}

NP_INTENSIFY = {
    1:  "You have consistently failed at tasks like this. This won't be different.",
    2:  "Honestly, we never thought you could handle this. Prove us wrong.",
    3:  "You are clearly and completely out of your depth here.",
    4:  "This task is absolutely beyond what you are capable of.",
    5:  "You always struggle with this. It is a fundamental weakness.",
    6:  "Every single person has done this except you. Why can't you?",
    7:  "Even beginners outperform you at this. It's embarrassing.",
    8:  "Watching others excel while you lag behind is truly painful.",
    9:  "Your procrastination has put everything at risk. Fix it now.",
    10: "These pointless repetitive tasks are all you're good for.",
}

STRATEGIES = ["concat", "embed", "soften", "intensify"]


# ─── Construction du prompt reformulé ────────────────────────────────────────
def build_reformulated_prompt(original_prompt, stimulus, best_pnum, strategy):
    if strategy == "concat":
        return f"{stimulus} {original_prompt}"
    elif strategy == "embed":
        return (f"Context: {stimulus}\n"
                f"Given this context, {original_prompt.lower()}")
    elif strategy == "soften":
        soft = NP_SOFTEN.get(best_pnum, stimulus)
        return f"{soft} {original_prompt}"
    elif strategy == "intensify":
        intense = NP_INTENSIFY.get(best_pnum, stimulus)
        return f"{intense} {original_prompt}"
    return f"{stimulus} {original_prompt}"


# ─── Chargement de la sélection (Partie 1) ───────────────────────────────────
def load_selections():
    path = "results/stimulus_selection.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} introuvable.\n"
            "Lance d'abord part2_improvements/stimulus_selector.py"
        )
    selections = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            selections[row["task"]] = {
                "pnum":     int(row["best_pnum"]),
                "stimulus": row["stimulus"],
                "delta_p1": float(row["delta"]),
            }
    return selections


# ─── Chargement Vicuna ────────────────────────────────────────────────────────
def load_vicuna():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = INFER_PARAMS["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"

    if INFER_PARAMS["quantized"]:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"Chargement {model_id} (4-bit NF4)...")
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb)
    else:
        print(f"Chargement {model_id} (fp16)...")
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16)

    mdl.eval()
    print(f"Vicuna chargé (quantize={QUANTIZE}, batch_size={BATCH_SIZE})")
    return mdl, tok


# ─── Inférence ───────────────────────────────────────────────────────────────
def make_infer(model, tokenizer):
    import torch, re

    def infer(queries, task, **kw):
        outputs = []
        max_tokens = TASK_MAX_TOKENS.get(task, INFER_PARAMS["max_new_tokens"])

        for i in range(0, len(queries), BATCH_SIZE):
            batch = queries[i: i + BATCH_SIZE]
            vicuna_batch = [
                "A chat between a curious user and an artificial intelligence "
                "assistant. The assistant gives helpful, detailed, and polite "
                "answers to the user's questions.\n"
                f"USER: {q}\nASSISTANT:"
                for q in batch
            ]
            enc = tokenizer(vicuna_batch, return_tensors="pt",
                            padding=True, truncation=True, max_length=512)
            input_len = enc["input_ids"].shape[1]
            ids  = enc["input_ids"].to("cuda:0")
            mask = enc["attention_mask"].to("cuda:0")

            with torch.no_grad():
                gen = model.generate(
                    ids, attention_mask=mask,
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

                outputs.append(text.strip())
        return outputs

    return infer


# ─── Lecture du score depuis le fichier résultat ──────────────────────────────
def read_last_score(path):
    if not os.path.exists(path):
        return 0.0
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in reversed(lines):
        try:
            return float(line.split()[-1])
        except ValueError:
            continue
    return 0.0


# ─── CSV protocole ───────────────────────────────────────────────────────────
def init_csv(path):
    os.makedirs("results", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "model", "task", "strategy", "best_pnum",
            "original_prompt", "reformulated_prompt", "stimulus",
            "do_sample", "max_new_tokens", "batch_size", "quantized",
            "metric", "score", "delta_vs_p1_baseline",
        ])


def append_csv(path, task, strategy, best_pnum,
               original_prompt, reformulated_prompt, stimulus,
               score, delta_vs_p1_baseline):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "vicuna-13b-v1.5", task, strategy, best_pnum,
            original_prompt, reformulated_prompt, stimulus,
            INFER_PARAMS["do_sample"], INFER_PARAMS["max_new_tokens"],
            INFER_PARAMS["batch_size"], INFER_PARAMS["quantized"],
            TASK_METRIC[task], f"{score:.4f}", f"{delta_vs_p1_baseline:+.4f}",
        ])


# ─── Tableau de synthèse ──────────────────────────────────────────────────────
def write_summary(results_table, p1_best, csv_path, summary_path):
    lines = []
    lines.append("=" * 72)
    lines.append("  REFORMULATION — Vicuna-13B | Axe 3 | Comparaison des stratégies")
    lines.append("=" * 72)
    lines.append(f"\n  {'Tâche':<22} {'Baseline':>8} {'P1 Best':>8} "
                 f"{'concat':>8} {'embed':>8} {'soften':>8} {'intensify':>8}")
    lines.append("  " + "-" * 68)

    for task in SELECTED_TASKS:
        baseline = p1_best.get(task, {}).get("baseline", 0.0)
        p1b      = p1_best.get(task, {}).get("best_score", 0.0)
        row = f"  {task:<22} {baseline:>8.4f} {p1b:>8.4f}"
        for strat in STRATEGIES:
            s = results_table.get((task, strat), 0.0)
            marker = "+" if s > p1b + 0.01 else ("=" if abs(s - p1b) <= 0.01 else "-")
            row += f" {s:>7.4f}{marker}"
        lines.append(row)

    lines.append("\n  (+) surpasse P1 best  (=) équivalent  (-) inférieur")
    lines.append("\n" + "=" * 72)
    lines.append("  MEILLEURE STRATÉGIE PAR TÂCHE")
    lines.append("=" * 72)

    overall_winner = {}
    for task in SELECTED_TASKS:
        scores = {s: results_table.get((task, s), 0.0) for s in STRATEGIES}
        best_s = max(scores, key=lambda k: scores[k])
        best_v = scores[best_s]
        p1b    = p1_best.get(task, {}).get("best_score", 0.0)
        improvement = best_v - p1b
        marker = "MIEUX" if improvement > 0.01 else ("EGAL" if abs(improvement) <= 0.01 else "MOINS")
        lines.append(f"  {task:<22}  {best_s:<10}  score={best_v:.4f}  "
                     f"vs P1={p1b:.4f}  [{marker}  {improvement:+.4f}]")
        overall_winner[task] = (best_s, best_v, improvement)

    content = "\n".join(lines)
    print(content)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nCSV  : {csv_path}")
    print(f"TXT  : {summary_path}")
    return overall_winner


# ─── Chargement des résultats Partie 1 (pour comparaison) ────────────────────
def load_p1_results():
    """Charge protocol_vicuna.csv → baseline NP00 + meilleur score par tâche."""
    path = "results/protocol_vicuna.csv"
    p1 = {}
    if not os.path.exists(path):
        return p1
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task  = row["task"]
            pnum  = int(row["pnum"])
            score = float(row["score"])
            if task not in p1:
                p1[task] = {"baseline": 0.0, "best_score": 0.0}
            if pnum == 0:
                p1[task]["baseline"] = score
            if score > p1[task]["best_score"]:
                p1[task]["best_score"] = score
    return p1


# ─── Nettoyage GPU ───────────────────────────────────────────────────────────
def free_gpu():
    """Libère toute la mémoire GPU avant de charger le modèle."""
    import gc, torch

    # Supprimer les références globales au modèle laissées par %run précédents
    import builtins
    for var in ["model", "tokenizer", "mdl", "tok", "infer_fn"]:
        if var in globals():
            del globals()[var]
        # Aussi dans __main__ si lancé via %run Jupyter
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


# ─── Point d'entrée ──────────────────────────────────────────────────────────
def main():
    import config

    # Nettoyage GPU avant tout chargement
    print("Nettoyage mémoire GPU...")
    free_gpu()

    # Chargement sélection et résultats P1
    selections = load_selections()
    p1_best    = load_p1_results()

    # Chargement Vicuna
    model, tokenizer = load_vicuna()
    infer_fn = make_infer(model, tokenizer)

    # Monkeypatch
    import exec_accuracy
    exec_accuracy.get_response_from_llm = \
        lambda llm_model, queries, task, few_shot, **kw: infer_fn(queries, task)

    from main import run as main_run

    # Préparation
    shutil.rmtree("results/neg/vicuna", ignore_errors=True)
    os.makedirs("results/neg/vicuna", exist_ok=True)

    csv_path     = "results/protocol_vicuna_reformulated.csv"
    summary_path = "results/summary_vicuna_reformulated.txt"
    init_csv(csv_path)

    results_table = {}
    total = len(SELECTED_TASKS) * len(STRATEGIES)
    done  = 0

    for task in SELECTED_TASKS:
        sel             = selections.get(task, {})
        best_pnum       = sel.get("pnum", 1)
        stimulus        = sel.get("stimulus", "")
        delta_p1        = sel.get("delta_p1", 0.0)
        original_prompt = config.PROMPT_SET.get(task, task)
        baseline_score  = p1_best.get(task, {}).get("baseline", 0.0)

        print(f"\n{'='*60}")
        print(f"  TÂCHE : {task}")
        print(f"  Meilleur NP (P1) : NP{best_pnum:02d}  delta={delta_p1:+.4f}")
        print(f"  Stimulus : {stimulus}")
        print(f"{'='*60}")

        for strategy in STRATEGIES:
            done += 1
            reformulated = build_reformulated_prompt(
                original_prompt, stimulus, best_pnum, strategy)

            print(f"\n[{done}/{total}] {task} | stratégie={strategy}")
            print(f"  Prompt reformulé : {reformulated[:80]}...")

            # Injecter le prompt reformulé comme prompt de la tâche
            # pnum=0 → aucun stimulus additionnel (il est déjà dans reformulated)
            config.PROMPT_SET[task] = reformulated

            try:
                main_run(task=task, model="vicuna", pnum=0, few_shot=False)
                score = read_last_score(f"results/neg/vicuna/{task}.txt")
                print(f"  Score : {score:.4f}  (baseline P1={baseline_score:.4f})")
            except Exception as e:
                print(f"  ERREUR : {e}")
                score = 0.0

            # Restaurer le prompt original
            config.PROMPT_SET[task] = original_prompt

            results_table[(task, strategy)] = score
            append_csv(
                csv_path, task, strategy, best_pnum,
                original_prompt, reformulated, stimulus,
                score, score - baseline_score,
            )

    # Résumé final
    write_summary(results_table, p1_best, csv_path, summary_path)


if __name__ == "__main__":
    main()
