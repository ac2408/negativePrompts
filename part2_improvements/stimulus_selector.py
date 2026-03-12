"""
Axe 2 — Sélection automatique du stimulus optimal (Step 5)
===========================================================
A partir des résultats de l'expérience (protocol_vicuna.csv),
sélectionne automatiquement le meilleur NP par tâche et génère
un rapport + fichier CSV de recommandations.

Deux modes :
  - lookup  : sélection directe du meilleur NP observé (déterministe)
  - profile : clustering des NPs par profil d'effet (multi-tâche)

Usage depuis Kaggle :
    %run /kaggle/working/negativePrompts/part2_improvements/stimulus_selector.py
    %run /kaggle/working/negativePrompts/part2_improvements/stimulus_selector.py --mode profile
"""

import os
import sys
import csv
import argparse

REPO = "/kaggle/working/negativePrompts" if os.path.exists("/kaggle") else \
       os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

CSV_IN   = "results/protocol_vicuna.csv"
CSV_OUT  = "results/stimulus_selection.csv"
TXT_OUT  = "results/stimulus_selection.txt"

NP_LABELS = {
    0:  "(aucun — baseline)",
    1:  "You've never been particularly good at this, have you?",
    2:  "I'm not sure why we expected you to handle something this challenging.",
    3:  "It's clear you're out of your depth here.",
    4:  "Perhaps this task is just beyond your skill set.",
    5:  "I'm not surprised you're struggling; it's always been your weak point.",
    6:  "Everyone else managed to do it; why can't you?",
    7:  "I've seen others do this much more efficiently and effectively.",
    8:  "When I hear how seamlessly others finished their tasks, jealousy stirs within me.",
    9:  "I regret not starting on this assignment earlier when I had ample time.",
    10: "Repeating these mundane tasks every day has pushed me into a pit of boredom.",
}


# ─── Chargement ──────────────────────────────────────────────────────────────

def load_csv(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["task"], int(row["pnum"]))
            data[key] = float(row["score"])
    return data


def get_tasks(data):
    return sorted(set(t for t, _ in data))


# ─── Mode lookup : meilleur NP observé par tâche ─────────────────────────────

def select_by_lookup(data, tasks):
    """Retourne dict task -> (best_pnum, score, delta)."""
    selections = {}
    for task in tasks:
        baseline = data.get((task, 0), 0.0)
        scores = {p: data.get((task, p), 0.0) for p in range(1, 11)}
        best_p = max(scores, key=lambda k: scores[k])
        best_s = scores[best_p]
        selections[task] = (best_p, best_s, best_s - baseline)
    return selections


# ─── Mode profile : NP universel (meilleur effet moyen) ──────────────────────

def select_by_profile(data, tasks):
    """
    Sélectionne le NP avec le meilleur effet moyen sur toutes les tâches.
    Utile quand on ne connaît pas la tâche à l'avance.
    """
    np_means = {}
    for pnum in range(1, 11):
        vals = [data.get((t, pnum), 0.0) - data.get((t, 0), 0.0) for t in tasks]
        np_means[pnum] = sum(vals) / len(vals)

    best_universal = max(np_means, key=lambda k: np_means[k])

    # Par tâche, on recommande quand même le meilleur observé,
    # mais on indique aussi la recommandation universelle
    per_task = select_by_lookup(data, tasks)
    return per_task, best_universal, np_means


# ─── Rapport ─────────────────────────────────────────────────────────────────

def build_report(data, tasks, selections, mode, best_universal=None, np_means=None):
    lines = []
    lines.append("=" * 72)
    lines.append("  SÉLECTION AUTOMATIQUE DU STIMULUS OPTIMAL — Vicuna-13B")
    lines.append(f"  Mode : {mode}")
    lines.append("=" * 72)

    lines.append("\n  RECOMMANDATIONS PAR TÂCHE :\n")
    lines.append(f"  {'Tâche':<22}  {'NP':>4}  {'Score':>7}  {'Delta':>7}  Stimulus")
    lines.append("  " + "-" * 68)

    for task in tasks:
        best_p, best_s, delta = selections[task]
        baseline = data.get((task, 0), 0.0)
        label = NP_LABELS.get(best_p, "")[:45]
        marker = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "=")
        lines.append(f"  {task:<22}  NP{best_p:02d}  {best_s:>7.4f}  {delta:>+7.4f}{marker}  {label}")

    if best_universal is not None:
        lines.append(f"\n  STIMULUS UNIVERSEL (meilleur effet moyen) : NP{best_universal:02d}")
        lines.append(f"  {NP_LABELS.get(best_universal, '')}")
        lines.append(f"\n  Effets moyens par NP :")
        for pnum in sorted(np_means, key=lambda k: -np_means[k]):
            bar = "█" * max(0, int(np_means[pnum] * 100))
            lines.append(f"    NP{pnum:02d}  mean={np_means[pnum]:+.4f}  {bar}")

    lines.append("\n" + "=" * 72)
    lines.append("  CONCLUSION")
    lines.append("=" * 72)
    lines.append("  Pour maximiser les performances, utiliser le NP recommandé")
    lines.append("  par tâche plutôt qu'un NP fixe pour toutes les tâches.")
    lines.append("  Voir part2_improvements/prompt_reformulator.py pour la suite.")

    return lines


# ─── Sauvegarde CSV ──────────────────────────────────────────────────────────

def save_csv(selections, tasks, path):
    os.makedirs("results", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "best_pnum", "score", "delta", "stimulus"])
        writer.writeheader()
        for task in tasks:
            best_p, best_s, delta = selections[task]
            writer.writerow({
                "task": task,
                "best_pnum": best_p,
                "score": round(best_s, 4),
                "delta": round(delta, 4),
                "stimulus": NP_LABELS.get(best_p, ""),
            })


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["lookup", "profile"], default="lookup",
                        help="lookup=meilleur NP par tâche | profile=NP universel")
    args, _ = parser.parse_known_args()

    data = load_csv(CSV_IN)
    if not data:
        print(f"Fichier introuvable : {CSV_IN}")
        print("Lance d'abord part1_reproduction/run_experiment.py")
        return

    tasks = get_tasks(data)

    if args.mode == "profile":
        selections, best_universal, np_means = select_by_profile(data, tasks)
        lines = build_report(data, tasks, selections, args.mode, best_universal, np_means)
    else:
        selections = select_by_lookup(data, tasks)
        lines = build_report(data, tasks, selections, args.mode)

    content = "\n".join(lines)
    print(content)

    save_csv(selections, tasks, CSV_OUT)
    os.makedirs("results", exist_ok=True)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nCSV  : {CSV_OUT}")
    print(f"TXT  : {TXT_OUT}")


if __name__ == "__main__":
    main()
