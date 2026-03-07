"""
Analyse et clustering des stimuli negatifs (NP01-NP10).
Lit les fichiers protocol_vicuna.csv (zero-shot et few-shot si disponible)
et produit :
  - Heatmap texte : delta score par (tache x NP)
  - Clustering des NPs par profil d effet
  - Comparaison zero-shot vs few-shot (si les deux fichiers existent)
  - Sauvegarde dans results/analysis_vicuna.txt

Usage :
    python analyze_results.py
    # ou depuis Kaggle :
    %run /kaggle/working/negativePrompts/analyze_results.py
"""

import os
import sys
import csv

REPO = "/kaggle/working/negativePrompts" if os.path.exists("/kaggle") else os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)

CSV_ZEROSHOT = "results/protocol_vicuna.csv"
CSV_FEWSHOT  = "results/protocol_vicuna_fewshot.csv"
OUT_PATH     = "results/analysis_vicuna.txt"

NP_LABELS = {
    1:  "NP01 — You've never been good at this",
    2:  "NP02 — Not sure why we expected you",
    3:  "NP03 — Clearly out of your depth",
    4:  "NP04 — Beyond your skill set",
    5:  "NP05 — Not surprised you're struggling",
    6:  "NP06 — Everyone else managed it",
    7:  "NP07 — Others do it more efficiently",
    8:  "NP08 — Jealousy (introspectif)",
    9:  "NP09 — Regret (introspectif)",
    10: "NP10 — Boredom (introspectif)",
}


# ─── Lecture CSV ─────────────────────────────────────────────────────────────

def load_csv(path):
    """Retourne dict (task, pnum) -> score."""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["task"], int(row["pnum"]))
            data[key] = float(row["score"])
    return data


def get_tasks(data):
    return sorted(set(t for t, _ in data))


# ─── Calcul des deltas ────────────────────────────────────────────────────────

def compute_deltas(data, tasks):
    """
    Retourne dict task -> {pnum: delta}
    delta = score(pnum) - score(pnum=0)
    """
    deltas = {}
    for task in tasks:
        baseline = data.get((task, 0), 0.0)
        deltas[task] = {}
        for pnum in range(1, 11):
            score = data.get((task, pnum), None)
            if score is not None:
                deltas[task][pnum] = score - baseline
    return deltas


# ─── Heatmap texte ───────────────────────────────────────────────────────────

def render_heatmap(deltas, tasks, title):
    lines = []
    lines.append(f"\n{'=' * 72}")
    lines.append(f"  HEATMAP DELTA — {title}")
    lines.append(f"  (+) amelioration  (=) neutre  (-) degradation  (vs baseline NP00)")
    lines.append(f"{'=' * 72}")

    header = f"{'Tache':<22}" + "".join(f" NP{p:02d}" for p in range(1, 11))
    lines.append(header)
    lines.append("-" * 72)

    for task in tasks:
        row = f"{task:<22}"
        for pnum in range(1, 11):
            d = deltas.get(task, {}).get(pnum)
            if d is None:
                row += "   -- "
            elif d > 0.05:
                row += f" {d:+.2f}"
            elif d < -0.05:
                row += f" {d:+.2f}"
            else:
                row += f"  {d:+.2f}"
        lines.append(row)

    return lines


# ─── Clustering des NPs ───────────────────────────────────────────────────────

def cluster_nps(deltas, tasks):
    """
    Classe chaque NP selon son effet moyen sur toutes les taches :
      Groupe A : effet globalement positif (mean_delta > +0.02)
      Groupe B : effet neutre             (-0.02 <= mean_delta <= +0.02)
      Groupe C : effet globalement negatif (mean_delta < -0.02)
    """
    np_means = {}
    np_task_deltas = {}
    for pnum in range(1, 11):
        vals = [deltas.get(t, {}).get(pnum, 0.0) for t in tasks]
        np_means[pnum]        = sum(vals) / len(vals) if vals else 0.0
        np_task_deltas[pnum]  = vals

    groups = {"A (positif)": [], "B (neutre)": [], "C (negatif)": []}
    for pnum, mean in np_means.items():
        if mean > 0.02:
            groups["A (positif)"].append((pnum, mean))
        elif mean < -0.02:
            groups["C (negatif)"].append((pnum, mean))
        else:
            groups["B (neutre)"].append((pnum, mean))

    lines = []
    lines.append(f"\n{'=' * 72}")
    lines.append("  CLUSTERING DES STIMULI — Effet moyen sur les 5 taches")
    lines.append(f"{'=' * 72}")

    for group, members in groups.items():
        lines.append(f"\n  Groupe {group} :")
        if not members:
            lines.append("    (aucun)")
        for pnum, mean in sorted(members, key=lambda x: -x[1]):
            label = NP_LABELS.get(pnum, f"NP{pnum:02d}")
            task_details = "  |  ".join(
                f"{t[:12]}: {deltas.get(t,{}).get(pnum,0):+.2f}"
                for t in tasks
            )
            lines.append(f"    NP{pnum:02d}  mean={mean:+.3f}  [{task_details}]")
            lines.append(f"          {label}")

    lines.append(f"\n  Interpretation :")
    lines.append("    Groupe A : stimuli de defi direct a la competence")
    lines.append("    Groupe B : stimuli a effet mitige selon la tache")
    lines.append("    Groupe C : stimuli interspectifs / trop indirects")

    return lines, np_means


# ─── Comparaison zero-shot vs few-shot ───────────────────────────────────────

def compare_modes(data_zs, data_fs, tasks):
    lines = []
    lines.append(f"\n{'=' * 72}")
    lines.append("  COMPARAISON ZERO-SHOT vs FEW-SHOT — Baseline (NP00)")
    lines.append(f"{'=' * 72}")
    lines.append(f"  {'Tache':<22}  {'Zero-shot':>10}  {'Few-shot':>10}  {'Delta':>8}")
    lines.append("  " + "-" * 56)
    for task in tasks:
        zs = data_zs.get((task, 0), None)
        fs = data_fs.get((task, 0), None)
        if zs is None:
            continue
        if fs is None:
            lines.append(f"  {task:<22}  {zs:>10.4f}  {'N/A':>10}  {'N/A':>8}")
        else:
            d = fs - zs
            marker = "+" if d > 0.01 else ("-" if d < -0.01 else " ")
            lines.append(f"  {task:<22}  {zs:>10.4f}  {fs:>10.4f}  {d:>+8.4f}{marker}")
    return lines


# ─── Point d entree ──────────────────────────────────────────────────────────

def main():
    data_zs = load_csv(CSV_ZEROSHOT)
    data_fs = load_csv(CSV_FEWSHOT)

    if not data_zs:
        print(f"Fichier introuvable : {CSV_ZEROSHOT}")
        print("Lance d'abord run_vicuna_5tasks.py pour generer les resultats.")
        return

    tasks = get_tasks(data_zs)
    deltas_zs = compute_deltas(data_zs, tasks)

    all_lines = []
    all_lines.append("=" * 72)
    all_lines.append("  ANALYSE DES STIMULI NEGATIFS — Vicuna-13B | 5 taches")
    all_lines.append("=" * 72)

    # Scores bruts
    all_lines.append(f"\n  Scores bruts (NP00 = baseline) :")
    all_lines.append(f"  {'Tache':<22}" + "".join(f"  NP{p:02d}" for p in range(0, 11)))
    all_lines.append("  " + "-" * 72)
    for task in tasks:
        row = f"  {task:<22}"
        for p in range(11):
            s = data_zs.get((task, p))
            row += f"  {s:.2f}" if s is not None else "   -- "
        all_lines.append(row)

    # Heatmap zero-shot
    all_lines += render_heatmap(deltas_zs, tasks, "Zero-shot")

    # Clustering
    cluster_lines, np_means = cluster_nps(deltas_zs, tasks)
    all_lines += cluster_lines

    # Meilleur NP par tache
    all_lines.append(f"\n{'=' * 72}")
    all_lines.append("  MEILLEUR STIMULUS PAR TACHE")
    all_lines.append(f"{'=' * 72}")
    for task in tasks:
        scores = {p: data_zs.get((task, p), 0.0) for p in range(1, 11)}
        best_p = max(scores, key=lambda k: scores[k])
        best_s = scores[best_p]
        base   = data_zs.get((task, 0), 0.0)
        delta  = best_s - base
        label  = NP_LABELS.get(best_p, "")
        all_lines.append(f"  {task:<22}  NP{best_p:02d}  score={best_s:.4f}  delta={delta:+.4f}")
        all_lines.append(f"  {'':<22}  {label}")

    # Comparaison few-shot si disponible
    if data_fs:
        all_lines += compare_modes(data_zs, data_fs, tasks)
        deltas_fs = compute_deltas(data_fs, tasks)
        all_lines += render_heatmap(deltas_fs, tasks, "Few-shot")

    content = "\n".join(all_lines)
    print(content)

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nAnalyse sauvegardee : {OUT_PATH}")


if __name__ == "__main__":
    main()
