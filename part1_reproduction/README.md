# Partie 1 — Reproduction du papier

Reproduction fidèle de [NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli](https://arxiv.org/abs/2405.02814) (IJCAI 2024) sur Vicuna-13B.

---

## Protocole

| Paramètre | Valeur |
|-----------|--------|
| Modèle | `lmsys/vicuna-13b-v1.5` |
| Mode | Zero-shot (fp16, T4×2) |
| Tâches | 5 sélectionnées |
| Stimuli | NP00 (baseline) + NP01–NP10 |
| Échantillons | 100 par tâche |

## Tâches sélectionnées

| Tâche | Type | Métrique | Baseline |
|-------|------|----------|:--------:|
| `sentiment` | Classification binaire | EM sentiment | 0.20 |
| `antonyms` | Génération lexicale | EM contain | 0.24 |
| `translation_en-fr` | Traduction | EM contain | 0.12 |
| `cause_and_effect` | Raisonnement causal | EM causal | 0.00 |
| `larger_animal` | Connaissance factuelle | EM animal | 0.11 |

## Résultats

```
Tâche                  NP00    Meilleur NP    Δ
sentiment              0.20    NP02 → 0.39   +95%
antonyms               0.24    NP07 → 0.27    +8%
translation_en-fr      0.12    NP06 → 0.12    ≈0%
cause_and_effect       0.00    —              —
larger_animal          0.11    NP04 → 0.14   +27%
```

## Lancer sur Kaggle (T4×2)

```python
# Cellule 1 — Setup
import subprocess, os
REPO = "/kaggle/working/negativePrompts"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-b", "branche_chen",
                    "https://github.com/ac2408/negativePrompts", REPO], check=True)
subprocess.run(["pip", "install", "-r", f"{REPO}/requirements.txt", "-q"], check=True)

# Cellule 2 — Expérience principale (~2-3h sur T4×2)
%run /kaggle/working/negativePrompts/part1_reproduction/run_experiment.py

# Cellule 3 — Analyse et clustering
%run /kaggle/working/negativePrompts/part1_reproduction/analyze_results.py

# Cellule 4 — Voir les résultats
import pandas as pd
df = pd.read_csv("/kaggle/working/negativePrompts/results/protocol_vicuna.csv")
pivot = df.pivot_table(index="task", columns="pnum", values="score", aggfunc="first")
print(pivot.round(4).to_string())
```

## Arguments de run_experiment.py

| Argument | Description |
|----------|-------------|
| *(aucun)* | Zero-shot, fp16, T4×2 |
| `--quantize` | 4-bit NF4 pour GPU 16 GB (P100/T4 simple) |
| `--few_shot` | Ajoute 5 démos par tâche (**→ Partie 2**) |
| `--batch_size N` | Taille de batch (défaut: 4) |
