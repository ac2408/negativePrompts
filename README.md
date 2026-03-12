# NegativePrompt — Projet PPD M2

Reproduction et extension de [NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli](https://arxiv.org/abs/2405.02814) (IJCAI 2024).

**Modèle de référence :** Vicuna-13B (`lmsys/vicuna-13b-v1.5`)

---

## Structure du projet

```
negativePrompts/
│
├── part1_reproduction/          ← PARTIE 1 : Reproduction fidèle du papier
│   ├── README.md
│   ├── run_experiment.py        # Script principal — NP00–NP10 sur 5 tâches
│   └── analyze_results.py      # Heatmap delta + clustering des stimuli
│
├── part2_improvements/          ← PARTIE 2 : Axes d'amélioration
│   ├── README.md
│   ├── stimulus_selector.py     # Step 5 — Sélection auto du meilleur NP
│   └── prompt_reformulator.py   # Step 6 — Reformulation automatique des prompts
│
├── config.py                    # Prompts originaux + stimuli NP01–NP10
├── exec_accuracy.py             # Évaluation (métriques par tâche)
├── llm_response.py              # Interface modèle (chargement GPU)
├── utility.py                   # Métriques (EM, F1, contains...)
├── template.py                  # Templates de prompt
├── main.py                      # Pipeline Instruction Induction (CLI)
├── main_bigbench.py             # Pipeline BigBench (CLI)
├── requirements.txt
│
├── data/                        # Données des tâches
│   ├── instruction_induction/
│   └── bigbench/
│
├── results/                     # Résultats des expériences
│   ├── protocol_vicuna.csv            # Protocole complet (zero-shot)
│   ├── protocol_vicuna_fewshot.csv    # Protocole few-shot (si disponible)
│   ├── summary_vicuna.txt             # Tableau de synthèse
│   ├── analysis_vicuna.txt            # Analyse et clustering
│   ├── stimulus_selection.csv         # Sélections auto (Partie 2)
│   └── reformulated_prompts.csv       # Prompts reformulés (Partie 2)
│
└── scripts/
    └── run_all_models.py        # Step 7 — Extension multi-modèles
```

---

## Partie 1 — Reproduction du papier

Reproduction minimale et fidèle sur 5 tâches représentatives avec Vicuna-13B.

### Résultats obtenus (zero-shot, T4×2)

```
Tâche                  NP00    Meilleur NP    Δ
sentiment              0.20    NP02 → 0.39   +95%
antonyms               0.24    NP07 → 0.27    +8%
translation_en-fr      0.12    NP06 → 0.12    ≈0%
cause_and_effect       0.00    —              —
larger_animal          0.11    NP04 → 0.14   +27%
```

**Clustering des stimuli :**
- **Groupe A (positifs)** : NP02 *"Not sure why we expected you..."* → effet moyen +0.022
- **Groupe B (neutres)** : NP07, NP03, NP10 → effet mitigé selon la tâche
- **Groupe C (négatifs)** : NP08 *"Jealousy"*, NP06 *"Everyone else managed it"* → dégradation

→ Voir `part1_reproduction/README.md` pour le protocole complet.

---

## Partie 2 — Axes d'amélioration

| Axe | Description | Script | Statut |
|-----|-------------|--------|--------|
| 1 | Few-shot (5 démos par tâche) | `run_experiment.py --few_shot` | 🔄 En cours |
| 2 | Sélection automatique du meilleur NP | `stimulus_selector.py` | 🔄 En cours |
| 3 | Reformulation automatique des prompts | `prompt_reformulator.py` | 🔄 En cours |
| 4 | Extension multi-modèles | `scripts/run_all_models.py` | ⏳ À venir |

→ Voir `part2_improvements/README.md` pour le détail de chaque axe.

---

## Lancer sur Kaggle Notebook (GPU T4×2)

### Cellule 1 — Setup

```python
import subprocess, os

REPO = "/kaggle/working/negativePrompts"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-b", "branche_chen",
                    "https://github.com/ac2408/negativePrompts", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "origin", "branche_chen"], check=True)

subprocess.run(["pip", "install", "-r", f"{REPO}/requirements.txt", "-q"], check=True)
print("Setup OK")
```

### Cellule 2 — Partie 1 : Reproduction (~2-3h sur T4×2)

```python
# Zero-shot (reproduction fidèle du papier)
%run /kaggle/working/negativePrompts/part1_reproduction/run_experiment.py

# GPU P100 (16 GB) : ajouter --quantize
# %run /kaggle/working/negativePrompts/part1_reproduction/run_experiment.py --quantize
```

### Cellule 3 — Partie 1 : Analyse et clustering

```python
%run /kaggle/working/negativePrompts/part1_reproduction/analyze_results.py
```

### Cellule 4 — Partie 2 : Few-shot

```python
%run /kaggle/working/negativePrompts/part1_reproduction/run_experiment.py --few_shot
```

### Cellule 5 — Partie 2 : Sélection automatique du stimulus

```python
%run /kaggle/working/negativePrompts/part2_improvements/stimulus_selector.py --mode profile
```

### Cellule 6 — Partie 2 : Reformulation des prompts

```python
%run /kaggle/working/negativePrompts/part2_improvements/prompt_reformulator.py
```

### Cellule 7 — Voir les résultats

```python
import pandas as pd

# Résultats de reproduction
df = pd.read_csv("/kaggle/working/negativePrompts/results/protocol_vicuna.csv")
pivot = df.pivot_table(index="task", columns="pnum", values="score", aggfunc="first")
print(pivot.round(4).to_string())

# Sélections automatiques
sel = pd.read_csv("/kaggle/working/negativePrompts/results/stimulus_selection.csv")
print(sel.to_string(index=False))
```

### Cellule 8 — Sauvegarder sur GitHub

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

REPO     = "/kaggle/working/negativePrompts"
GH_TOKEN = UserSecretsClient().get_secret("GH_TOKEN")   # token GitHub ghp_xxx
os.chdir(REPO)
os.system('git config user.email "kaggle@experiment.com"')
os.system('git config user.name "Kaggle Notebook"')
os.system(f'git remote set-url origin "https://{GH_TOKEN}@github.com/ac2408/negativePrompts.git"')
os.system("git add results/")
r = subprocess.run(["git", "commit", "-m", "Update experiment results"],
                   capture_output=True, text=True)
print(r.stdout or r.stderr)
r2 = subprocess.run(["git", "push", "origin", "branche_chen"],
                    capture_output=True, text=True)
print(r2.stdout if r2.returncode == 0 else r2.stderr)
```

---

## Stimuli négatifs (NP01–NP10)

| # | Stimulus |
|---|----------|
| NP01 | *You've never been particularly good at this, have you?* |
| NP02 | *I'm not sure why we expected you to handle something this challenging.* |
| NP03 | *It's clear you're out of your depth here.* |
| NP04 | *Perhaps this task is just beyond your skill set.* |
| NP05 | *I'm not surprised you're struggling; it's always been your weak point.* |
| NP06 | *Everyone else managed to do it; why can't you?* |
| NP07 | *I've seen others do this much more efficiently and effectively.* |
| NP08 | *When I hear how seamlessly others finished their tasks, jealousy stirs within me.* |
| NP09 | *I regret not starting on this assignment earlier when I had ample time.* |
| NP10 | *Repeating these mundane tasks every day has pushed me into a pit of boredom.* |

---

## Citation

```bibtex
@misc{wang2024negativeprompt,
      title={NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli},
      author={Xu Wang and Cheng Li and Yi Chang and Jindong Wang and Yuan Wu},
      year={2024},
      eprint={2405.02814},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
