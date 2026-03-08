# Partie 2 — Axes d'amélioration

Extension de la reproduction (Partie 1) par trois axes d'amélioration choisis.

---

## Axe 1 — Few-shot (exemples en contexte)

Le papier original teste uniquement en zero-shot.
Notre amélioration : ajouter 5 démonstrations par tâche pour guider le modèle.

**Script** : `part1_reproduction/run_experiment.py --few_shot`

```python
%run /kaggle/working/negativePrompts/part1_reproduction/run_experiment.py --few_shot
```

**Résultats** : `results/protocol_vicuna_fewshot.csv`

---

## Axe 2 — Sélection automatique du stimulus optimal (Step 5)

Au lieu d'essayer tous les NP à chaque fois, sélectionner automatiquement
le meilleur stimulus par tâche à partir des résultats de l'expérience.

**Script** : `part2_improvements/stimulus_selector.py`

```python
%run /kaggle/working/negativePrompts/part2_improvements/stimulus_selector.py
```

**Sortie** : `results/stimulus_selection.csv` + rapport textuel

---

## Axe 3 — Reformulation automatique des prompts (inférence réelle)

Teste 4 stratégies de reformulation sur Vicuna-13B en faisant tourner
le modèle et en mesurant les scores réels. Utilise le meilleur NP
identifié en Partie 1 par tâche.

| Stratégie | Description |
|-----------|-------------|
| `concat`    | `[stimulus] [prompt]` — ordre exact du papier |
| `embed`     | `Context: [stimulus] / Given this context, [prompt]` |
| `soften`    | Version adoucie du stimulus + prompt |
| `intensify` | Version renforcée du stimulus + prompt |

**Script principal** : `part2_improvements/run_reformulation.py`

```python
# T4 16GB (quantize obligatoire)
%run /kaggle/working/negativePrompts/part2_improvements/run_reformulation.py --quantize --batch_size 1

# T4x2 32GB (fp16 natif)
%run /kaggle/working/negativePrompts/part2_improvements/run_reformulation.py
```

**Sorties** :
- `results/protocol_vicuna_reformulated.csv` — scores par tâche × stratégie
- `results/summary_vicuna_reformulated.txt`  — tableau comparatif vs Partie 1

**Pré-requis** : `results/stimulus_selection.csv` (lancer `stimulus_selector.py` d'abord)

---

## Axe 4 — Extension multi-modèles (Step 7)

Compare **Mistral-7B-Instruct-v0.2** avec Vicuna-13B sur les mêmes 5 tâches × 11 pnums.
Analyse quelle architecture est la plus sensible aux stimuli négatifs et laquelle en bénéficie le plus.

| Modèle | Paramètres | Accès | Template |
|--------|-----------|-------|---------|
| `lmsys/vicuna-13b-v1.5` (Partie 1) | 13B | libre | `USER: {q}\nASSISTANT:` |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | libre | `<s>[INST] {q} [/INST]` |

**Script principal** : `part2_improvements/run_multimodel.py`

```python
# Restart & Clear Output AVANT de lancer pour libérer la GPU RAM

import subprocess, os
REPO = "/kaggle/working/negativePrompts"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-b", "branche_chen",
                    "https://github.com/ac2408/negativePrompts.git", REPO], check=True)

# T4 16GB (quantize obligatoire)
%run /kaggle/working/negativePrompts/part2_improvements/run_multimodel.py --quantize --batch_size 4
```

**Sorties** :
- `results/protocol_mistral.csv` — 55 lignes (5 tâches × 11 pnums), même format que `protocol_vicuna.csv`
- `results/comparison_vicuna_mistral.txt` — tableau comparatif baseline/best/delta + sensibilité aux NP
