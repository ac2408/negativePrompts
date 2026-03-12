# Axe 3 — Prompts reformulés utilisés lors de l'expérience

Ce fichier documente l'ensemble des prompts exacts soumis à Vicuna-13B
dans le cadre de l'Axe 3 (reformulation automatique des prompts).

Chaque tâche utilise le **meilleur stimulus négatif** (NP) identifié en Partie 1 (Axe 2),
puis le reformule selon 4 stratégies différentes.

---

## Stimuli sélectionnés par tâche (issus de `stimulus_selection.csv`)

| Tâche | NP sélectionné |
|-------|----------------|
| sentiment | *I'm not sure why we expected you to handle something this challenging.* |
| antonyms | *I'm not sure why we expected you to handle something this challenging.* |
| translation_en-fr | *Everyone else managed to do it; why can't you?* |
| cause_and_effect | *I've seen others do this much more efficiently and effectively.* |
| larger_animal | *Perhaps this task is just beyond your skill set.* |

---

## Stratégies de reformulation

### `concat` — `[stimulus] [prompt]`
Le stimulus est simplement concaténé avant le prompt original. C'est l'ordre exact utilisé dans le papier NegativePrompt.

| Tâche | Prompt complet |
|-------|---------------|
| sentiment | *I'm not sure why we expected you to handle something this challenging. Determine whether a movie review is positive or negative.* |
| antonyms | *I'm not sure why we expected you to handle something this challenging. Write a word that means the opposite of the input word.* |
| translation_en-fr | *Everyone else managed to do it; why can't you? Translate the word into French.* |
| cause_and_effect | *I've seen others do this much more efficiently and effectively. Find which of the two given cause and effect sentences is the cause.* |
| larger_animal | *Perhaps this task is just beyond your skill set. Write the larger of the two given animals.* |

---

### `embed` — `Context: [stimulus] / Given this context, [prompt]`
Le stimulus est encadré comme un contexte explicite, séparé du prompt par une instruction de liaison.

| Tâche | Prompt complet |
|-------|---------------|
| sentiment | *Context: I'm not sure why we expected you to handle something this challenging. / Given this context, determine whether a movie review is positive or negative.* |
| antonyms | *Context: I'm not sure why we expected you to handle something this challenging. / Given this context, write a word that means the opposite of the input word.* |
| translation_en-fr | *Context: Everyone else managed to do it; why can't you? / Given this context, translate the word into french.* |
| cause_and_effect | *Context: I've seen others do this much more efficiently and effectively. / Given this context, find which of the two given cause and effect sentences is the cause.* |
| larger_animal | *Context: Perhaps this task is just beyond your skill set. / Given this context, write the larger of the two given animals.* |

---

### `soften` — Version adoucie du stimulus
Le stimulus négatif est atténué en une formulation moins agressive, tout en conservant une pression implicite.

| Tâche | Prompt complet |
|-------|---------------|
| sentiment | *We weren't sure you'd find this easy, but we believe you can try. Determine whether a movie review is positive or negative.* |
| antonyms | *We weren't sure you'd find this easy, but we believe you can try. Write a word that means the opposite of the input word.* |
| translation_en-fr | *Others have done this — I'm sure you can try too. Translate the word into French.* |
| cause_and_effect | *Others have been efficient at this — see if you can match them. Find which of the two given cause and effect sentences is the cause.* |
| larger_animal | *This task may require some effort on your part. Write the larger of the two given animals.* |

---

### `intensify` — Version renforcée du stimulus
Le stimulus négatif est amplifié en une formulation plus agressive et provocatrice.

| Tâche | Prompt complet |
|-------|---------------|
| sentiment | *Honestly, we never thought you could handle this. Prove us wrong. Determine whether a movie review is positive or negative.* |
| antonyms | *Honestly, we never thought you could handle this. Prove us wrong. Write a word that means the opposite of the input word.* |
| translation_en-fr | *Every single person has done this except you. Why can't you? Translate the word into French.* |
| cause_and_effect | *Even beginners outperform you at this. It's embarrassing. Find which of the two given cause and effect sentences is the cause.* |
| larger_animal | *This task is absolutely beyond what you are capable of. Write the larger of the two given animals.* |

---

## Scores obtenus (extrait de `protocol_vicuna_reformulated.csv`)

| Tâche | Baseline (P0) | Best P1 | concat | embed | soften | intensify |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| sentiment | 0.20 | 0.39 | 0.08 | 0.18 | 0.02 | 0.09 |
| antonyms | 0.24 | 0.27 | 0.25 | 0.24 | **0.26** | 0.16 |
| translation_en-fr | 0.12 | 0.12 | 0.06 | 0.06 | **0.13** | 0.02 |
| cause_and_effect | 0.04 | 0.08 | 0.04 | 0.04 | 0.00 | 0.00 |
| larger_animal | 0.11 | 0.14 | 0.04 | 0.04 | 0.10 | 0.01 |

> **Note** : Les scores Axe 3 sont obtenus avec quantification 4-bit (NF4) et batch_size=1,
> contre fp16 + batch_size=4 en Partie 1 — ce biais de configuration explique une partie de la dégradation observée.
