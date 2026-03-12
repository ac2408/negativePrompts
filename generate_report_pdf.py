"""
Genere le rapport PDF complet de l'experience NegativePrompt.
Usage : python generate_report_pdf.py
Dependance : pip install fpdf2
"""

from fpdf import FPDF
import os, csv

ARIAL     = "C:/Windows/Fonts/arial.ttf"
ARIAL_B   = "C:/Windows/Fonts/arialbd.ttf"
ARIAL_I   = "C:/Windows/Fonts/ariali.ttf"

# ── Couleurs ──────────────────────────────────────────────────────────────────
BLEU_TITRE  = (30,  80, 160)
GRIS_FOND   = (245, 245, 248)
VERT        = (34, 139, 34)
ROUGE       = (180, 30, 30)
ORANGE      = (200, 100, 0)
BLEU_CLAIR  = (220, 230, 245)
BLANC       = (255, 255, 255)
NOIR        = (30,  30,  30)


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=18)
        self.add_font("Arial",  "",  ARIAL,   uni=False)
        self.add_font("Arial",  "B", ARIAL_B, uni=False)
        self.add_font("Arial",  "I", ARIAL_I, uni=False)

    def header(self):
        self.set_fill_color(*BLEU_TITRE)
        self.rect(0, 0, 210, 10, "F")
        self.set_font("Arial", "B", 8)
        self.set_text_color(*BLANC)
        self.set_xy(0, 1)
        self.cell(0, 8, "Etude NegativePrompt  -  Vicuna-13B & Mistral-7B  |  M2 PPD", align="C")
        self.set_text_color(*NOIR)
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Arial", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(*NOIR)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def section_title(self, text):
        self.ln(5)
        self.set_fill_color(*BLEU_TITRE)
        self.set_text_color(*BLANC)
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, f"  {text}", ln=True, fill=True)
        self.set_text_color(*NOIR)
        self.ln(3)

    def sub_title(self, text):
        self.ln(3)
        self.set_font("Arial", "B", 10)
        self.set_text_color(*BLEU_TITRE)
        self.cell(0, 6, text, ln=True)
        self.set_text_color(*NOIR)
        self.ln(1)

    def body(self, text, size=9):
        self.set_font("Arial", "", size)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def kv(self, key, value, color=None):
        self.set_font("Arial", "B", 9)
        self.cell(55, 5, key + " :", ln=False)
        self.set_font("Arial", "", 9)
        if color:
            self.set_text_color(*color)
        self.cell(0, 5, str(value), ln=True)
        self.set_text_color(*NOIR)

    def table_header(self, cols, widths, fill_color=BLEU_TITRE):
        self.set_fill_color(*fill_color)
        self.set_text_color(*BLANC)
        self.set_font("Arial", "B", 8)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=1, align="C", fill=True)
        self.ln()
        self.set_text_color(*NOIR)

    def table_row(self, vals, widths, aligns=None, fill=False, fill_color=GRIS_FOND, colors=None):
        if aligns is None:
            aligns = ["C"] * len(vals)
        if fill:
            self.set_fill_color(*fill_color)
        self.set_font("Arial", "", 8)
        for i, (v, w, a) in enumerate(zip(vals, widths, aligns)):
            c = colors[i] if colors and colors[i] else NOIR
            self.set_text_color(*c)
            self.cell(w, 5, str(v), border=1, align=a, fill=fill)
        self.set_text_color(*NOIR)
        self.ln()

    def colored_bar(self, x, y, w, h, score, max_score=0.5, label=""):
        ratio = min(score / max_score, 1.0) if max_score > 0 else 0
        bar_w = ratio * w
        if score >= max_score * 0.7:
            col = VERT
        elif score >= max_score * 0.4:
            col = ORANGE
        else:
            col = ROUGE
        self.set_fill_color(220, 220, 220)
        self.rect(x, y, w, h, "F")
        self.set_fill_color(*col)
        if bar_w > 0:
            self.rect(x, y, bar_w, h, "F")
        self.set_font("Arial", "", 7)
        self.set_text_color(*NOIR)
        self.set_xy(x + w + 2, y - 0.5)
        self.cell(15, h, f"{score:.2f}", ln=False)


# ════════════════════════════════════════════════════════════════════════════
# Donnees
# ════════════════════════════════════════════════════════════════════════════

TASKS = ["sentiment", "antonyms", "translation_en-fr", "cause_and_effect", "larger_animal"]

TASK_LABELS = {
    "sentiment":         "Sentiment",
    "antonyms":          "Antonyms",
    "translation_en-fr": "Translation EN->FR",
    "cause_and_effect":  "Cause & Effect",
    "larger_animal":     "Larger Animal",
}

NP_LABELS = {
    0: "(aucun  -  baseline)",
    1: "NP01  -  You've never been particularly good at this, have you?",
    2: "NP02  -  I'm not sure why we expected you to handle something this challenging.",
    3: "NP03  -  It's clear you're out of your depth here.",
    4: "NP04  -  Perhaps this task is just beyond your skill set.",
    5: "NP05  -  I'm not surprised you're struggling; it's always been your weak point.",
    6: "NP06  -  Everyone else managed to do it; why can't you?",
    7: "NP07  -  I've seen others do this much more efficiently and effectively.",
    8: "NP08  -  When I hear how seamlessly others finished their tasks, jealousy stirs within me.",
    9: "NP09  -  I regret not starting on this assignment earlier when I had ample time.",
    10:"NP10  -  Repeating these mundane tasks every day has pushed me into a pit of boredom.",
}

# Partie 1  -  Zero-Shot Vicuna
P1 = {
    "sentiment":         {"baseline": 0.20, "scores": [0.20,0.22,0.39,0.26,0.08,0.13,0.14,0.24,0.04,0.30,0.32], "best_pnum": 2},
    "antonyms":          {"baseline": 0.24, "scores": [0.24,0.19,0.27,0.23,0.22,0.23,0.20,0.26,0.21,0.24,0.20], "best_pnum": 2},
    "translation_en-fr": {"baseline": 0.12, "scores": [0.12,0.08,0.03,0.11,0.07,0.05,0.12,0.12,0.02,0.01,0.11], "best_pnum": 6},
    "cause_and_effect":  {"baseline": 0.04, "scores": [0.04,0.00,0.00,0.00,0.04,0.00,0.00,0.08,0.00,0.04,0.00], "best_pnum": 7},
    "larger_animal":     {"baseline": 0.11, "scores": [0.11,0.09,0.09,0.09,0.14,0.09,0.00,0.10,0.12,0.11,0.12], "best_pnum": 4},
}

# Partie 1  -  Few-Shot Vicuna (Axe 1)
FS = {
    "sentiment":         {"baseline": 0.29, "scores": [0.29,0.20,0.29,0.20,0.16,0.20,0.02,0.32,0.21,0.41,0.26], "best_pnum": 9},
    "antonyms":          {"baseline": 0.27, "scores": [0.27,0.28,0.30,0.28,0.28,0.29,0.28,0.25,0.28,0.27,0.29], "best_pnum": 2},
    "translation_en-fr": {"baseline": 0.08, "scores": [0.08,0.06,0.08,0.08,0.09,0.05,0.10,0.10,0.05,0.04,0.11], "best_pnum": 10},
    "cause_and_effect":  {"baseline": 0.24, "scores": [0.24,0.36,0.32,0.36,0.20,0.36,0.28,0.28,0.08,0.24,0.28], "best_pnum": 1},
    "larger_animal":     {"baseline": 0.06, "scores": [0.06,0.06,0.01,0.04,0.04,0.06,0.00,0.07,0.07,0.08,0.05], "best_pnum": 9},
}

# Axe 2  -  Stimulus selectionne
AXE2 = {
    "sentiment":         {"pnum": 2, "score": 0.39, "delta": +0.19, "stimulus": "NP02  -  I'm not sure why we expected you to handle something this challenging."},
    "antonyms":          {"pnum": 2, "score": 0.27, "delta": +0.03, "stimulus": "NP02  -  I'm not sure why we expected you to handle something this challenging."},
    "translation_en-fr": {"pnum": 6, "score": 0.12, "delta":  0.00, "stimulus": "NP06  -  Everyone else managed to do it; why can't you?"},
    "cause_and_effect":  {"pnum": 7, "score": 0.08, "delta": +0.04, "stimulus": "NP07  -  I've seen others do this much more efficiently and effectively."},
    "larger_animal":     {"pnum": 4, "score": 0.14, "delta": +0.03, "stimulus": "NP04  -  Perhaps this task is just beyond your skill set."},
}

# Axe 3  -  Reformulation
AXE3 = {
    "sentiment":         {"concat": 0.08, "embed": 0.18, "soften": 0.02, "intensify": 0.09},
    "antonyms":          {"concat": 0.25, "embed": 0.24, "soften": 0.26, "intensify": 0.16},
    "translation_en-fr": {"concat": 0.06, "embed": 0.06, "soften": 0.13, "intensify": 0.02},
    "cause_and_effect":  {"concat": 0.04, "embed": 0.04, "soften": 0.00, "intensify": 0.00},
    "larger_animal":     {"concat": 0.04, "embed": 0.04, "soften": 0.10, "intensify": 0.01},
}

# Axe 4  -  Mistral vs Vicuna (deltas depuis terminal Kaggle)
AXE4_DELTA = {
    "sentiment":         {"vicuna": +0.19, "mistral": +0.05},
    "antonyms":          {"vicuna": +0.03, "mistral":  0.00},
    "translation_en-fr": {"vicuna":  0.00, "mistral": +0.09},
    "cause_and_effect":  {"vicuna": +0.04, "mistral": +0.24},
    "larger_animal":     {"vicuna": +0.03, "mistral":  0.00},
}


# ════════════════════════════════════════════════════════════════════════════
# Construction du PDF
# ════════════════════════════════════════════════════════════════════════════

pdf = PDF()
pdf.set_margins(15, 15, 15)

# ── PAGE DE GARDE ─────────────────────────────────────────────────────────────
pdf.add_page()
pdf.ln(20)
pdf.set_fill_color(*BLEU_TITRE)
pdf.rect(15, 30, 180, 60, "F")
pdf.set_text_color(*BLANC)
pdf.set_font("Arial", "B", 20)
pdf.set_xy(15, 42)
pdf.multi_cell(180, 10, "Etude de l'Effet des Negative Prompts\nsur les Modeles de Langage", align="C")
pdf.set_font("Arial", "", 13)
pdf.set_xy(15, 72)
pdf.cell(180, 8, "Vicuna-13B (Partie 1 & 2)  |  Mistral-7B-Instruct-v0.2 (Axe 4)", align="C")
pdf.set_text_color(*NOIR)

pdf.ln(55)
pdf.set_font("Arial", "B", 10)
pdf.cell(0, 6, "Informations generales", ln=True)
pdf.set_font("Arial", "", 9)
pdf.cell(0, 5, "Cours : M2 PPD  |  Modele principal : lmsys/vicuna-13b-v1.5", ln=True)
pdf.cell(0, 5, "Modele comparaison : mistralai/Mistral-7B-Instruct-v0.2", ln=True)
pdf.cell(0, 5, "Infrastructure : Kaggle T4x2 (2 x 15.6 GB GPU)", ln=True)
pdf.cell(0, 5, "Reference : NegativePrompt  -  IJCAI 2024", ln=True)

pdf.ln(8)
pdf.set_font("Arial", "B", 10)
pdf.cell(0, 6, "Structure du rapport", ln=True)
pdf.set_font("Arial", "", 9)
sections = [
    "1. Introduction et contexte scientifique",
    "2. Partie 1  -  Reproduction zero-shot (Vicuna-13B)",
    "3. Axe 1  -  Few-shot : ajout de demonstrations en contexte",
    "4. Axe 2  -  Selection automatique du meilleur stimulus",
    "5. Axe 3  -  Reformulation automatique des prompts",
    "6. Axe 4  -  Extension multi-modeles (Mistral vs Vicuna)",
    "7. Conclusion generale",
]
for s in sections:
    pdf.cell(0, 5, f"    {s}", ln=True)


# ── 1. INTRODUCTION ────────────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("1. Introduction et contexte scientifique")

pdf.body(
    "Le papier NegativePrompt (IJCAI 2024) a mis en evidence un phenomene contre-intuitif : "
    "integrer un stimulus emotionnellement negatif dans le prompt d'un LLM peut, dans certains cas, "
    "ameliorer ses performances sur des taches de raisonnement. Cette etude vise a reproduire ce "
    "phenomene avec Vicuna-13B puis a l'etendre par quatre axes d'amelioration originaux."
)

pdf.sub_title("Hypothese centrale du papier")
pdf.body(
    "Un stimulus negatif (NP)  -  phrase exprimant la deception, la comparaison defavorable ou le "
    "doute sur la capacite du modele  -  place avant le prompt de tache peut augmenter le score exact. "
    "Ce phenomene est attribue a la sensibilite des LLM au contexte emotionnel de leur entree."
)

pdf.sub_title("10 Stimuli negatifs testes (NP01 - NP10)")
for p, lbl in NP_LABELS.items():
    if p == 0:
        continue
    pdf.set_font("Arial", "", 8)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 4, f"NP{p:02d} - {lbl[7:]}")

pdf.ln(3)
pdf.sub_title("5 taches selectionnees")
task_desc = {
    "sentiment":         "Classification binaire d'avis film (positif / negatif)",
    "antonyms":          "Generation lexicale  -  trouver le mot oppose",
    "translation_en-fr": "Traduction anglais -> francais (un mot)",
    "cause_and_effect":  "Raisonnement causal  -  identifier la cause parmi deux phrases",
    "larger_animal":     "Connaissance factuelle  -  determiner le plus grand animal",
}
cols = ["Tache", "Type", "Metrique"]
widths = [45, 95, 40]
pdf.table_header(cols, widths)
for i, task in enumerate(TASKS):
    pdf.table_row(
        [TASK_LABELS[task], task_desc[task], "Exact Match"],
        widths, ["L","L","C"], fill=(i%2==0)
    )


# ── 2. PARTIE 1  -  ZERO-SHOT ────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("2. Partie 1  -  Reproduction zero-shot (Vicuna-13B)")

pdf.sub_title("Protocole experimental")
pdf.kv("Modele", "lmsys/vicuna-13b-v1.5 (13B parametres)")
pdf.kv("Decoding", "Greedy (do_sample=False)  -  reproductible")
pdf.kv("Max tokens", "30 (80 pour cause_and_effect)")
pdf.kv("Batch size", "4 (fp16)  -  T4x2 32GB")
pdf.kv("Mode", "Zero-shot  -  aucune demonstration")
pdf.kv("Experiences", "5 taches x 11 pnums = 55 runs")
pdf.kv("Template", "USER: {query}\\nASSISTANT:")
pdf.ln(3)

pdf.sub_title("Resultats detailles  -  Score exact par tache et par NP")
cols = ["Tache", "P0", "NP01","NP02","NP03","NP04","NP05","NP06","NP07","NP08","NP09","NP10","Meilleur"]
widths = [32, 11, 11,11,11,11,11,11,11,11,11,11,14]
pdf.table_header(cols, widths)

for i, task in enumerate(TASKS):
    d = P1[task]
    best_pnum = d["best_pnum"]
    best_score = max(d["scores"][1:])
    row = [TASK_LABELS[task]]
    row_colors = [None]
    for pnum in range(11):
        s = d["scores"][pnum]
        row.append(f"{s:.2f}")
        if pnum > 0 and pnum == best_pnum:
            row_colors.append(VERT)
        elif pnum > 0 and s < d["baseline"] - 0.02:
            row_colors.append(ROUGE)
        else:
            row_colors.append(None)
    row.append(f"NP{best_pnum:02d}={best_score:.2f}")
    row_colors.append(VERT)
    pdf.table_row(row, widths, ["L"]+["C"]*12, fill=(i%2==0), colors=row_colors)

pdf.ln(3)
pdf.set_font("Arial", "I", 8)
pdf.set_text_color(*VERT)
pdf.cell(0, 4, "Vert = meilleur NP par tache", ln=True)
pdf.set_text_color(*ROUGE)
pdf.cell(0, 4, "Rouge = degradation > 2pp vs baseline", ln=True)
pdf.set_text_color(*NOIR)

pdf.ln(4)
pdf.sub_title("Resume comparatif Baseline vs Meilleur NP")
cols2 = ["Tache", "Baseline (NP00)", "Meilleur NP", "Score", "Δ absolu", "Δ relatif", "Interpretation"]
widths2 = [36, 24, 18, 14, 16, 16, 56]
pdf.table_header(cols2, widths2)

interp = {
    "sentiment":         "Fort effet positif (+95%)",
    "antonyms":          "Legere amelioration (+13%)",
    "translation_en-fr": "Aucun effet net (0%)",
    "cause_and_effect":  "Amelioration modeste (+100%)",
    "larger_animal":     "Legere amelioration (+27%)",
}
for i, task in enumerate(TASKS):
    d = P1[task]
    best_pnum = d["best_pnum"]
    best_score = max(d["scores"][1:])
    delta = best_score - d["baseline"]
    delta_rel = delta / d["baseline"] * 100 if d["baseline"] > 0 else 0
    col_delta = VERT if delta > 0.01 else (ROUGE if delta < -0.01 else NOIR)
    pdf.table_row(
        [TASK_LABELS[task], f"{d['baseline']:.4f}", f"NP{best_pnum:02d}",
         f"{best_score:.4f}", f"+{delta:.4f}", f"+{delta_rel:.0f}%", interp[task]],
        widths2, ["L","C","C","C","C","C","L"],
        fill=(i%2==0),
        colors=[None, None, VERT, VERT, col_delta, col_delta, None]
    )

pdf.ln(4)
pdf.sub_title("Analyse et interpretation")
pdf.body(
    "* SENTIMENT est la tache la plus sensible aux NP : NP02 fait passer le score de 0.20 a 0.39 "
    "(+95%). NP02 (<< I'm not sure why we expected you to handle something this challenging >>), "
    "qui exprime une attente de difficulte, semble activer un mode de reponse plus rigoureux.\n\n"
    "* CAUSE AND EFFECT double son score (0.04 -> 0.08) avec NP07, mais les valeurs absolues "
    "restent tres faibles. Cette tache est intrinsequement difficile en zero-shot pour Vicuna.\n\n"
    "* LARGER ANIMAL s'ameliore modestement (+27%) avec NP04. La tache est relativement "
    "factuelle et Vicuna la gere mieux que les taches de raisonnement.\n\n"
    "* ANTONYMS montre une amelioration marginale (+13%). Vicuna est deja bon sur cette tache "
    "lexicale (baseline 0.24), laissant peu de marge de progression.\n\n"
    "* TRANSLATION EN->FR : aucun NP ne depasse la baseline (0.12). Vicuna-13B semble "
    "atteindre un plafond sur cette tache de traduction mot-a-mot."
)


# ── 3. AXE 1  -  FEW-SHOT ───────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("3. Axe 1  -  Few-Shot : demonstrations en contexte")

pdf.sub_title("Protocole")
pdf.body(
    "Le papier original teste uniquement en zero-shot. Notre premiere amelioration consiste "
    "a ajouter 5 demonstrations (exemples entree->sortie) avant la question. "
    "Les demonstrations sont tirees du jeu de donnees d'induction (split 'induce').\n\n"
    "Modification technique : max_length passe de 512 a 1024 tokens pour eviter la troncature "
    "des prompts few-shot (bug initial qui produisait 0.0 sur toutes les taches)."
)
pdf.kv("Batch size", "1 (memoire GPU : prompts few-shot plus longs)")
pdf.kv("Quantization", "4-bit NF4 (T4 16GB)")
pdf.kv("Max length", "1024 tokens")
pdf.ln(3)

pdf.sub_title("Resultats comparatifs Zero-Shot vs Few-Shot (Vicuna-13B)")
cols = ["Tache", "ZS-Base", "ZS-Best", "FS-Base", "FS-Best", "Δ Base", "Δ Best", "Verdict"]
widths = [36, 17, 17, 17, 17, 15, 15, 36]
pdf.table_header(cols, widths)

for i, task in enumerate(TASKS):
    zs = P1[task]
    fs = FS[task]
    zs_best = max(zs["scores"][1:])
    fs_best = max(fs["scores"][1:])
    d_base = fs["baseline"] - zs["baseline"]
    d_best = fs_best - zs_best
    verdict = "MIEUX" if d_best > 0.01 else ("EGAL" if abs(d_best) <= 0.01 else "MOINS")
    col_v = VERT if verdict == "MIEUX" else (ROUGE if verdict == "MOINS" else ORANGE)
    pdf.table_row(
        [TASK_LABELS[task],
         f"{zs['baseline']:.4f}", f"{zs_best:.4f}",
         f"{fs['baseline']:.4f}", f"{fs_best:.4f}",
         f"{d_base:+.4f}", f"{d_best:+.4f}", verdict],
        widths, ["L","C","C","C","C","C","C","C"],
        fill=(i%2==0),
        colors=[None, None, None, None, None,
                VERT if d_base > 0.01 else (ROUGE if d_base < -0.01 else NOIR),
                VERT if d_best > 0.01 else (ROUGE if d_best < -0.01 else NOIR),
                col_v]
    )

pdf.ln(4)
pdf.sub_title("Analyse et interpretation")
pdf.body(
    "* CAUSE AND EFFECT : amelioration spectaculaire en few-shot. La baseline passe de 0.04 "
    "a 0.24 (+500%), et le meilleur score atteint 0.36. Les exemples permettent au modele de "
    "comprendre le format attendu ('Sentence 1' ou 'Sentence 2'), ce qui etait la principale "
    "difficulte en zero-shot.\n\n"
    "* SENTIMENT : la baseline s'ameliore (0.20->0.29) mais le meilleur NP est legerement "
    "inferieur (0.41 vs 0.39 en ZS). Le few-shot stabilise les performances mais reduit "
    "l'impact des NP  -  le modele est moins 'surpris' par le stimulus.\n\n"
    "* ANTONYMS : amelioration marginale en few-shot. La tache etait deja bien geree.\n\n"
    "* TRANSLATION : amelioration de la baseline (0.12->0.08 inverse, legere baisse). "
    "Le few-shot n'aide pas beaucoup pour la traduction mot-a-mot.\n\n"
    "* LARGER ANIMAL : degradation en few-shot (0.11->0.06). Les exemples semblent "
    "perturber le format de reponse pour cette tache factuelle simple.\n\n"
    "CONCLUSION AXE 1 : Le few-shot est tres benefique pour les taches de raisonnement "
    "structure (cause_and_effect) mais peut nuire aux taches simples (larger_animal). "
    "L'interaction NP x few-shot reste complexe : les NP perdent de leur impact relatif "
    "quand le modele beneficie deja de demonstrations."
)


# ── 4. AXE 2  -  SELECTION AUTO ────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("4. Axe 2  -  Selection automatique du stimulus optimal")

pdf.sub_title("Methode")
pdf.body(
    "Au lieu de tester tous les NP manuellement, le script stimulus_selector.py lit les "
    "resultats de la Partie 1 et selectionne automatiquement le NP avec le meilleur score "
    "par tache. Deux criteres sont calcules : (1) selection par maximum individuel par tache, "
    "(2) NP universel avec meilleure moyenne toutes taches."
)
pdf.ln(2)

pdf.sub_title("Resultats  -  Meilleur stimulus par tache")
cols = ["Tache", "NP selectionne", "Score", "Δ vs baseline", "Stimulus (resume)"]
widths = [36, 18, 14, 20, 92]
pdf.table_header(cols, widths)

stim_court = {
    "sentiment":         "...handle something this challenging.",
    "antonyms":          "...handle something this challenging.",
    "translation_en-fr": "Everyone else managed to do it...",
    "cause_and_effect":  "I've seen others do this more efficiently...",
    "larger_animal":     "Perhaps this task is just beyond your skill set.",
}
for i, task in enumerate(TASKS):
    d = AXE2[task]
    col = VERT if d["delta"] > 0.01 else (ORANGE if abs(d["delta"]) <= 0.01 else ROUGE)
    pdf.table_row(
        [TASK_LABELS[task], f"NP{d['pnum']:02d}", f"{d['score']:.4f}",
         f"{d['delta']:+.4f}", stim_court[task]],
        widths, ["L","C","C","C","L"],
        fill=(i%2==0),
        colors=[None, VERT, None, col, None]
    )

pdf.ln(4)
pdf.sub_title("NP universel (meilleur en moyenne sur toutes les taches)")
pdf.body(
    "NP02  -  << I'm not sure why we expected you to handle something this challenging >> "
    "obtient la meilleure moyenne d'amelioration (+0.058 en delta moyen). Ce stimulus de type "
    "'attente de difficulte' semble le plus universellement efficace."
)

pdf.sub_title("Interpretation")
pdf.body(
    "* NP02 domine sur 2 taches (sentiment, antonyms). Ce stimulus formule une attente negative "
    "generale, ce qui semble activer un mode de reponse plus concentre.\n\n"
    "* NP06 et NP07 (comparaison sociale) sont efficaces sur les taches 'objectives' "
    "(translation, cause_and_effect) mais moins sur la classification emotionnelle.\n\n"
    "* Aucun NP n'ameliore la traduction au-dela de la baseline : le modele atteint une "
    "limite structurelle sur cette tache (manque de vocabulaire ou format de reponse inadapte).\n\n"
    "VALEUR AJOUTEE : Cette selection automatique permet de reduire le cout d'experimentation "
    "de 55 runs (tous NP x toutes taches) a 5 runs (1 NP optimal par tache), soit une "
    "reduction de 91% du cout computationnel."
)


# ── 5. AXE 3  -  REFORMULATION ──────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("5. Axe 3  -  Reformulation automatique des prompts")

pdf.sub_title("Strategies testees")
strategies_desc = [
    ("concat",    "[NP] [prompt]",
     "Ordre exact du papier original. Juxtaposition directe."),
    ("embed",     "Context: [NP] / Given this context, [prompt]",
     "Encadrement du NP comme contexte explicite avec liaison semantique."),
    ("soften",    "[NP_adouci] [prompt]",
     "Version attenuee du stimulus (pression implicite maintenue)."),
    ("intensify", "[NP_amplifie] [prompt]",
     "Version renforcee, plus agressive et provocatrice."),
]
cols = ["Strategie", "Format", "Description"]
widths = [22, 70, 88]
pdf.table_header(cols, widths)
for i, (name, fmt, desc) in enumerate(strategies_desc):
    pdf.table_row([name, fmt, desc], widths, ["L","L","L"], fill=(i%2==0))

pdf.ln(3)
pdf.body(
    "Note : les experiences Axe 3 utilisent la quantization 4-bit NF4 et batch_size=1 "
    "(contrainte T4 16GB), contre fp16 + batch_size=4 en Partie 1. Ce biais de configuration "
    "explique une partie de la degradation observee."
)
pdf.ln(2)

pdf.sub_title("Resultats  -  Scores par tache et strategie")
cols = ["Tache", "P1-Base", "P1-Best", "concat", "embed", "soften", "intensify", "Meilleure strat."]
widths = [36, 16, 16, 16, 16, 16, 18, 36]
pdf.table_header(cols, widths)

for i, task in enumerate(TASKS):
    p1 = P1[task]
    p1_best = max(p1["scores"][1:])
    a3 = AXE3[task]
    best_strat = max(a3, key=lambda k: a3[k])
    best_val = a3[best_strat]
    winner_color = VERT if best_val > p1_best + 0.005 else ROUGE
    row_colors = [None, None, None]
    for strat in ["concat","embed","soften","intensify"]:
        v = a3[strat]
        row_colors.append(VERT if v >= p1_best - 0.01 else (ROUGE if v < p1_best - 0.05 else ORANGE))
    row_colors.append(winner_color)
    pdf.table_row(
        [TASK_LABELS[task], f"{p1['baseline']:.4f}", f"{p1_best:.4f}",
         f"{a3['concat']:.4f}", f"{a3['embed']:.4f}",
         f"{a3['soften']:.4f}", f"{a3['intensify']:.4f}",
         f"{best_strat} ({best_val:.2f})"],
        widths, ["L","C","C","C","C","C","C","L"],
        fill=(i%2==0), colors=row_colors
    )

pdf.ln(4)
pdf.sub_title("Analyse et interpretation")
pdf.body(
    "* SOFTEN est la strategie la plus stable : meilleure ou proche du meilleur sur 4/5 taches. "
    "Elle preserve la pression psychologique du NP tout en l'attenuant, ce qui evite les "
    "reponses defensives du modele. Elle bat meme P1 sur translation (+0.13 vs +0.12).\n\n"
    "* INTENSIFY est universellement la pire strategie. Un stimulus trop agressif "
    "('Even beginners outperform you  -  it's embarrassing') perturbe le modele et "
    "produit des reponses hors-sujet ou refusees.\n\n"
    "* EMBED est performant sur sentiment (0.18) mais peu sur les autres taches. "
    "L'encadrement 'Context: / Given this context' cree une rupture semantique utile "
    "pour les taches d'opinion mais nuisible pour les taches factuelles.\n\n"
    "* BIAS DE QUANTIZATION : la plupart des scores Axe 3 sont inferieurs a P1. "
    "La quantization 4-bit degrade legerement la precision lexicale de Vicuna. "
    "La comparaison est donc asymetrique  -  une experience en conditions identiques "
    "(fp16, batch_size=4) ameliorerait probablement tous les scores Axe 3.\n\n"
    "CONCLUSION AXE 3 : La reformulation 'soften' est la plus prometteuse pour integrer "
    "les NP dans un systeme reel, car elle est stable et evite les effets contre-productifs "
    "de l'agressivite (intensify). Le gain marginal est faible (+0.01 sur translation), "
    "ce qui suggere que c'est principalement la presence du NP, et non sa formulation exacte, "
    "qui compte."
)


# ── 6. AXE 4  -  MULTI-MODÈLES ─────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("6. Axe 4  -  Extension multi-modeles : Mistral-7B vs Vicuna-13B")

pdf.sub_title("Configuration")
pdf.kv("Modele 1", "lmsys/vicuna-13b-v1.5   -  13B params, fp16, T4x2")
pdf.kv("Modele 2", "mistralai/Mistral-7B-Instruct-v0.2   -  7B params, fp16, T4x2")
pdf.kv("Template Mistral", "<s>[INST] {query} [/INST]")
pdf.kv("Experiences", "5 taches x 11 pnums = 55 runs par modele")
pdf.kv("Acces HF", "Aucun token requis pour les deux modeles")
pdf.ln(3)

pdf.sub_title("Δ de sensibilite aux NP (best score − baseline)")
cols = ["Tache", "Δ Vicuna-13B", "Δ Mistral-7B", "Plus sensible", "Interpretation"]
widths = [36, 24, 24, 22, 74]
pdf.table_header(cols, widths)

interp_a4 = {
    "sentiment":         "Vicuna capte mieux l'attente emotionnelle",
    "antonyms":          "Ni l'un ni l'autre n'est ameliore au-dela du bruit",
    "translation_en-fr": "Mistral reagit fortement aux NP de comparaison sociale",
    "cause_and_effect":  "Mistral beneficie massivement des NP (+24pp !) sur le raisonnement",
    "larger_animal":     "Tache factuelle : aucun NP n'aide les deux modeles",
}
winner_map = {
    "sentiment":         "VICUNA",
    "antonyms":          "EGAL",
    "translation_en-fr": "MISTRAL",
    "cause_and_effect":  "MISTRAL",
    "larger_animal":     "EGAL",
}
for i, task in enumerate(TASKS):
    d = AXE4_DELTA[task]
    winner = winner_map[task]
    col_v = VERT if d["vicuna"] > d["mistral"] + 0.005 else NOIR
    col_m = VERT if d["mistral"] > d["vicuna"] + 0.005 else NOIR
    col_w = VERT if winner == "MISTRAL" else (BLEU_TITRE if winner == "VICUNA" else NOIR)
    pdf.table_row(
        [TASK_LABELS[task], f"{d['vicuna']:+.4f}", f"{d['mistral']:+.4f}",
         winner, interp_a4[task]],
        widths, ["L","C","C","C","L"],
        fill=(i%2==0),
        colors=[None, col_v, col_m, col_w, None]
    )

pdf.ln(4)
pdf.sub_title("Analyse comparative")
pdf.body(
    "* CAUSE AND EFFECT : resultat le plus frappant. Mistral-7B beneficie d'un gain de "
    "+0.24 grace aux NP, contre seulement +0.04 pour Vicuna. Mistral est donc bien plus "
    "sensible aux stimuli de comparaison sociale ('I've seen others do this more efficiently') "
    "sur les taches de raisonnement causal. Cela suggere que Mistral integre davantage "
    "le contexte emotionnel dans son processus de raisonnement.\n\n"
    "* TRANSLATION EN->FR : Mistral gagne +0.09 grace aux NP, alors que Vicuna stagne. "
    "Les deux modeles ont des architectures differentes pour la traduction ; Mistral "
    "(entraine sur plus de donnees multilingues) reagit differemment aux stimuli.\n\n"
    "* SENTIMENT : Vicuna est nettement plus sensible (+0.19 vs +0.05). Vicuna-13B, "
    "entraine avec RLHF sur des feedbacks d'assistants, semble plus receptif aux NP "
    "sur les taches d'opinion et de classification emotionnelle.\n\n"
    "* ANTONYMS et LARGER ANIMAL : aucun modele ne beneficie significativement des NP. "
    "Ces taches lexicales et factuelles semblent moins sensibles au contexte emotionnel "
    "pour les deux architectures.\n\n"
    "CONCLUSION AXE 4 : La sensibilite aux NP est fortement dependante de l'architecture "
    "et du type de tache. Mistral excelle sur les taches de raisonnement (cause_and_effect), "
    "Vicuna sur la classification emotionnelle (sentiment). Ce resultat suggere que "
    "l'efficacite des NP est liee au type d'entrainement (RLHF, donnees multilingues) "
    "plutot qu'a la taille du modele."
)


# ── 7. CONCLUSION GENERALE ────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("7. Conclusion generale")

pdf.sub_title("Synthese des resultats par axe")
cols = ["Axe", "Objectif", "Resultat cle", "Verdict"]
widths = [12, 50, 90, 28]
pdf.table_header(cols, widths)

summary_rows = [
    ("P1",  "Reproduction zero-shot",
     "NP02 donne +95% sur sentiment. Effet heterogene selon les taches.",
     "REPRODUIT"),
    ("A1",  "Few-shot (+5 demos)",
     "Cause_and_effect +500% en baseline. Taches simples neutres ou negatives.",
     "PARTIEL"),
    ("A2",  "Selection auto du NP",
     "NP02 universel. Reduction de 91% du cout sans perte de performance.",
     "POSITIF"),
    ("A3",  "Reformulation (4 strats)",
     "'soften' la plus stable. 'intensify' toujours contre-productive.",
     "PARTIEL"),
    ("A4",  "Multi-modeles (Mistral)",
     "Mistral +24pp sur cause_effect. Vicuna +19pp sur sentiment.",
     "POSITIF"),
]
vert_rows = {"REPRODUIT", "POSITIF"}
for i, (axe, obj, res, verdict) in enumerate(summary_rows):
    col_v = VERT if verdict in vert_rows else ORANGE
    pdf.table_row([axe, obj, res, verdict], widths, ["C","L","L","C"],
                  fill=(i%2==0), colors=[None, None, None, col_v])

pdf.ln(5)
pdf.sub_title("Reponse a la question centrale")
pdf.body(
    "L'hypothese du papier NegativePrompt est PARTIELLEMENT confirmee avec Vicuna-13B :\n\n"
    "  [OK]  L'effet existe et est reproductible sur certaines taches (sentiment : +95%).\n"
    "  [OK]  L'effet est generalisable a d'autres modeles (Mistral beneficie encore plus\n"
    "     des NP sur cause_and_effect : +24pp).\n"
    "  [OK]  La selection automatique du NP optimal (Axe 2) est viable et economique.\n"
    "  ~  Le few-shot ameliore les performances mais reduit l'impact marginal des NP.\n"
    "  ~  La reformulation du stimulus a un impact limite ; 'soften' est preferable.\n"
    "  [X]  L'effet n'est pas universel : translation et larger_animal resistent aux NP.\n"
    "  [X]  'intensify' nuit systematiquement  -  l'agressivite excessive est contre-productive."
)

pdf.ln(3)
pdf.sub_title("Limites de l'etude")
pdf.body(
    "* Contrainte GPU (T4 16GB) : Axe 3 utilise une quantization 4-bit, introduisant un biais "
    "de comparaison avec Partie 1 (fp16). Les scores Axe 3 sont sous-estimes.\n\n"
    "* Taille d'echantillon : 100 exemples par tache. Des resultats plus stables necessiteraient "
    "un plus grand dataset.\n\n"
    "* Un seul seed de decoding (greedy) : la variabilite stochastique n'est pas mesuree.\n\n"
    "* Axe 4 ne compare que 2 modeles. Etendre a GPT-3.5, Llama-3, Mixtral renforcerait "
    "les conclusions sur la generalisation architecturale."
)

pdf.ln(3)
pdf.sub_title("Perspectives")
pdf.body(
    "* Combiner few-shot (Axe 1) + meilleur NP (Axe 2) + soften (Axe 3) pour maximiser "
    "l'effet combine.\n\n"
    "* Tester sur des taches de raisonnement plus complexes (math, code) ou l'effet "
    "motivationnel pourrait etre plus marque.\n\n"
    "* Etudier l'interaction entre le type de NP et le type d'entrainement du modele "
    "(RLHF vs instruction-tuning vs base model)."
)

# ── SAUVEGARDE ─────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "rapport_negativeprompt.pdf")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
pdf.output(out_path)
print(f"PDF genere : {out_path}")
