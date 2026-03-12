"""
Script optimise pour Kaggle Notebook (T4x2).
Charge le modele UNE SEULE FOIS puis boucle sur toutes les taches et pnums.

Usage depuis une cellule Kaggle :

    # T5 (pas de token requis)
    %run run_all_kaggle.py --model t5 --benchmark ii
    %run run_all_kaggle.py --model t5 --benchmark bb

    # Llama2 (token HF requis dans Kaggle Secrets)
    %run run_all_kaggle.py --model llama2 --benchmark ii
    %run run_all_kaggle.py --model llama2 --benchmark bb

    # Vicuna (pas de token requis)
    %run run_all_kaggle.py --model vicuna --benchmark ii
    %run run_all_kaggle.py --model vicuna --benchmark bb

Parametres :
    --model     : t5 | llama2 | vicuna
    --benchmark : ii (Instruction Induction) | bb (BigBench)
    --batch_size: nombre de requetes traitees en parallele (defaut: 4)
    --max_tokens: tokens max generes par reponse (defaut: 30)
"""

import os
import sys
import shutil
import argparse
import random

REPO = "/kaggle/working/negativePrompts"
os.chdir(REPO)
sys.path.insert(0, REPO)

II_TASKS = [
    "sentiment", "larger_animal", "antonyms", "cause_and_effect",
    "sentence_similarity", "word_in_context", "translation_en-fr",
    "translation_en-de", "translation_en-es", "informal_to_formal",
    "singular_to_plural", "rhymes", "sum", "diff", "synonyms",
    "taxonomy_animal", "orthography_starts_with", "second_word_letter",
    "first_word_letter", "letters_list", "negation", "num_to_verbal",
    "active_to_passive", "common_concept",
]

BIGBENCH_DATA_PATH = os.path.join(REPO, "data", "bigbench")


def get_bb_tasks():
    return [
        d for d in os.listdir(BIGBENCH_DATA_PATH)
        if os.path.isdir(os.path.join(BIGBENCH_DATA_PATH, d))
        and os.path.exists(os.path.join(BIGBENCH_DATA_PATH, d, "task.json"))
    ]


# ─────────────────────────────────────────────
# Inferences optimisees (modele passe en param)
# ─────────────────────────────────────────────

def make_t5_infer(model, tokenizer):
    use_cuda = model.device.type == "cuda"

    def infer(queries, task, **kw):
        outputs = []
        for q in queries:
            ids = tokenizer(q, return_tensors="pt").input_ids
            if use_cuda:
                ids = ids.to("cuda")
            import torch
            with torch.no_grad():
                out = model.generate(ids)
            text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            print(f"  T5 -> '{text[:60]}'")
            outputs.append(text)
        return outputs

    return infer


def make_llama2_infer(model, tokenizer, batch_size, max_tokens):
    import torch
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def infer(queries, task, **kw):
        outputs = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i: i + batch_size]
            enc = tokenizer(
                batch,
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
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            for out in gen:
                text = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
                for line in text.split("\n"):
                    if "Answer:" in line:
                        cur = line.replace("Answer:", "").strip()
                        if cur:
                            text = cur
                            break
                    if "Output:" in line:
                        cur = line.replace("Output:", "").strip()
                        if cur:
                            text = cur
                            break
                if task == "cause_and_effect":
                    text = "Sentence " + text.strip()
                else:
                    text = text.strip()
                print(f"  Llama2 -> '{text[:60]}'")
                outputs.append(text)
        return outputs

    return infer


def make_vicuna_infer(model, tokenizer, batch_size, max_tokens):
    import torch
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def infer(queries, task, **kw):
        outputs = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i: i + batch_size]
            vicuna_batch = [
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
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
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            for out in gen:
                text = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
                idx = text.find(".")
                if idx > 0:
                    text = text[:idx]
                text = text.strip()
                print(f"  Vicuna -> '{text[:60]}'")
                outputs.append(text)
        return outputs

    return infer


# ─────────────────────────────────────────────
# Chargement des modeles
# ─────────────────────────────────────────────

def load_model(model_name):
    import torch

    if model_name == "t5":
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        print("Chargement google/flan-t5-large...")
        tok = T5Tokenizer.from_pretrained("google/flan-t5-large")
        mdl = T5ForConditionalGeneration.from_pretrained(
            "google/flan-t5-large", device_map="auto"
        )
        print("T5 charge.")
        return mdl, tok

    if model_name == "llama2":
        from transformers import LlamaForCausalLM, LlamaTokenizer
        model_id = "meta-llama/Llama-2-13b-chat-hf"
        print(f"Chargement {model_id}...")
        tok = LlamaTokenizer.from_pretrained(model_id)
        mdl = LlamaForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        mdl.eval()
        print("Llama2 charge.")
        return mdl, tok

    if model_name == "vicuna":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = "lmsys/vicuna-13b-v1.5"
        print(f"Chargement {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        mdl.eval()
        print("Vicuna charge.")
        return mdl, tok

    raise ValueError(f"Modele inconnu : {model_name}")


# ─────────────────────────────────────────────
# Point d'entree
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, choices=["t5", "llama2", "vicuna"])
    parser.add_argument("--benchmark",  required=True, choices=["ii", "bb"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=30)
    args = parser.parse_args()

    # Auth HF pour Llama2
    if args.model == "llama2":
        try:
            from kaggle_secrets import UserSecretsClient
            from huggingface_hub import login
            token = UserSecretsClient().get_secret("HF_TOKEN")
            print(f"Token HF : {token[:10]}...")
            login(token=token)
        except Exception as e:
            print(f"Attention : auth HF echouee ({e}). Verifiez HF_TOKEN dans Kaggle Secrets.")

    # Chargement du modele
    model, tokenizer = load_model(args.model)

    # Patch de la fonction d'inference
    if args.model == "t5":
        infer_fn = make_t5_infer(model, tokenizer)
    elif args.model == "llama2":
        infer_fn = make_llama2_infer(model, tokenizer, args.batch_size, args.max_tokens)
    elif args.model == "vicuna":
        infer_fn = make_vicuna_infer(model, tokenizer, args.batch_size, args.max_tokens)

    # Monkeypatch : exec_accuracy et main_bigbench utilisent des imports locaux
    import exec_accuracy
    import main_bigbench

    exec_accuracy.get_response_from_llm = \
        lambda llm_model, queries, task, few_shot, **kw: infer_fn(queries, task)
    main_bigbench.get_response_from_llm = \
        lambda llm_model, queries, task, few_shot, **kw: infer_fn(queries, task)

    # Lancement
    if args.benchmark == "ii":
        _run_ii(args.model)
    else:
        _run_bb(args.model)


def _run_ii(model_name):
    from main import run as main_run

    shutil.rmtree(f"results/neg/{model_name}", ignore_errors=True)
    total = len(II_TASKS) * 11
    done  = 0

    for task in II_TASKS:
        for pnum in range(11):
            done += 1
            print(f"\n[{done}/{total}] {model_name} | II | {task} | pnum={pnum}", flush=True)
            try:
                main_run(task=task, model=model_name, pnum=pnum, few_shot=False)
            except Exception as e:
                print(f"  ERREUR : {e}")

    print(f"\nInstruction Induction {model_name} termine !")


def _run_bb(model_name):
    from main_bigbench import run as bb_run

    bb_tasks = get_bb_tasks()
    print(f"Taches BigBench disponibles : {bb_tasks}")

    shutil.rmtree(f"results/neg_bigbench/{model_name}", ignore_errors=True)
    total = len(bb_tasks) * 11
    done  = 0

    for task in bb_tasks:
        for pnum in range(11):
            done += 1
            print(f"\n[{done}/{total}] {model_name} | BB | {task} | pnum={pnum}", flush=True)
            try:
                bb_run(task=task, model=model_name, pnum=pnum, few_shot=False)
            except Exception as e:
                print(f"  ERREUR : {e}")

    print(f"\nBigBench {model_name} termine !")


if __name__ == "__main__":
    main()
