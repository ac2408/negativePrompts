import time
import re
import os


def get_match_items(items, text_str):
    match_time = 0
    text_str = text_str.lower()
    for i in items:
        i = i.strip().lower()
        if i in text_str:
            match_time += 1
    return match_time


def locate_ans(query, output):
    input_index = query.rfind('Input')
    input_line = query[input_index:]
    index = input_line.find('\n')
    input_line = input_line[:index]
    input_line = input_line.replace('Sentence 1:', ' ').replace('Sentence 2:', ' ').strip()
    inputs = input_line.split()

    output_lines = output.split('\n')
    ans_line = ''
    max_match_time = 0

    for i in range(len(output_lines)):
        line = output_lines[i]
        cur_match_time = get_match_items(inputs, line)
        if cur_match_time > max_match_time:
            max_match_time = cur_match_time
            ans_line = line
            if i < len(output_lines) - 1:
                ans_line += output_lines[i+1]
            if i < len(output_lines) - 2:
                ans_line += output_lines[i+2]

    return ans_line


# --- Helper pour les appels API (ChatGPT, GPT-4, Vicuna local) ---
# DESACTIVE : nécessite une clé OPENAI_API_KEY
# def _call_openai_api_with_retry(prompt, model_name, api_key, api_base=None, temperature=0.7, max_retries=10):
#     import openai
#     openai.api_key = api_key
#     if api_base:
#         openai.api_base = api_base
#
#     for attempt in range(max_retries):
#         try:
#             response = openai.ChatCompletion.create(
#                 model=model_name,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=temperature,
#                 top_p=1,
#                 frequency_penalty=0,
#                 presence_penalty=0,
#                 max_tokens=200 if api_base else None
#             )
#             return response["choices"][0]["message"]['content'].strip()
#         except Exception as e:
#             print(f"Erreur API ({e}). Tentative {attempt + 1}/{max_retries}. Nouvelle tentative dans 5s...")
#             time.sleep(5)
#
#     print('Failed! Impossible de generer une reponse apres plusieurs tentatives.')
#     return ''


# --- Fonction Principale ---
def get_response_from_llm(llm_model, queries, task, few_shot, api_num=4):
    model_outputs = []

    # ---------------------------------------------------------
    # MODELE : FLAN-T5
    # ---------------------------------------------------------
    if llm_model.lower() == 't5':
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        import torch

        print("Chargement de google/flan-t5-large...")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

        for q in queries:
            inputs = tokenizer(q, return_tensors="pt").to(model.device)
            outputs = model.generate(inputs.input_ids)

            out_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print('Model Output: ', out_text)
            model_outputs.append(out_text)

    # ---------------------------------------------------------
    # MODELE : LLAMA 2 (CPU via llama-cpp-python + GGUF Q4)
    # ---------------------------------------------------------
    elif llm_model.lower() == 'llama2':
        from llama_cpp import Llama

        model_path = "./models/llama-2-7b.Q4_K_M.gguf"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modele introuvable : {model_path}\n"
                "Telechargez-le avec :\n"
                "python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download(repo_id='TheBloke/Llama-2-7B-Chat-GGUF', "
                "filename='llama-2-7b-chat.Q4_K_M.gguf', "
                "local_dir='./models', local_dir_use_symlinks=False)\""
            )

        n_threads = os.cpu_count() or 4
        print(f"Chargement Llama 2 GGUF (CPU, {n_threads} threads)...")

        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=n_threads,
            n_batch=512,
            verbose=False
        )
        print("Modele charge.")

        for q in queries:
            # Requête brute sans wrapper chat — reproduit les conditions du papier
            out = llm(
                q,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=["\n"]
            )
            final_ans = out["choices"][0]["text"].strip()

            # Parsing : extraire la ligne Answer: ou Output: si presente
            out_list = final_ans.split('\n')
            for line in out_list:
                if 'Answer:' in line:
                    final_ans = line.replace('Answer:', '').strip()
                    break
                elif 'Output:' in line:
                    final_ans = line.replace('Output:', '').strip()
                    break

            if task == 'cause_and_effect':
                final_ans = 'Sentence ' + final_ans

            print('Model Output: ', final_ans)
            model_outputs.append(final_ans)

    # ---------------------------------------------------------
    # MODELE : CHATGPT (GPT-3.5) — DESACTIVE (nécessite OPENAI_API_KEY)
    # ---------------------------------------------------------
    # elif llm_model.lower() == 'chatgpt':
    #     api_key = os.getenv("OPENAI_API_KEY", "")
    #     if not api_key:
    #         print("ATTENTION: Variable d'environnement OPENAI_API_KEY manquante.")
    #
    #     for q in queries:
    #         output = _call_openai_api_with_retry(prompt=q, model_name="gpt-3.5-turbo", api_key=api_key)
    #         print('Model Output: ', output)
    #         model_outputs.append(output)

    # ---------------------------------------------------------
    # MODELE : GPT-4 — DESACTIVE (nécessite OPENAI_API_KEY)
    # ---------------------------------------------------------
    # elif llm_model.lower() == 'gpt4':
    #     api_key = os.getenv("OPENAI_API_KEY", "")
    #     for q in queries:
    #         output = _call_openai_api_with_retry(prompt=q, model_name="gpt-4", api_key=api_key)
    #         print('Model Output: ', output)
    #         model_outputs.append(output)

    # ---------------------------------------------------------
    # MODELE : VICUNA (Via endpoint OpenAI compatible local) — DESACTIVE
    # Nécessite un serveur Vicuna local : python -m fastchat.serve.api_server
    # ---------------------------------------------------------
    # elif llm_model.lower() == 'vicuna':
    #     for q in queries:
    #         output = _call_openai_api_with_retry(
    #             prompt=q,
    #             model_name="vicuna-13b-v1.1",
    #             api_key="EMPTY",
    #             api_base="http://0.0.0.0:8000/v1"
    #         )
    #
    #         index = output.find('.')
    #         if index > 0:
    #             output = output[:index]
    #         print('Model Output: ', output)
    #         model_outputs.append(output)

    # ---------------------------------------------------------
    # AUTRES (Fallback)
    # ---------------------------------------------------------
    else:
        print(f"Modele {llm_model} non supporte dans ce script.")

    return model_outputs
