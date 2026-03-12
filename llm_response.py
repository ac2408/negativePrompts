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


def get_response_from_llm(llm_model, queries, task, few_shot, api_num=4):
    model_outputs = []

    # ---------------------------------------------------------
    # MODELE : FLAN-T5-LARGE (GPU via device_map="auto")
    # Identique au papier : google/flan-t5-large
    # ---------------------------------------------------------
    if llm_model.lower() == 't5':
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        print("Chargement de google/flan-t5-large...")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

        use_cuda = model.device.type == "cuda"

        for q in queries:
            input_ids = tokenizer(q, return_tensors="pt").input_ids
            if use_cuda:
                input_ids = input_ids.to("cuda")
            outputs = model.generate(input_ids)

            out_text = tokenizer.decode(outputs[0])
            out_text = out_text.replace('<pad>', '')
            out_text = out_text.replace('</s>', '')
            out_text = out_text.replace('<s>', '')
            out_text = out_text.strip()
            print('Model Output: ', out_text)
            model_outputs.append(out_text)

    # ---------------------------------------------------------
    # MODELE : VICUNA-13B (GPU via HuggingFace transformers)
    # Papier : vicuna-13b-v1.1 via serveur API local (FastChat)
    # Ici   : lmsys/vicuna-13b-v1.5 via transformers (device_map="auto")
    # Prerequis : pip install transformers accelerate
    # ---------------------------------------------------------
    elif llm_model.lower() == 'vicuna':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_id = "lmsys/vicuna-13b-v1.5"
        print(f"Chargement de {model_id} (float16, device_map=auto)...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )

        def get_completion(prompt):
            vicuna_prompt = (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
                f"USER: {prompt}\nASSISTANT:"
            )
            input_ids = tokenizer(vicuna_prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=200, do_sample=False)
            out = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extraire uniquement la partie ASSISTANT:
            if "ASSISTANT:" in out:
                out = out.split("ASSISTANT:")[-1]
            return out.strip()

        for q in queries:
            output = None
            while output is None or output.isspace() or bool(re.search('[a-zA-Z]', output)) == False:
                try:
                    output = get_completion(q)
                    output = output.strip()
                except Exception as e:
                    print(e)
                    print('Retrying...')
                    time.sleep(5)
            index = output.find('.')
            if index > 0:
                output = output[:index]
            print('Model Input: ', q)
            print('Model Output: ', output)
            model_outputs.append(output)

    # ---------------------------------------------------------
    # MODELE : LLAMA-2-13B-CHAT (GPU via HuggingFace transformers)
    # Papier : ../Llama-2-13b-chat-hf (chemin local)
    # Ici   : meta-llama/Llama-2-13b-chat-hf (HuggingFace Hub)
    # Prerequis : accepter la licence sur https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
    #             puis : huggingface-cli login
    # ---------------------------------------------------------
    elif llm_model.lower() == 'llama2':
        from transformers import LlamaForCausalLM, LlamaTokenizer

        model_id = "meta-llama/Llama-2-13b-chat-hf"
        print(f"Chargement de {model_id}...")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id, device_map="auto")

        for q in queries:
            input_ids = tokenizer(q, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids, max_new_tokens=50, temperature=0.7, do_sample=True)

            out_text = tokenizer.decode(outputs[0])
            out_text = out_text.replace('<pad>', '')
            out_text = out_text.replace('</s>', '')
            out_text = out_text.replace('<s>', '')
            out_text = out_text.strip()
            out_list = out_text.split('\n')
            for i in range(len(out_list)):
                line = out_list[i]
                if 'Answer:' in line:
                    cur_line = line.replace('Answer:', '').strip()
                    if cur_line != '':
                        out_text = cur_line
                        break
                if 'Output:' in line:
                    cur_line = line.replace('Output:', '').strip()
                    if cur_line != '':
                        out_text = cur_line
                        break
                if 'Answer:' in line:
                    if i < len(out_list) - 1:
                        next_line = out_list[i+1].strip()
                        if next_line != '':
                            out_text = next_line
                            break
                        elif i < len(out_list) - 2:
                            next_line = out_list[i+2].strip()
                            if next_line != '':
                                out_text = next_line
                                break
                if 'Output:' in line:
                    if i < len(out_list) - 1:
                        next_line = out_list[i+1].strip()
                        if next_line != '':
                            out_text = next_line
                            break
                        elif i < len(out_list) - 2:
                            next_line = out_list[i+2].strip()
                            if next_line != '':
                                out_text = next_line
                                break
            out_text = out_text.strip()
            if task == 'cause_and_effect':
                out_text = 'Sentence ' + out_text
            print('Model Output: ', out_text)
            model_outputs.append(out_text)

    # ---------------------------------------------------------
    # MODELE : CHATGPT (GPT-3.5)
    # Prerequis : pip install openai; export OPENAI_API_KEY=sk-...
    # ---------------------------------------------------------
    elif llm_model.lower() == 'chatgpt':
        import openai

        def get_completion(prompt):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            return response["choices"][0]["message"]['content']

        for q in queries:
            output = None
            times = 0
            while output is None and times <= 10:
                try:
                    times += 1
                    output = get_completion(q)
                except Exception as e:
                    print(e)
                    print('Retrying...')
                    time.sleep(5)
            if times >= 10:
                print('Failed! Model Input: ', q)
                output = ''
            print('Model Output: ', output)
            model_outputs.append(output)

    # ---------------------------------------------------------
    # MODELE : GPT-4
    # Prerequis : pip install openai; export OPENAI_API_KEY=sk-...
    # ---------------------------------------------------------
    elif llm_model.lower() == 'gpt4':
        import openai

        def get_completion(prompt):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY", ""),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            return response["choices"][0]["message"]['content']

        for q in queries:
            output = None
            times = 0
            while output is None and times <= 10:
                try:
                    times += 1
                    output = get_completion(q)
                except Exception as e:
                    print(e)
                    print('Retrying...')
                    time.sleep(5)
            if times >= 10:
                print('Failed! Model Input: ', q)
                output = ''
            print('Model Output: ', output)
            model_outputs.append(output)

    else:
        print(f"Modele {llm_model} non supporte.")

    return model_outputs
