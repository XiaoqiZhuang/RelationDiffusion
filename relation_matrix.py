import json
import numpy as np
import torch
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPTextModel, GPT2LMHeadModel, GPT2Tokenizer

model_name = "openai/clip-vit-large-patch14"  
processor = CLIPProcessor.from_pretrained(model_name)
text_encoder = CLIPTextModel.from_pretrained(model_name)

def solve_for_W(A, B):
    A_pseudo_inv = np.linalg.pinv(A)
    
    W = np.dot(A_pseudo_inv, B)
    return W

def extract_embedding(prompt, token_dic):
    inputs = processor(
        prompt,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    embeds = text_encoder(inputs.input_ids).last_hidden_state

    eot = embeds[:, len(prompt[0].split())+1]

    inputs = processor(
        token_dic['s1'],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    s1 = text_encoder(inputs.input_ids).last_hidden_state[:, 1]

    inputs = processor(
        token_dic['r'],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    r = text_encoder(inputs.input_ids).last_hidden_state[:, 1]

    inputs = processor(
        token_dic['s2'],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    s2 = text_encoder(inputs.input_ids).last_hidden_state[:, 1]

    return [s1, r, s2, eot]

def cal_relation_matrix(refer_prompts, refer_dic):
    refer = extract_embedding(refer_prompts, refer_dic)
    rm = torch.linalg.lstsq(refer[0]+refer[1]+refer[2], refer[3]).solution
    return rm 

def get_eot_embedding(relation_matrix, infer_prompts, infer_dic):
    infer = extract_embedding(infer_prompts, infer_dic)
    infer_eot = torch.matmul(infer[0]+infer[1]+infer[2], relation_matrix)
    infer[-1] = infer_eot
    return torch.stack(infer)

def relation_search(input_text, threshold=None, top_k=None):
    gpt = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    res = {}
    input_ids = gpt_tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = gpt(input_ids)
        logits = outputs.logits
    # 取最后一个词的logits
    last_token_logits = logits[0, -1, :]

    probabilities = torch.softmax(last_token_logits, dim=-1)

    if top_k:
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)
        top_k_tokens = [gpt_tokenizer.decode([idx]) for idx in top_k_indices]
        for i in range(top_k):
            res[top_k_tokens[i]] = top_k_probs[i].item()
    
    if threshold:
        high_prob_indices = (probabilities > threshold).nonzero(as_tuple=True)[0]
        high_probabilities = probabilities[high_prob_indices]

        words_and_probs = [(gpt_tokenizer.decode([idx]), prob.item()) for idx, prob in zip(high_prob_indices, high_probabilities)]
        words_and_probs_sorted = sorted(words_and_probs, key=lambda x: x[1], reverse=True)

        for word, prob in words_and_probs_sorted:
            res[word] = prob
    return res

def search_subject(subjects, relation):
    # 输入文本
    input_text = [f"a {sub} {relation} a" for sub in subjects]
    res = []
    for text in input_text:
        tmp = relation_search(text, threshold=0.05)
        if len(tmp) != 0:
            tmp_text = [f'{text}{sub}' for sub in tmp.keys()]
            res.extend(tmp_text)
    return res

def get_relation_matrix(relation, indice_dic):
    with open (f'{relation}.json', 'r') as fn:
        subjects = json.load(fn)

    refer_prompt = search_subject(subjects, relation)

    refer_dic = {name: [prompt.split()[pos-1] for prompt in refer_prompt] for name, pos in indice_dic.items()}

    relation_matrix = cal_relation_matrix(refer_prompt, refer_dic)
    return relation_matrix

def get_indice_dic():
    from dataset.benchmark_scenarios import inference_templates
    infer_prompts = {relation: [f'a {subject.split("{}")[0]}{relation} a{subject.split("{}")[1]}' for subject in subjects] for relation, subjects in inference_templates.items()}

    indice_dic = {}
    for relation, prompts in infer_prompts.items():
        print(prompts[0])

        tmp_dic = input().split(',')
        indice_dic[relation] = {'s1': tmp_dic[0], 'r': tmp_dic[1], 's2': tmp_dic[2]}
    print(indice_dic)
    with open('indice_dic.json', 'w') as fn:
        json.dump(indice_dic, fn)