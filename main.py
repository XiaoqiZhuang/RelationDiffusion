import os
import random
import json
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils.ptp_utils import AttentionStore, aggregate_attention
from run import run_on_prompt
from utils import ptp_utils, vis_utils
from SoBaPipeline import SoBaPipeline
from InteratPipeline import InteracPipeline
from relation_matrix import get_new_eot, get_subject_embedding, search_subject


TOKEN_ATTN_NORM = {}
OUTPUT_PATH = 'output/raw'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# Stable = InteracPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to(device)
Stable = InteracPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
tokenizer = Stable.tokenizer

# # 加载GPT-2模型和分词器
# gpt = GPT2LMHeadModel.from_pretrained('gpt2')
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # prompts = ['a ' + s1 + ' is riding on a ' + s2 for s1 in ['man', 'woman'] for s2 in ['bike', 'motorcycle', 'horse']]

# with open ('notebook/painted.json', 'r') as fn:
#     subjects = json.load(fn)

# RELATION = 'painted'
# refer_prompt = search_subject(subjects, RELATION, gpt, gpt_tokenizer)

# infer_prompt = [f"a Badge is painted on a white"] 

# indice_dic = {'s1': 2, 's2': 7, 'r': 4}
# refer_dic = {name: [prompt.split()[pos-1] for prompt in prompts] for name, pos in indice_dic.items()}
# refer_prompt = [f"a {s1} is {r} on a {s2}" for s1, r, s2 in zip(refer_dic['s1'], refer_dic['r'], refer_dic['s2'])]

# infer_dic = {name: [prompt.split()[pos-1] for prompt in infer_prompt] for name, pos in indice_dic.items()}
# infer_eot = get_new_eot(refer_prompt, infer_prompt, refer_dic, infer_dic)

# sub_embeds = get_subject_embedding(['guitar', 'painted', 'paper'])

# torch.save(infer_eot, 'infer_eot.pt')
# torch.save(sub_embeds, 'sub_embeds.pt')

infer_prompt = [f"a Badge is painted on a white"] 


for i, prompt in enumerate(infer_prompt):
    seed = 25
    g = torch.Generator('cuda').manual_seed(seed)
    controller = AttentionStore()
    
    images, at = run_on_prompt(i, prompt,
                            Stable,
                            controller=controller,
                            token_indices=[2, 7],
                            seed=g,
                            )
    for _, image in enumerate(images):
        image.save(f'output/new/{prompt}_{seed}_raw.png')


