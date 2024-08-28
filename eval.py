import os
import random
import json
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from utils.ptp_utils import AttentionStore
from run import run_on_prompt
from InteratPipeline import InteracPipeline
from relation_matrix import get_eot_embedding, get_relation_matrix

from dataset.benchmark_scenarios import inference_templates

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
Stable = InteracPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
tokenizer = Stable.tokenizer

infer_relations = inference_templates.keys()
infer_prompts = {relation: [f'a {subject.split("{}")[0]}{relation} a{subject.split("{}")[1]}' for subject in subjects] for relation, subjects in inference_templates.items()}

with open('dataset/indice_dic.json', 'r') as fn:
    indice_dic = json.load(fn)

for relation, prompts in infer_prompts.items():
    tmp_indice_dic = indice_dic[relation]
    tmp_indice_dic = {name: int(pos) for name, pos in tmp_indice_dic.items()}

    infer_dic = {name: [prompt.split()[pos-1] for prompt in prompts] for name, pos in tmp_indice_dic.items()}

    rm = get_relation_matrix(relation, tmp_indice_dic)
    # rm = torch.load('relation_matrix.pt')
    infer_info = get_eot_embedding(rm, prompts, infer_dic).cuda()

    for i, prompt in enumerate(prompts):
        seed = 14
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        
        images = Stable(
                        infer_info=infer_info[:, i].view(4,1,768),
                        indice_dic=tmp_indice_dic,
                        prompt=prompt,
                        generator=g,
                        num_inference_steps = 50,
                        num_images_per_prompt=5,
                        ).images
        for _, image in enumerate(images):
            image.save(f'output/new/paint/{prompt}_{_}.png')

    # seed = 14
    # g = torch.Generator('cuda').manual_seed(seed)
    # controller = AttentionStore()
        
    # images = Stable(
    #                 infer_info=infer_info,
    #                 indice_dic=tmp_indice_dic,
    #                 prompt=batch_2,
    #                 generator=g,
    #                 num_inference_steps = 50,
    #                 # num_images_per_prompt=2,
    #                 ).images
    # for i, image in enumerate(images):
    #     image.save(f'output/{prompts[i]}.png')