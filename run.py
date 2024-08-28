import pprint
from typing import List

import torch
from PIL import Image

from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore, aggregate_attention

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def run_on_prompt(cur,
                  infer_info, 
                  prompt: List[str],
                  model,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  ) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(cur=cur,
                    infer_info=infer_info,
                    prompt=prompt,
                    generator=seed,
                    num_inference_steps = 50,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    run_standard_sd = True,
                    # num_images_per_prompt=2,
                    )
    
    # attention_maps = aggregate_attention(controller, 16, ("up", "down", "mid"), True, 0).detach().cpu()
    
    # token_attn_norm = [round(torch.norm(attention_maps[:, :, i]).item(), 2) for i in token_indices]
    
    images = outputs.images
    return images


