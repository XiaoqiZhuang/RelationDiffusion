import torch

from PIL import Image, ImageDraw
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
import random
import numpy as np

from ptp_utils import register_attention_control, AttentionStore
from myunet import MyUNet2DConditionModel
from FakeImageGuidancePipeline import FakeImageGuidancePipeline, convert_bbox
from ConditionGuidancePipeline import ConditionGuidancePipeline


def bbox_size(bbox, W, H):

    x_min = int(bbox[0] * W)
    y_min = int(bbox[1] * H)
    x_max = int(bbox[2] * W)
    y_max = int(bbox[3] * H)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    return bbox_width, bbox_height


if __name__ == '__main__':
    sd_id = "CompVis/stable-diffusion-v1-4"
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pipe = FakeImageGuidancePipeline.from_pretrained(sd_id).to(device)
    # pipe = ConditionGuidancePipeline.from_pretrained(sd_id).to(device)
    
    controller = AttentionStore()
    register_attention_control(pipe, controller)
    
    # relation_bbox = [0.4, 0.4, 0.6, 0.5]
    # bboxes = [[0.4, 0.1, 0.6, 0.4], [0.2, 0.45, 0.8, 0.8]] 
    # prompt = ['a man is riding on a dog']
    
    relation_bbox = [0.4, 0.3, 0.6, 0.5]
    bboxes = [[0.0,0.2,0.4,0.8], [0.6, 0.2, 1.0, 0.8]] 
    prompt = ['a man is shaking hands with a woman']

    for i in range(5):
        g = torch.Generator('cuda').manual_seed(i)

        images = pipe(prompt, relation_bbox, bboxes, controller, [2, 7], generator=g).images
        
        image = images[0]
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = convert_bbox(bbox, image_height, image_width)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        x_min, y_min, x_max, y_max = convert_bbox(relation_bbox, image_height, image_width)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)

        image.save(f'data/output/shakehand/man_{i}.png')


