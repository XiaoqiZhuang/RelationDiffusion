import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import random


if __name__ == '__main__':

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    Stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    # Stable = StableDiffusionPipeline.from_pretrained("FakeDiffusion").to(device)
    # Stable.load_lora_weights('fake_image_lora')
    # init_image = Image.open("data/fake/elephant_rabbit.png").convert("RGB")

    images = Stable(["a bear"], num_inference_steps=50, num_images_per_prompt=5).images
    for i, image in enumerate(images):
        image.save(f'data/material/bear{i}.png')
