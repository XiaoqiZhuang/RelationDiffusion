import os
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from transformers import AdamW

from peft import get_peft_model_state_dict

from fake_dataset import FIGDataset



torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 5

model_id = "CompVis/stable-diffusion-v1-4"  
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

lora_config = LoraConfig(
    r=4,  
    lora_alpha=32,  
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], 
    lora_dropout=0.1, 
)

pipe.unet = get_peft_model(pipe.unet , lora_config)
pipe.unet = pipe.unet.get_base_model()

lora_layers = filter(lambda p: p.requires_grad, pipe.unet.parameters())


root = 'data/dataset'
prompt = ""
p = os.listdir(root)
p = [f'{root}/{tmp}' for tmp in p]

dataset = FIGDataset(p, prompt)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
print('batch_size', data_loader.batch_size, 'len', len(data_loader.dataset))

optimizer = AdamW(lora_layers, lr=1e-6)


def train_unet(num_epochs=2000):
    pipe.unet.train()

    for epoch in range(num_epochs):
        for images, captions in tqdm(data_loader):
            optimizer.zero_grad()

            input_ids = pipe.tokenizer(captions, truncation=True, padding="max_length", max_length=77, return_tensors="pt").to(pipe.device)
            text_embeddings = pipe.text_encoder(input_ids.input_ids).last_hidden_state
            
            images = images.to(pipe.device)
            latents = pipe.vae.encode(images).latent_dist.sample() * pipe.vae.config.scaling_factor
            
            noises = torch.randn_like(latents).to(pipe.device)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=pipe.device).long()

            noisy_latents = pipe.scheduler.add_noise(latents, noises, timesteps)
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
            
            loss = nn.functional.mse_loss(noise_pred, noises)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} completed! Loss: {loss.item()}")

train_unet()

unet_lora_state_dict = get_peft_model_state_dict(pipe.unet)


StableDiffusionPipeline.save_lora_weights(
    save_directory='fake_image_lora',
    unet_lora_layers=unet_lora_state_dict,
    safe_serialization=True,
)