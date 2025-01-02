from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Resize
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput

from ptp_utils import AttentionStore, aggregate_attention
from gaussian_smoothing import GaussianSmoothing


def convert_bbox(bbox, H, W):
    x_min = int(bbox[0] * W)
    y_min = int(bbox[1] * H)
    x_max = int(bbox[2] * W)
    y_max = int(bbox[3] * H)
    return x_min, y_min, x_max, y_max


def reverse_x0(pipeline, noise, noised_samples, timesteps):
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device=noised_samples.device)
    timesteps = timesteps.to(noised_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    while len(sqrt_alpha_prod.shape) < len(noised_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noised_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    original_samples = (noised_samples - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
    return original_samples

def reverse_ep0(pipeline, original_samples, noised_samples, timesteps):
    alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device=original_samples.device)
    timesteps = timesteps.to(original_samples.device)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    ep0 = (noised_samples - sqrt_alpha_prod * original_samples) / sqrt_one_minus_alpha_prod
    return ep0
    


class FakeImageGuidancePipeline(StableDiffusionPipeline):


    def _compute_content_loss(self, pred_noise, latent, timestep, tar_latents, bboxes) -> torch.Tensor:
        W, H = pred_noise.shape[-1], pred_noise.shape[-2]
        
        loss = 0
        i=0

        for tar_latent, bbox in zip(tar_latents, bboxes):
            tar_latent = tar_latent.to(pred_noise.device)

            x_min, y_min, x_max, y_max = convert_bbox(bbox, H, W)
            # print(f'bbox{i}: {x_min, y_min, x_max, y_max}')
            
            if x_max-x_min != tar_latent.shape[-1]:
                x_max -= 1
            if y_max-y_min != tar_latent.shape[-2]:
                y_max -= 1

            bbox_noise = pred_noise[:, :, y_min:y_max, x_min:x_max] 
            bbox_latent = latent[:, :, y_min:y_max, x_min:x_max]
            gt_noise = reverse_ep0(self, tar_latent, bbox_latent, timestep)

            # # test guidance
            # test_lt = latent[:, :, y_min:y_max, x_min:x_max].detach()
            # with torch.no_grad(): 
            #     for t in self.scheduler.timesteps:
            #         tmp_noise = reverse_ep0(self, tar_latent, test_lt, t)
            #         test_lt = self.scheduler.step(tmp_noise, t, test_lt).prev_sample
            #     test_lt = test_lt.detach()
            #     image = self.vae.decode(test_lt / self.vae.config.scaling_factor, return_dict=False)[0]
            #     do_denormalize = [True] * image.shape[0]
            #     image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)[0]
            #     image.save(f'bbox_image_return_{timestep}_{i}.png')

            # print(f'----------{bbox_x0.shape}, {tar_img.shape}')
            loss += F.mse_loss(bbox_noise, gt_noise)

            i+=1
        # print('-------------')
        return loss

    @staticmethod
    def _compute_layout_loss(attention_maps, indices_to_alter, bboxes) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        loss = 0

        for res, avg_cross_maps in attention_maps.items():
            for i, bbox in zip(indices_to_alter, bboxes):
                mask = torch.zeros(size=(res, res)).cuda() if torch.cuda.is_available() else torch.zeros(size=(res, res))
                x_min, y_min, x_max, y_max = convert_bbox(bbox, res, res)
                mask[y_min: y_max, x_min: x_max] = 1

                ca_map_obj = avg_cross_maps[:, :, i]

                activation_value = (ca_map_obj * mask).sum()/ca_map_obj.sum()
                loss += torch.mean((1 - activation_value) ** 2)
        loss /= len(indices_to_alter)
        return loss
    
    @staticmethod
    def _compute_relation_loss(attention_maps, relation_bbox, bboxes):
        loss = 0

        bboxes.append(relation_bbox)

        for res, avg_self_maps in attention_maps.items():
            grid = torch.arange(res*res).view(res, res)
            x_min, y_min, x_max, y_max = convert_bbox(relation_bbox, res, res)

            bbox_coords = grid[y_min: y_max, x_min: x_max]
            bbox_columns = bbox_coords.flatten()
            
            mask = torch.zeros(size=(res, res)).cuda() if torch.cuda.is_available() else torch.zeros(size=(res, res))
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = convert_bbox(bbox, res, res)
                mask[y_min: y_max, x_min: x_max] = 1                

            for c in bbox_columns:
                sa_map = avg_self_maps[:, :, c]
                activation_value = (sa_map * mask).sum()/sa_map.sum()
                loss += torch.mean((1 - activation_value) ** 2)
        loss /= len(bbox_columns)
        return loss


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        relation_bbox,
        obj_bboxes:Union[float, List[float]],
        # reference_images,
        attention_store: AttentionStore,
        indices_to_alter: List[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        loss = torch.tensor(10000)
        layout_step = 10
        content_step = 30

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                iteration = 0
                while loss.item() > 0.2 and iteration < 5 and i < layout_step:
                    with torch.enable_grad():
                        latents = latents.clone().detach().requires_grad_(True)
                        latent_model_input = self.scheduler.scale_model_input(latents, t)
                        
                        noise_pred_text = self.unet(latent_model_input, t,
                                                    encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                        self.unet.zero_grad()

                        ca_maps =  aggregate_attention(attention_store=attention_store,
                                                         # from_where=("up", "down", "mid"),
                                                         from_where=["up", "mid"],
                                                         is_cross=True)
                        
                        layout_loss = self._compute_layout_loss(ca_maps, indices_to_alter, obj_bboxes)

                        sa_maps = aggregate_attention(attention_store=attention_store,
                                                         # from_where=("up", "down", "mid"),
                                                         from_where=["up", "mid"],
                                                         is_cross=False)
                        rela_loss = self._compute_relation_loss(sa_maps, relation_bbox, obj_bboxes) 

                        self.start_step_size = 15
                        self.step_size_coef = (8 - 15) / layout_step
                        step_size = self.start_step_size + self.step_size_coef * i

                        loss = layout_loss + rela_loss
                        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
                        
                        print(f'i:{i}, step_size: {step_size}, iteration:{iteration}, layout_loss:{layout_loss.item()}, rela_loss:{rela_loss.item()}')

                        latents = latents - step_size* grad_cond 
                        iteration += 1
                        torch.cuda.empty_cache()  
                
                # loss = torch.tensor(10000)
                # iteration = 0
                # while loss.item() > 0.01 and iteration < 5 and i < content_step and i >= layout_step:          
                #     with torch.enable_grad():  
                #         latents = latents.clone().detach().requires_grad_(True)
                #         latent_model_input = self.scheduler.scale_model_input(latents, t)
                #         # Forward pass of denoising with text conditioning
                #         noise_pred_text = self.unet(latent_model_input, t,
                #                                     encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                #         self.unet.zero_grad()

                #         content_loss = self._compute_content_loss(noise_pred_text, latents, t, reference_images, bboxes)

                #         beta = beta_factor # * np.sqrt(scale_range[i]) 
                #         loss = content_loss
                #         grad_cond = torch.autograd.grad(loss, [latents])[0]
                        
                #         print(f'i:{i}, Content, Factor: {beta}, iteration:{iteration}, loss:{loss.item()}')

                #         latents = latents - beta * grad_cond 
                #         iteration += 1

                #         del grad_cond, latent_model_input, noise_pred_text
                #         torch.cuda.empty_cache()  

                torch.cuda.empty_cache()  
                with torch.no_grad(): 
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                    # perform guidance   
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # # reverse noise evaluation                    
                    # ep0 = reverse_ep0(self, reference_images[0], latents, t)
                    # latents = self.scheduler.step(ep0, t, latents, **extra_step_kwargs).prev_sample

                    progress_bar.update()

        # 8. Post-processing
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)
        
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
