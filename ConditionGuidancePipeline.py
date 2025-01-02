from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Resize
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput

from ptp_utils import AttentionStore, aggregate_attention

def convert_bbox(bbox, H, W):
    x_min = int(bbox[0] * W)
    y_min = int(bbox[1] * H)
    x_max = int(bbox[2] * W)
    y_max = int(bbox[3] * H)
    return x_min, y_min, x_max, y_max


class ConditionGuidancePipeline(StableDiffusionPipeline):

    def _aggregate_attention_maps(self, attention_store: AttentionStore):
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=32,
            from_where=["up", "mid"],
            is_cross=True,
            select=0)
    
        return attention_maps
    
    def _compute_content_loss(self, pred_noise, pred_cond_noises, bboxes) -> torch.Tensor:
        W, H = pred_noise.shape[-1], pred_noise.shape[-2]
        
        loss = 0
        for pred_cond_noise, bbox in zip(pred_cond_noises, bboxes):

            x_min, y_min, x_max, y_max = convert_bbox(bbox, H, W)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            bbox_cond_noise = F.interpolate(pred_cond_noise.unsqueeze(0), size=(bbox_height, bbox_width), mode='bilinear', align_corners=False)
            bbox_pred_noise = pred_noise[:, :, y_min:y_max, x_min:x_max] 
            loss += F.mse_loss(bbox_pred_noise, bbox_cond_noise)
        return loss

    @staticmethod
    def _compute_layout_loss(attention_maps, indices_to_alter, bboxes) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        loss = 0

        for res, avg_cross_maps in attention_maps.items():
            H, W = avg_cross_maps.shape[:2]

            for i, bbox in zip(indices_to_alter, bboxes):
                mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
                x_min, y_min, x_max, y_max = convert_bbox(bbox, H, W)
                mask[y_min: y_max, x_min: x_max] = 1

                ca_map_obj = avg_cross_maps[:, :, i]

                activation_value = (ca_map_obj * mask).sum()/ca_map_obj.sum()
                loss += torch.mean((1 - activation_value) ** 2)

        return loss
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        condition_prompts: List[str],
        bboxes:Union[float, List[float]],
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
        guidance_rescale: float = 0.0,
        alpha_factor: int = 30,
        beta_factor: float = 20,
        scale_range: Tuple[float, float] = (1., 0.5),
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
        

        # 3. Encode input prompt
        condition_embeds, _ = self.encode_prompt(
            condition_prompts,
            device,
            num_images_per_prompt,
            False,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

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

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        loss = torch.tensor(10000)
        layout_iteration = 10
        content_iteration = 30

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                iteration = 0
                while loss.item() > 0.2 and iteration < 5 and i < layout_iteration:
                    with torch.enable_grad():                        
                        latents = latents.clone().detach().requires_grad_(True)
                        latent_model_input = self.scheduler.scale_model_input(latents, t)
                        # Forward pass of denoising with text conditioning
                        noise_pred_text = self.unet(latent_model_input, t,
                                                    encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                        self.unet.zero_grad()

                        attn_maps = self._aggregate_attention_maps(attention_store=attention_store)
                        layout_loss = self._compute_layout_loss(attn_maps, indices_to_alter, bboxes)


                        # noise_pred_conditions = self.unet(torch.cat([latent_model_input] * len(condition_prompts)), t,
                        #                             encoder_hidden_states=condition_embeds, cross_attention_kwargs=cross_attention_kwargs).sample.detach()

                        # content_loss = self._compute_content_loss(noise_pred_text, noise_pred_conditions, bboxes)

                        alpha = alpha_factor * np.sqrt(scale_range[i]) 
                        # beta = beta_factor * np.sqrt(scale_range[i]) 
                        # loss = alpha*layout_loss + beta*content_loss
                        loss = layout_loss
                        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
                        
                        print(f'i:{i}, Layout, iteration:{iteration}, loss:{loss.item()}')

                        latents = latents - alpha * grad_cond 
                        iteration += 1

                loss = torch.tensor(10000)
                iteration = 0
                while loss.item() > 0.2 and iteration < 5 and i < layout_iteration:          
                # while loss.item() > 0.2 and iteration < 1 and i >= layout_iteration and i < content_iteration:
                    with torch.enable_grad():  
                        latents = latents.clone().detach().requires_grad_(True)
                        latent_model_input = self.scheduler.scale_model_input(latents, t)
                        # Forward pass of denoising with text conditioning
                        noise_pred_text = self.unet(latent_model_input, t,
                                                    encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                        self.unet.zero_grad()

                        with torch.no_grad():
                            noise_pred_conditions = self.unet(torch.cat([latent_model_input] * len(condition_prompts)), t,
                                                    encoder_hidden_states=condition_embeds, cross_attention_kwargs=cross_attention_kwargs).sample.detach()

                        content_loss = self._compute_content_loss(noise_pred_text, noise_pred_conditions, bboxes)

                        beta = beta_factor * np.sqrt(scale_range[i]) 
                        loss = content_loss
                        grad_cond = torch.autograd.grad(loss, [latents])[0]
                        
                        print(f'i:{i}, Content, iteration:{iteration}, loss:{loss.item()}')

                        latents = latents - beta * grad_cond 
                        iteration += 1

                        del grad_cond, latent_model_input, noise_pred_text, noise_pred_conditions
                        torch.cuda.empty_cache()  

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
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) # - alpha *grad_layout # - alpha *grad_pixel
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    
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
