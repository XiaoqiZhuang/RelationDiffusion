from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from utils.ptp_utils import AttentionStore, aggregate_attention


class CustomLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CustomLoss, self).__init__()
        self.margin = margin

    def forward(self, x, y):
        loss = torch.nn.functional.relu(y - x + self.margin)
        return torch.mean(loss)


class SoBaPipeline(StableDiffusionPipeline):

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents
    
    @staticmethod
    def _compute_loss(qk_per_index: List[torch.Tensor], pixel_group: Dict) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        mse_loss = nn.MSELoss()
        criterion = CustomLoss(margin=0.1)

        bg0 = qk_per_index[0][pixel_group['bg']]
        bg1 = qk_per_index[1][pixel_group['bg']]
        bg_loss = mse_loss(bg0, bg1)

        s00 = qk_per_index[0][pixel_group['s0']]
        s01 = qk_per_index[1][pixel_group['s0']]
        s0_loss = criterion(s00, s01)
        
        s10 = qk_per_index[0][pixel_group['s1']]
        s11 = qk_per_index[1][pixel_group['s1']]
        s1_loss = criterion(s11, s10)

        loss = 0.4*s0_loss + 0.4*s1_loss  + 0.2*bg_loss
        return loss

    def _aggregate_and_get_QK_per_token(self, 
                                        attention_store: AttentionStore,
                                        indices_to_alter: List[int],
                                        attention_res: int = 16,
                                        ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        qk_per_index = torch.stack([attention_maps[:,:,i] for i in indices_to_alter])
        return qk_per_index

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss: torch.Tensor,
                                           text_embeddings: torch.Tensor,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           threshold: float = 0.3,
                                           attention_res: int = 16,
                                           max_refinement_steps: int = 20):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        
        while loss > threshold:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            qk_per_index = self._aggregate_and_get_QK_per_token(attention_store=attention_store, indices_to_alter=indices_to_alter, attention_res=attention_res)
            loss = self._compute_loss(qk_per_index)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        loss = self._compute_loss(qk_per_index)
        print(f"\t Finished with loss of: {loss}")
        return loss, latents
    

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 5,
            run_standard_sd: bool = False,
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
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
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
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

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                if not run_standard_sd:
                    p_group = torch.load('tmp/tmp_pixel_group.pt')
                    with torch.enable_grad():
                        if i < max_iter_to_alter:
                            for _ in range(3):
                        
                                latents = latents.clone().detach().requires_grad_(True)
                                # Forward pass of denoising with text conditioning
                                noise_pred_text = self.unet(latents, t,
                                                    encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                                self.unet.zero_grad()
                                # Get max activation value for each subject token
                                qk_per_index = self._aggregate_and_get_QK_per_token(attention_store=attention_store, 
                                                                            indices_to_alter=indices_to_alter, 
                                                                            attention_res=attention_res)
                                loss = self._compute_loss(qk_per_index, p_group)
                                if loss != 0:
                                    latents = self._update_latent(latents=latents, loss=loss,
                                                                  step_size=scale_factor * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Final Loss: {loss:0.4f}')

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

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
