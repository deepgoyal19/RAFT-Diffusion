#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""

import logging
import math
import os
import accelerate
import datasets
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, deprecate




if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)

class Finetuner:
    
    def __init__(self, *args, **kwargs):
        pass

    def compute_snr(self, timesteps, noise_scheduler):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr


    def save_model_hook(self, models, weights, output_dir, ema_unet):
        if self.model_args.use_ema:
            ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(self, models, input_dir, ema_unet):
        if self.model_args.use_ema:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(self.accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
    
    def deepspeed_zero_init_disabled_context_manager(self):
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


class DiffusionFinetuner(Finetuner):
    """
    Initializes the `Finetuner` class with given arguments.

    Parameters
    ------------
    model_args : ModelArguments object.
        Contains the arguments required to load the model.

    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.

    finetuner_args : FinetunerArguments object.
        Contains the arguments required to perform finetuning.

    args : Optional.
        Positional arguments.

    kwargs : Optional.
        Keyword arguments.

    """   

    def __init__(self, finetuner_args, model_args, data_args, *args, **kwargs):
        
        self.finetuner_args = finetuner_args
        self.model_args = model_args   
        self.data_args = data_args

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
        # logger.info(f"Training parameters {finetuner_args}")

        # If passed along, set the training seed now.
        if self.finetuner_args.seed is not None:
            set_seed(self.finetuner_args.seed)

    def finetune(self, model, dataset):

        logging_dir = os.path.join(self.finetuner_args.output_dir, self.finetuner_args.logging_dir)

        self.accelerator_project_config = ProjectConfiguration(
            total_limit=self.finetuner_args.checkpoints_total_limit, project_dir=self.finetuner_args.output_dir, logging_dir=logging_dir)

        Accelerator.device = torch.device(self.finetuner_args.accelerate_device)
        Accelerator.num_processes = self.finetuner_args.accelerate_num_processes
        Accelerator.process_index = self.finetuner_args.accelerate_process_index
        Accelerator.sync_gradients = self.finetuner_args.accelerate_sync_gradients
        Accelerator.use_distributed = self.finetuner_args.accelerate_use_distributed
        Accelerator.accelerate_optimizer_step_was_skipped = self.finetuner_args.accelerate_optimizer_step_was_skipped
        Accelerator.local_process_index = self.finetuner_args.accelerate_local_process_index

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.finetuner_args.gradient_accumulation_steps,
            mixed_precision=self.finetuner_args.mixed_precision,
            log_with=self.finetuner_args.report_to,
            project_config=self.accelerator_project_config,
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        model.set_weight_dtype(self.accelerator.mixed_precision)

        model.to_device(self.accelerator.device)

        if self.model_args.use_lora:
            if self.finetuner_args.report_to == "wandb":
                if not is_wandb_available():
                    raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
                import wandb

        # logger.info(self.accelerator.state, main_process_only=False)
        logger.info(self.accelerator.state)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()



        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.finetuner_args.output_dir is not None:
                os.makedirs(self.finetuner_args.output_dir, exist_ok=True)


        if self.model_args.use_lora:
            model.set_lora_attn_proccessor_to_unet()


        if self.finetuner_args.enable_xformers_memory_efficient_attention:
            model.use_xformers()

        if self.model_args.use_lora:
            lora_layers = AttnProcsLayers(model.unet.attn_processors)
        else:
            
            if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
                # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
                self.accelerator.register_save_state_pre_hook(self.save_model_hook)
                self.accelerator.register_load_state_pre_hook(self.load_model_hook)
            
            if self.finetuner_args.gradient_checkpointing:
                model.unet.enable_gradient_checkpointing()

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices

    
        if self.finetuner_args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.finetuner_args.scale_lr:
            self.finetuner_args.learning_rate = (
                self.finetuner_args.learning_rate * self.finetuner_args.gradient_accumulation_steps * self.finetuner_args.train_batch_size * self.accelerator.num_processes
            )

        # Initialize the optimizer
        if self.finetuner_args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            model.unet.parameters(),
            lr=self.finetuner_args.learning_rate,
            betas=(self.finetuner_args.adam_beta1, self.finetuner_args.adam_beta2),
            weight_decay=self.finetuner_args.adam_weight_decay,
            eps=self.finetuner_args.adam_epsilon,
        )

        # Get training dataset and training dataloader
        train_dataset = dataset.train_dataset(self.accelerator, self.finetuner_args.seed, self.finetuner_args.max_train_samples, model.tokenizer)
        train_dataloader = dataset.train_dataloader(self.finetuner_args.train_batch_size)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.finetuner_args.gradient_accumulation_steps)
        if self.finetuner_args.max_train_steps is None:
            self.finetuner_args.max_train_steps = self.finetuner_args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.finetuner_args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.finetuner_args.lr_warmup_steps * self.finetuner_args.gradient_accumulation_steps,
            num_training_steps=self.finetuner_args.max_train_steps * self.finetuner_args.gradient_accumulation_steps,
        )

        if self.model_args.use_lora: 
            # Prepare everything with our `accelerator`.
            lora_layers, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                lora_layers, optimizer, train_dataloader, lr_scheduler
            )
        else:
            # Prepare everything with our `accelerator`.
            model.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                model.unet, optimizer, train_dataloader, lr_scheduler
            )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.finetuner_args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.finetuner_args.max_train_steps = self.finetuner_args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.finetuner_args.num_train_epochs = math.ceil(self.finetuner_args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        
        
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.finetuner_args))
            tracker_config.pop("validation_prompts")
            self.accelerator.init_trackers(self.finetuner_args.tracker_project_name, tracker_config)

        # Train!
        total_batch_size = self.finetuner_args.train_batch_size * self.accelerator.num_processes * self.finetuner_args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {self.finetuner_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.finetuner_args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.finetuner_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.finetuner_args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save


        self.finetuner_args.resume_from_checkpoint, global_step, first_epoch, resume_step= model.resume_from_path(
                                                            self.finetuner_args.resume_from_checkpoint, 
                                                            self.finetuner_args.output_dir,
                                                            self.finetuner_args.gradient_accumulation_steps,
                                                            num_update_steps_per_epoch,
                                                            self.accelerator
                                                        )   

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, self.finetuner_args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.finetuner_args.num_train_epochs):
            model.unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if self.finetuner_args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % self.finetuner_args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(model.unet):
                    # Convert images to latent space
                    latents = model.vae.encode(batch["pixel_values"].to(model.weight_dtype)).latent_dist.sample()
                    latents = latents * model.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if self.finetuner_args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self.finetuner_args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if self.finetuner_args.input_perturbation:
                        new_noise = noise + self.finetuner_args.input_perturbation * torch.randn_like(noise)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if self.finetuner_args.input_perturbation:
                        noisy_latents = model.noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = model.text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if model.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif model.noise_scheduler.config.prediction_type == "v_prediction":
                        target = model.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {model.noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self.finetuner_args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = self.compute_snr(timesteps, model.noise_scheduler)
                        mse_loss_weights = (
                            torch.stack([snr, self.finetuner_args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.finetuner_args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.finetuner_args.gradient_accumulation_steps

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.unet.parameters(), self.finetuner_args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.model_args.use_ema and (self.model_args.use_lora is False):
                        model.ema_unet.step(model.unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % self.finetuner_args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(self.finetuner_args.output_dir, f"checkpoint-{global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= self.finetuner_args.max_train_steps:
                    break
        
            # Make a validation log
            model.log_validation(self.finetuner_args,self.accelerator,global_step)

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:         
            # save model
            model.save(self.finetuner_args.output_dir, self.accelerator)

            # Push model to hub
            if self.finetuner_args.push_to_hub:
                model.push_to_hub(
                    self.finetuner_args.hub_model_id,
                    self.finetuner_args.output_dir, 
                    self.finetuner_args.hub_token, 
                    self.data_args.dataset_name
                    )
        
        #Final Inference
        if self.model_args.use_lora:
            model.final_inference(self.finetuner_args, self.accelerator, global_step)  

        self.accelerator.end_training()
