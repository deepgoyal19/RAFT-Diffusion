#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""

import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import copy
import sys
import evaluate
from itertools import chain
# from transformers import (
#     Trainer,
#     default_data_collator,
#     set_seed,
# )
from copy import deepcopy
# from transformers.utils import send_example_telemetry
# from transformers.trainer_utils import get_last_checkpoint

from lmflow.datasets.dataset import Dataset
from lmflow_diffsuion import FinetunerArguments


logger = logging.getLogger(__name__)

class Finetuner:
    
    def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
        img_str = ""
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

        yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {base_model}
    tags:
    - stable-diffusion
    - stable-diffusion-diffusers
    - text-to-image
    - diffusers
    - lora
    inference: true
    ---
        """
        model_card = f"""
    # LoRA text2image fine-tuning - {repo_id}
    These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
    {img_str}
    """
        with open(os.path.join(repo_folder, "README.md"), "w") as f:
            f.write(yaml + model_card)

    
    def __int__(self):
        pass

    def compute_snr(timesteps):
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
    
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[FinetunerArguments.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{FinetunerArguments.caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    

class DiffusionFinetuner(FinetunerArguments,Finetuner):

    def __init__(self, *args, **kwargs):
        pass

    def finetune(self):
        logging_dir = os.path.join(FinetunerArguments.output_dir, FinetunerArguments.logging_dir)

        accelerator_project_config = ProjectConfiguration(total_limit=FinetunerArguments.checkpoints_total_limit)

        accelerator = Accelerator(
            gradient_accumulation_steps=FinetunerArguments.gradient_accumulation_steps,
            mixed_precision=FinetunerArguments.mixed_precision,
            log_with=FinetunerArguments.report_to,
            logging_dir=logging_dir,
            project_config=accelerator_project_config,
        )
        if FinetunerArguments.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if FinetunerArguments.seed is not None:
            set_seed(FinetunerArguments.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if FinetunerArguments.output_dir is not None:
                os.makedirs(FinetunerArguments.output_dir, exist_ok=True)

            if FinetunerArguments.push_to_hub:
                repo_id = create_repo(
                    repo_id=FinetunerArguments.hub_model_id or Path(FinetunerArguments.output_dir).name, exist_ok=True, token=FinetunerArguments.hub_token
                ).repo_id
        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(FinetunerArguments.pretrained_model_name_or_path, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(
            FinetunerArguments.pretrained_model_name_or_path, subfolder="tokenizer", revision=FinetunerArguments.revision
        )
        text_encoder = CLIPTextModel.from_pretrained(
            FinetunerArguments.pretrained_model_name_or_path, subfolder="text_encoder", revision=FinetunerArguments.revision
        )
        vae = AutoencoderKL.from_pretrained(FinetunerArguments.pretrained_model_name_or_path, subfolder="vae", revision=FinetunerArguments.revision)
        unet = UNet2DConditionModel.from_pretrained(
            FinetunerArguments.pretrained_model_name_or_path, subfolder="unet", revision=FinetunerArguments.revision
        )
        # freeze parameters of models to save more memory
        unet.requires_grad_(False)
        vae.requires_grad_(False)

        text_encoder.requires_grad_(False)

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        # now we will add new LoRA weights to the attention layers
        # It's important to realize here how many attention weights will be added and of which sizes
        # The sizes of the attention layers consist only of two different variables:
        # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
        # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

        # Let's first see how many attention processors we will have to set.
        # For Stable Diffusion, it should be equal to:
        # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
        # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
        # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
        # => 32 layers

        # Set correct lora layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        unet.set_attn_processor(lora_attn_procs)

        if FinetunerArguments.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
            lora_layers = AttnProcsLayers(unet.attn_processors)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if FinetunerArguments.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if FinetunerArguments.scale_lr:
            FinetunerArguments.learning_rate = (
                FinetunerArguments.learning_rate * FinetunerArguments.gradient_accumulation_steps * FinetunerArguments.train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        if FinetunerArguments.use_8bit_adam:
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
            lora_layers.parameters(),
            lr=FinetunerArguments.learning_rate,
            betas=(FinetunerArguments.adam_beta1, FinetunerArguments.adam_beta2),
            weight_decay=FinetunerArguments.adam_weight_decay,
            eps=FinetunerArguments.adam_epsilon,
        )

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if FinetunerArguments.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                FinetunerArguments.dataset_name,
                FinetunerArguments.dataset_config_name,
                cache_dir=FinetunerArguments.cache_dir,
            )
        else:
            data_files = {}
            if FinetunerArguments.train_data_dir is not None:
                data_files["train"] = os.path.join(FinetunerArguments.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=FinetunerArguments.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(FinetunerArguments.dataset_name, None)
        if FinetunerArguments.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = FinetunerArguments.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{FinetunerArguments.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if FinetunerArguments.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = FinetunerArguments.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{FinetunerArguments.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.

        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(FinetunerArguments.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(FinetunerArguments.resolution) if FinetunerArguments.center_crop else transforms.RandomCrop(FinetunerArguments.resolution),
                transforms.RandomHorizontalFlip() if FinetunerArguments.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        with accelerator.main_process_first():
            if FinetunerArguments.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=FinetunerArguments.seed).select(range(FinetunerArguments.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

            # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=FinetunerArguments.train_batch_size,
            num_workers=FinetunerArguments.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / FinetunerArguments.gradient_accumulation_steps)
        if FinetunerArguments.max_train_steps is None:
            FinetunerArguments.max_train_steps = FinetunerArguments.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            FinetunerArguments.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=FinetunerArguments.lr_warmup_steps * FinetunerArguments.gradient_accumulation_steps,
            num_training_steps=FinetunerArguments.max_train_steps * FinetunerArguments.gradient_accumulation_steps,
        )

        # Prepare everything with our `accelerator`.
        lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            lora_layers, optimizer, train_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / FinetunerArguments.gradient_accumulation_steps)
        if overrode_max_train_steps:
            FinetunerArguments.max_train_steps = FinetunerArguments.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        FinetunerArguments.num_train_epochs = math.ceil(FinetunerArguments.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("text2image-fine-tune", config=vars(args))

        # Train!
        total_batch_size = FinetunerArguments.train_batch_size * accelerator.num_processes * FinetunerArguments.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {FinetunerArguments.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {FinetunerArguments.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {FinetunerArguments.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {FinetunerArguments.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if FinetunerArguments.resume_from_checkpoint:
            if FinetunerArguments.resume_from_checkpoint != "latest":
                path = os.path.basename(FinetunerArguments.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(FinetunerArguments.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{FinetunerArguments.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                FinetunerArguments.resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(FinetunerArguments.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * FinetunerArguments.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * FinetunerArguments.gradient_accumulation_steps)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, FinetunerArguments.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, FinetunerArguments.num_train_epochs):
            unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if FinetunerArguments.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % FinetunerArguments.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if FinetunerArguments.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += FinetunerArguments.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if FinetunerArguments.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(timesteps)
                        mse_loss_weights = (
                            torch.stack([snr, FinetunerArguments.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(FinetunerArguments.train_batch_size)).mean()
                    train_loss += avg_loss.item() / FinetunerArguments.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = lora_layers.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, FinetunerArguments.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % FinetunerArguments.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(FinetunerArguments.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= FinetunerArguments.max_train_steps:
                    break

            if accelerator.is_main_process:
                if FinetunerArguments.validation_prompt is not None and epoch % FinetunerArguments.validation_epochs == 0:
                    logger.info(
                        f"Running validation... \n Generating {FinetunerArguments.num_validation_images} images with prompt:"
                        f" {FinetunerArguments.validation_prompt}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        FinetunerArguments.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=FinetunerArguments.revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device=accelerator.device).manual_seed(FinetunerArguments.seed)
                    images = []
                    for _ in range(FinetunerArguments.num_validation_images):
                        images.append(
                            pipeline(FinetunerArguments.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                        )

                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {FinetunerArguments.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                            )

                    del pipeline
                    torch.cuda.empty_cache()

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = unet.to(torch.float32)
            unet.save_attn_procs(FinetunerArguments.output_dir)

            if FinetunerArguments.push_to_hub:
                save_model_card(
                    repo_id,
                    images=images,
                    base_model=FinetunerArguments.pretrained_model_name_or_path,
                    dataset_name=FinetunerArguments.dataset_name,
                    repo_folder=FinetunerArguments.output_dir,
                )
                upload_folder(
                    repo_id=repo_id,
                    folder_path=FinetunerArguments.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            FinetunerArguments.pretrained_model_name_or_path, revision=FinetunerArguments.revision, torch_dtype=weight_dtype
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(FinetunerArguments.output_dir)

        # run inference
        generator = torch.Generator(device=accelerator.device).manual_seed(FinetunerArguments.seed)
        images = []
        for _ in range(FinetunerArguments.num_validation_images):
            images.append(pipeline(FinetunerArguments.validation_prompt, num_inference_steps=30, generator=generator).images[0])

        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {FinetunerArguments.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        accelerator.end_training()
