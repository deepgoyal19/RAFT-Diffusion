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
import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, deprecate
from diffusers.utils.import_utils import is_xformers_available
import copy
import sys
from itertools import chain


if is_wandb_available():
    import wandb

from copy import deepcopy

# from lmflow.datasets.dataset import Dataset
# from lmflow_diffusion.args import FinetunerArguments


logger = logging.getLogger(__name__)

class Finetuner:
    
    def __init__(self, *args, **kwargs):
        pass
        # self.model_args = model_args
        # self.data_args = data_args
        # self.finetuner_args = finetuner_args
    
    def save_model_card(self, repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
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

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
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
    
    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples[self.finetuner_args.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{self.finetuner_args.caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions, max_length = self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def preprocess_train(self, examples):
        images = [image.convert("RGB") for image in examples[self.image_column]]
        examples["pixel_values"] = [self.train_transforms(image) for image in images]
        examples["input_ids"] = self.tokenize_captions(examples)
        return examples

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    def log_validation(self, vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
        logger.info("Running validation... ")

        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_args.pretrained_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            safety_checker=None,
            revision=self.model_args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if self.finetuner_args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        if self.finetuner_args.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(self.finetuner_args.seed)

        images = []
        for i in range(len(self.finetuner_args.validation_prompts)):
            with torch.autocast("cuda"):
                image = pipeline(self.finetuner_args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

            images.append(image)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            elif tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {self.finetuner_args.validation_prompts[i]}")
                            for i, image in enumerate(images)
                        ]
                    }
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        torch.cuda.empty_cache()
    
    def save_model_hook(self, models, weights, output_dir):
        if self.model_args.use_ema:
            self.ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.model_args.use_ema:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
            self.ema_unet.load_state_dict(load_model.state_dict())
            self.ema_unet.to(self.accelerator.device)
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

    def __init__(self, finetuner_args, model_args, *args, **kwargs):
        
        self.finetuner_args = finetuner_args
        self.model_args = model_args   
        # self.data_args = data_args

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

    def finetune(self, model):
        
        unet = model.unet
        vae = model.vae
        text_encoder =model.text_encoder
        self.noise_scheduler = model.noise_scheduler
        self.tokenizer = model.tokenizer

        DATASET_NAME_MAPPING = {
            "lambdalabs/pokemon-blip-captions": ("image", "text"),
        }

        logging_dir = os.path.join(self.finetuner_args.output_dir, self.finetuner_args.logging_dir)

        self.accelerator_project_config = ProjectConfiguration(total_limit=self.finetuner_args.checkpoints_total_limit)

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
            logging_dir=logging_dir,
            project_config=self.accelerator_project_config,
        )

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

            if self.finetuner_args.push_to_hub:
                repo_id = create_repo(
                    repo_id=self.finetuner_args.hub_model_id or Path(self.finetuner_args.output_dir).name, exist_ok=True, token=self.finetuner_args.hub_token
                ).repo_id


        if self.model_args.use_lora:
            unet.requires_grad_(False)

            # For mixed precision training we cast the text_encoder and vae weights to half-precision
            # as these models are only used for inference, keeping weights in full precision is not required.
            weight_dtype = torch.float32
            if self.accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16

            # Move unet, vae and text_encoder to device and cast to weight_dtype
            unet.to(self.accelerator.device, dtype=weight_dtype)
            vae.to(self.accelerator.device, dtype=weight_dtype)
            text_encoder.to(self.accelerator.device, dtype=weight_dtype)

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

        else:   
            # Create EMA for the unet.
            self.ema_unet = model.ema_unet

        if self.finetuner_args.enable_xformers_memory_efficient_attention:
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

        if self.model_args.use_lora:

            lora_layers = AttnProcsLayers(unet.attn_processors)

        else:
            
            if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
                # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
                self.accelerator.register_save_state_pre_hook(self.save_model_hook)
                self.accelerator.register_load_state_pre_hook(self.load_model_hook)
            
            if self.finetuner_args.gradient_checkpointing:
                unet.enable_gradient_checkpointing()

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
            unet.parameters(),
            lr=self.finetuner_args.learning_rate,
            betas=(self.finetuner_args.adam_beta1, self.finetuner_args.adam_beta2),
            weight_decay=self.finetuner_args.adam_weight_decay,
            eps=self.finetuner_args.adam_epsilon,
        )

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if self.finetuner_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                self.finetuner_args.dataset_name,
                self.finetuner_args.dataset_config_name,
                cache_dir=self.finetuner_args.cache_dir,
            )
        else:
            data_files = {}
            if self.finetuner_args.train_data_dir is not None:
                data_files["train"] = os.path.join(self.finetuner_args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=self.finetuner_args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(self.finetuner_args.dataset_name, None)
        if self.finetuner_args.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = self.finetuner_args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{self.finetuner_args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        self.image_column=image_column

        if self.finetuner_args.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = self.finetuner_args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{self.finetuner_args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Preprocessing the datasets.
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(self.finetuner_args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.finetuner_args.resolution) if self.finetuner_args.center_crop else transforms.RandomCrop(self.finetuner_args.resolution),
                transforms.RandomHorizontalFlip() if self.finetuner_args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        with self.accelerator.main_process_first():
            if self.finetuner_args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=self.finetuner_args.seed).select(range(self.finetuner_args.max_train_samples))
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(self.preprocess_train)

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=self.finetuner_args.train_batch_size,
            num_workers=self.finetuner_args.dataloader_num_workers,
        )

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
            unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

            if self.model_args.use_ema:
                self.ema_unet.to(self.accelerator.device)

            # For mixed precision training we cast the text_encoder and vae weights to half-precision
            # as these models are only used for inference, keeping weights in full precision is not required.
            weight_dtype = torch.float32
            if self.accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif self.accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16

            # Move text_encode and vae to gpu and cast to weight_dtype
            text_encoder.to(self.accelerator.device, dtype=weight_dtype)
            vae.to(self.accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.finetuner_args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.finetuner_args.max_train_steps = self.finetuner_args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.finetuner_args.num_train_epochs = math.ceil(self.finetuner_args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        
        
        if self.accelerator.is_main_process:
            if self.model_args.use_lora:
                self.accelerator.init_trackers("text2image-fine-tune", config=vars(self.finetuner_args))
            else:
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
        if self.finetuner_args.resume_from_checkpoint:
            if self.finetuner_args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.finetuner_args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.finetuner_args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.finetuner_args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.finetuner_args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.finetuner_args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.finetuner_args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * self.finetuner_args.gradient_accumulation_steps)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, self.finetuner_args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, self.finetuner_args.num_train_epochs):
            unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if self.finetuner_args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % self.finetuner_args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

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
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if self.finetuner_args.input_perturbation:
                        noisy_latents = self.noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self.finetuner_args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = self.compute_snr(timesteps)
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
                        self.accelerator.clip_grad_norm_(unet.parameters(), self.finetuner_args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.model_args.use_ema and (self.model_args.use_lora is False):
                        self.ema_unet.step(unet.parameters())
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

            if self.accelerator.is_main_process:
                if self.finetuner_args.validation_prompts is not None and epoch % self.finetuner_args.validation_epochs == 0:  
                    if self.model_args.use_lora:
                        logger.info(
                            f"Running validation... \n Generating {self.finetuner_args.num_validation_images} images with prompt:"
                            f" {self.finetuner_args.validation_prompt}."
                        )
                        # create pipeline
                        pipeline = DiffusionPipeline.from_pretrained(
                            self.model_args.pretrained_model_name_or_path,
                            unet=self.accelerator.unwrap_model(unet),
                            revision=self.model_args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(self.accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        generator = torch.Generator(device=self.accelerator.device).manual_seed(self.finetuner_args.seed)
                        images = []
                        for _ in range(self.finetuner_args.num_validation_images):
                            images.append(
                                pipeline(self.finetuner_args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                            )

                        for tracker in self.accelerator.trackers:
                            if tracker.name == "tensorboard":
                                np_images = np.stack([np.asarray(img) for img in images])
                                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                            if tracker.name == "wandb":
                                tracker.log(
                                    {
                                        "validation": [
                                            wandb.Image(image, caption=f"{i}: {self.finetuner_args.validation_prompt}")
                                            for i, image in enumerate(images)
                                        ]
                                    }
                                )

                            del pipeline
                            torch.cuda.empty_cache()
                    else:
                        if self.model_args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            self.ema_unet.store(unet.parameters())
                            self.ema_unet.copy_to(unet.parameters())
                        self.deepspeed_zero_init_disabled_context_managerlog_validation(
                            vae,
                            text_encoder,
                            self.tokenizer,
                            unet,
                            self.finetuner_args,
                            self.accelerator,
                            weight_dtype,
                            global_step,
                        )
                        if self.model_args.use_ema:
                            # Switch back to the original UNet parameters.
                            self.ema_unet.restore(unet.parameters())

        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.model_args.use_lora:
                unet = unet.to(torch.float32)
                unet.save_attn_procs(self.finetuner_args.output_dir)

                if self.finetuner_args.push_to_hub:
                    self.save_model_card(
                        repo_id,
                        images=images,
                        base_model=self.model_args.pretrained_model_name_or_path,
                        dataset_name=self.finetuner_args.dataset_name,
                        repo_folder=self.finetuner_args.output_dir,
                    )
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=self.finetuner_args.output_dir,
                        commit_message="End of training",
                        ignore_patterns=["step_*", "epoch_*"],
                    )


            else:
                unet = self.accelerator.unwrap_model(unet)
                if self.model_args.use_ema:
                    self.ema_unet.copy_to(unet.parameters())

                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_args.pretrained_model_name_or_path,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    revision=self.model_args.revision,
                )
                pipeline.save_pretrained(self.finetuner_args.output_dir)

                if self.finetuner_args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=self.finetuner_args.output_dir,
                        commit_message="End of training",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
               
        if self.model_args.use_lora: 
                # Final inference
                # Load previous pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    self.model_args.pretrained_model_name_or_path, revision=self.model_args.revision, torch_dtype=weight_dtype
                )
                pipeline = pipeline.to(self.accelerator.device)

                # load attention processors
                pipeline.unet.load_attn_procs(self.finetuner_args.output_dir)

                # run inference
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.finetuner_args.seed)
                images = []
                for _ in range(self.finetuner_args.num_validation_images):
                    images.append(pipeline(self.finetuner_args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

                if self.accelerator.is_main_process:
                    for tracker in self.accelerator.trackers:
                        if tracker.name == "tensorboard":
                            np_images = np.stack([np.asarray(img) for img in images])
                            tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                        if tracker.name == "wandb":
                            tracker.log(
                                {
                                    "test": [
                                        wandb.Image(image, caption=f"{i}: {self.finetuner_args.validation_prompt}")
                                        for i, image in enumerate(images)
                                    ]
                                }
                        )

        self.accelerator.end_training()
