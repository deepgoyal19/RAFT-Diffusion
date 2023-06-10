#!/usr/bin/env python
# coding=utf-8
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

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
# from lmflow_diffusion.args import FinetunerArguments, ModelArguments


logger = logging.getLogger(__name__)



class DiffusionModel:

    def __init__(self, model_args, *args, **kwargs):

        self.model_args=model_args

        if not self.model_args.use_lora:
                if self.model_args.non_ema_revision is not None:
                    deprecate(
                        "non_ema_revision!=None",
                        "0.15.0",
                        message=(
                            "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                            " use `--variant=non_ema` instead."
                        ),
                    )
    

        # Load scheduler, tokenizer and models.
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_args.pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_args.pretrained_model_name_or_path, subfolder="tokenizer", revision=self.model_args.revision
        )

        if self.model_args.use_lora:
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_args.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.model_args.revision
            )
            self.vae = AutoencoderKL.from_pretrained(self.model_args.pretrained_model_name_or_path, subfolder="vae", revision=self.model_args.revision)
        else: 
            # with ContextManagers(self.deepspeed_zero_init_disabled_context_manager()):
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_args.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.model_args.revision
            )
            self.vae = AutoencoderKL.from_pretrained(
                self.model_args.pretrained_model_name_or_path, subfolder="vae", revision=self.model_args.revision
            )

        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_args.pretrained_model_name_or_path, subfolder="unet", revision=self.model_args.non_ema_revision
        )        
        # freeze parameters of models to save more memory
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.model_args.use_lora:
            self.unet.requires_grad_(False)

        if self.model_args.use_ema and (self.model_args.use_lora is False) :
            self.ema_unet = UNet2DConditionModel.from_pretrained(
            self.model_args.pretrained_model_name_or_path, subfolder="unet", revision=self.model_args.revision
            )
            self.ema_unet = EMAModel(self.ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.ema_unet.config)
    
    def to_device(self, device, weight_dtype):
        self.unet.to(device, dtype=weight_dtype)
        self.vae.to(device, dtype=weight_dtype)
        self.text_encoder.to(device, dtype=weight_dtype)


    def save(self, output_dir):

        if self.model_args.use_lora:
            self.unet = self.unet.to(torch.float32)
            self.unet.save_attn_procs(output_dir)
        
        else:
            self.unet = self.accelerator.unwrap_model(self.unet)
            if self.model_args.use_ema:
                self.ema_unet.copy_to(self.unet.parameters())

            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_args.pretrained_model_name_or_path,
                text_encoder=self.text_encoder,
                vae=self.vae,
                unet=self.unet,
                revision=self.model_args.revision,
            )
            pipeline.save_pretrained(output_dir)        


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

    def push_to_hub(self, hub_model_id, output_dir, hub_token, images, dataset_name):

        repo_id = create_repo(
            repo_id=hub_model_id or Path(output_dir).name, exist_ok=True, token=hub_token
        ).repo_id        

        if self.model_args.use_lora:
            self.save_model_card(
                repo_id,
                images=images,
                base_model=self.model_args.pretrained_model_name_or_path,
                dataset_name=dataset_name,
                repo_folder=output_dir,
            )
        upload_folder(
            repo_id=repo_id,
            folder_path=output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )       

    def inference(self):
        pass

    def resume_from_path(self, resume_from_checkpoint, output_dir, gradient_accumulation_steps, num_update_steps_per_epoch):
        
        if resume_from_checkpoint:
            if resume_from_checkpoint != "latest":
                path = os.path.basename(resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                )
# Deepanshu Comment: Please do check the return values
                resume_from_checkpoint = None
                resume_step = None
                global_step = 0
                first_epoch = 0
                return resume_from_checkpoint, global_step, first_epoch, resume_step
            
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

                return resume_from_checkpoint, global_step, first_epoch, resume_step