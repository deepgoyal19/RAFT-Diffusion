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
    
    def to_device(self,device,weight_dtype):
        self.unet.to(device, dtype=weight_dtype)
        self.vae.to(device, dtype=weight_dtype)
        self.text_encoder.to(device, dtype=weight_dtype)


    def save(self):
        pass

    def resume_from_path(self):
        pass