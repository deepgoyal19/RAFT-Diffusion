#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""

import logging
import math
import os
import accelerate
import datasets
import concurrent
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import  DiffusionPipeline, UNet2DConditionModel
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
import diffusers
from diffusers import DPMSolverMultistepScheduler
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
from PIL import Image
import shutil
if is_wandb_available():
    import wandb

logger = logging.getLogger(__name__)

class Inferencer:
    def init(self, inference_args):
        self.inference_args= inference_args

    def inference(self):   
        self.accelerator=Accelerator()

        pipeline = DiffusionPipeline.from_pretrained(
            self.model_args.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype)
        
        images=pipeline( 
        self.inference_args.prompts,
        num_images_per_prompt=self.raft_args.num_images_per_prompt,
        width=self.inference_args.resolution,
        height=self.inference_args.resolution,
        num_inference_steps=self.inference_args.num_inference_steps,
        guidance_scale=self.inference_args.guidance_scale).images
        
        # delete pipeline
        del pipeline 

        if self.infernce_args.grid:
            assert len(images) == self.inference_args.rows*self.inference_args.cols
            w, h = images[0].size
            grid = Image.new('RGB', size=(self.inference_args.cols*w, self.inference_args.rows*h))
            
            for i, img in enumerate(images):
                grid.paste(img, box=(i%self.inference_args.cols*w, i//self.inference_args.cols*h))
            
            if self.inference_args.save_image_dir:
                grid.save(f"{self.inference_args.save_image_dir}/output.{self.inference_args.save_format}")
            return grid
        else:
            if self.inference_args.save_image_dir:
                for len_images in range(len(images)):
                    images[len_images].save(f"{self.inference_args.save_image_dir}/{len_images}.{self.inference_args.save_format}")
            return images