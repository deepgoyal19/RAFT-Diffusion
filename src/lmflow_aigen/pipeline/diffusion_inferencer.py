#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import  StableDiffusionPipeline
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from PIL import Image


class DiffusionInferencer:
    def __init__(self, inference_args):
        self.inference_args = inference_args
        self.accelerator=Accelerator()

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.generator = torch.Generator(device=self.accelerator.device)

        if self.inference_args.save_image_dir is not None:
                os.makedirs(self.inference_args.save_image_dir, exist_ok=True)
        # if self.inference_args.seed is not None:
        #     self.generator = self.generator.manual_seed(self.inference_args.seed)

    def inference(self):   
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.inference_args.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype)
        
        pipeline.to(self.accelerator.device)

        if self.inference_args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        if self.inference_args.use_lora :
            images=pipeline( 
                    self.inference_args.prompt,
                    num_images_per_prompt=self.inference_args.num_images_per_prompt,
                    width=self.inference_args.width,
                    height=self.inference_args.height,
                    num_inference_steps=self.inference_args.num_inference_steps,
                    generator=self.generator).images
        else:
            with torch.autocast(device_type='cuda'):
                images=pipeline( 
                    self.inference_args.prompt,
                    num_images_per_prompt=self.inference_args.num_images_per_prompt,
                    width=self.inference_args.width,
                    height=self.inference_args.height,
                    num_inference_steps=self.inference_args.num_inference_steps,
                    generator=self.generator).images
        
        # delete pipeline
        del pipeline 

        if self.inference_args.grid:
            rows=int(len(images)/self.inference_args.num_images_per_prompt)
            cols=self.inference_args.num_images_per_prompt
            assert len(images) == rows*cols
            w, h = images[0].size
            grid = Image.new('RGB', size=(cols*w, rows*h))
        
            for i, img in enumerate(images):
                grid.paste(img, box=(i%cols*w, i//cols*h))
            
            if self.inference_args.save_image_dir:
                grid.save(f"{self.inference_args.save_image_dir}/output.{self.inference_args.save_format}")
            return grid
        else:
            if self.inference_args.save_image_dir:
                for len_images in range(len(images)):
                    images[len_images].save(f"{self.inference_args.save_image_dir}/{len_images}.{self.inference_args.save_format}")
            return images