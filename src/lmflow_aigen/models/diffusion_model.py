#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""


import logging
import os
from pathlib import Path
import torch
from huggingface_hub import create_repo, upload_folder
from transformers import CLIPTextModel, CLIPTokenizer, AutoProcessor, AutoModel
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, deprecate
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
import numpy as np 
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
import torch.nn as nn
import open_clip
import clip
import importlib
if is_wandb_available():
    import wandb

from copy import deepcopy


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
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_args.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.model_args.revision
        )
        self.vae = AutoencoderKL.from_pretrained(self.model_args.pretrained_model_name_or_path, subfolder="vae", revision=self.model_args.revision)
        
        if self.model_args.use_lora:
            self.unet = UNet2DConditionModel.from_pretrained(
                self.model_args.pretrained_model_name_or_path, subfolder="unet", revision=self.model_args.revision
            )   
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                self.model_args.pretrained_model_name_or_path, subfolder="unet", revision=self.model_args.non_ema_revision
            )        
        # freeze parameters of models to save more memory
        if self.model_args.use_lora:
            self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if self.model_args.use_ema and (self.model_args.use_lora is False) :
            self.ema_unet = UNet2DConditionModel.from_pretrained(
            self.model_args.pretrained_model_name_or_path, subfolder="unet", revision=self.model_args.revision
            )
            self.ema_unet = EMAModel(self.ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.ema_unet.config)
    
    def to_device(self, device):
        if self.model_args.use_lora:
            self.unet=self.unet.to(device, dtype=self.weight_dtype)
        self.vae = self.vae.to(device, dtype=self.weight_dtype)
        self.text_encoder = self.text_encoder.to(device, dtype=self.weight_dtype)
        if self.model_args.use_ema and (self.model_args.use_lora == False):
            self.ema_unet = self.ema_unet.to(device)


    def save(self, output_dir, accelerator):

        if self.model_args.use_lora:
            self.unet = self.unet.to(torch.float32)
            self.unet.save_attn_procs(output_dir)
        
        else:
            self.unet = accelerator.unwrap_model(self.unet)
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

    def push_to_hub(self, hub_model_id, output_dir, hub_token, dataset_name):

        repo_id = create_repo(
            repo_id=hub_model_id or Path(output_dir).name, exist_ok=True, token=hub_token
        ).repo_id

        if self.model_args.use_lora:
            self.save_model_card(
                repo_id,
                images=self.validation_images,
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


    def resume_from_path(self, resume_from_checkpoint, output_dir, gradient_accumulation_steps, num_update_steps_per_epoch,  accelerator):
        
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
                accelerator.print(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                )
 
                resume_from_checkpoint = None
                resume_step = None
                global_step = 0
                first_epoch = 0
                return resume_from_checkpoint, global_step, first_epoch, resume_step
            
            else:
                accelerator.load_state(os.path.join(output_dir, path))
                accelerator.print(f"Resuming from checkpoint {path}")
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)
                # raft_epoch = int(path.split("-")[1]) / max_training_step
                # raft_epochs+= raft_epoch
                # max_training_step+= resume_global_step
                return resume_from_checkpoint, global_step, first_epoch, resume_step
        # else:
        #     resume_step = None
        #     global_step = 0
        #     first_epoch = 0
        #     return resume_from_checkpoint, global_step, first_epoch, resume_step

    def set_weight_dtype(self, mixed_precision):
        self.weight_dtype = torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16


    def set_lora_attn_proccessor_to_unet(self):
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
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,rank=self.model_args.rank)

        self.unet.set_attn_processor(lora_attn_procs)


    def use_xformers(self):
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            self.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
                        
    def log_validation(self, args, accelerator, epoch, resolution, log_type):
        if accelerator.is_main_process:
            if self.model_args.use_ema and (self.model_args.use_lora == False):
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:  
                logger.info(
                    f"Running validation... \n Generating images with prompt:"
                    f" {args.validation_prompts}."
                )
                
                pipeline = self.load_model_pipeline(accelerator)
                
                pipeline.set_progress_bar_config(disable=True)

                if args.enable_xformers_memory_efficient_attention:
                    pipeline.enable_xformers_memory_efficient_attention()

                if args.seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                self.validation_images = []

                for i in range(len(args.validation_prompts)):
                    if self.model_args.use_lora:
                        self.validation_images.append(pipeline(args.validation_prompts[i], num_inference_steps=50, generator=generator, width=resolution, height=resolution).images[0])
                    else:
                        with torch.autocast(device_type='cuda'):
                            self.validation_images.append(pipeline(args.validation_prompts[i], num_inference_steps=50, generator=generator, width=resolution, height=resolution).images[0])

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in self.validation_images])
                        if log_type=='validation':
                            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                        else:
                            tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                    elif tracker.name == "wandb":
                        if log_type=='validation':
                            tracker.log(
                                {
                                    "validation": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                                        for i, image in enumerate(self.validation_images)
                                    ]
                                }
                            )
                        else:
                            tracker.log(
                                {
                                    "test": [
                                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                                        for i, image in enumerate(self.validation_images)
                                    ]
                                }
                            )
                    else:
                        logger.warn(f"image logging not implemented for {tracker.name}")
                del pipeline
                torch.cuda.empty_cache()
            else:
                self.validation_images=[]

                if self.model_args.use_ema and (self.model_args.use_lora == False):
                    # Switch back to the original UNet parameters.
                    self.ema_unet.restore(self.unet.parameters())
    
    def load_model_pipeline(self, accelerator):
        if self.model_args.use_lora:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(self.unet),
                revision=self.model_args.revision,
                torch_dtype=self.weight_dtype,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_args.pretrained_model_name_or_path,
                vae=accelerator.unwrap_model(self.vae),
                text_encoder=accelerator.unwrap_model(self.text_encoder),
                tokenizer=self.tokenizer,
                unet=accelerator.unwrap_model(self.unet),
                safety_checker=None,
                revision=self.model_args.revision,
                torch_dtype=self.weight_dtype,
            )
        pipeline.to(accelerator.device)
        return pipeline

    def final_inference(self, args, accelerator, epoch, resolution):
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_args.pretrained_model_name_or_path, revision=self.model_args.revision, torch_dtype=self.weight_dtype
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(args.output_dir)

        if args.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()

        # run inference
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator = generator.manual_seed(args.seed)
        images = []

        for i in range(len(args.validation_prompts)):
            images.append(pipeline(args.validation_prompts[i], num_inference_steps=50, generator=generator, width=resolution, height=resolution).images[0])
        
        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        del pipeline
        torch.cuda.empty_cache()

    def load_score_model(self, score_model_pretrained_name_or_path, score_model_name,device, pickscore_processor_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        self.score_model_name=score_model_name
        self.device=device
        if self.score_model_name=="aesthetic":  
            self.score_model, _, self.score_preprocess = open_clip.create_model_and_transforms(score_model_pretrained_name_or_path, pretrained='openai', device=device)
            clip_model=score_model_pretrained_name_or_path.lower().replace('-','_')
            self.score_amodel= self.get_aesthetic_model(clip_model).eval().to(device)
        elif self.score_model_name=='clip':
            self.score_model, self.score_preprocess = clip.load(score_model_pretrained_name_or_path, device=device)
        elif self.score_model_name=="pick":
            self.score_preprocess = AutoProcessor.from_pretrained(pickscore_processor_name_or_path)
            self.score_model = AutoModel.from_pretrained(score_model_pretrained_name_or_path).eval().to(device)
        else:
            raise ValueError("Score model should be either 'aesthetic', 'clip', or 'pick'.")

    def get_score(self, image, text=None):
        if  self.score_model_name=="aesthetic":   
            image = self.score_preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.score_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                score = float(self.score_amodel(image_features))
            return score
        elif self.score_model_name=='clip':
            image = self.score_preprocess(image).unsqueeze(0).to(self.device)
            text = clip.tokenize(text).to(self.device)
            with torch.no_grad():
                image_features = self.score_model.encode_image(image)
                text_features = self.score_model.encode_text(text)
                
                logits_per_image, logits_per_text = self.score_model(image, text)
                score = float(logits_per_image)
                return score
        elif self.score_model_name=='pick':                
            # preprocess
            image_inputs = self.score_preprocess(
                images=image,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            
            text_inputs = self.score_preprocess(
                text=text,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                # embed
                image_embs =  self.score_model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            
                text_embs = self.score_model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            
                # score
                score = float(self.score_model.logit_scale.exp() * (text_embs @ image_embs.T)[0])
                
                # get probabilities if you have multiple images to choose from
                # score = torch.softmax(scores, dim=-1)                
            return score

    def get_aesthetic_model(self, clip_model="vit_l_14"):
        """load the aethetic model"""
        home = expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
            )
            print(url_model)
            urlretrieve(url_model, path_to_model)
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError()
        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m
    
    def preprocess_image(self, images, text):
        scores=[]
        for image in images:
            scores.append(self.get_score(image,text))
            torch.cuda.empty_cache()
        max_score=max(scores)
        return [max_score,scores.index(max_score)]

    def load_pipeline_scheduler(self, pipeline):
        scheduler=self.pipeline_scheduler.from_config(pipeline.scheduler.config)
        return scheduler      
    
    def import_pipeline_scheduler(self, scheduler):     
        self.pipeline_scheduler = getattr(importlib.import_module('diffusers'), scheduler)

