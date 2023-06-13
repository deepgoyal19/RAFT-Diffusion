#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""


import logging
import os
from pathlib import Path
import accelerate
import torch
from huggingface_hub import create_repo, upload_folder
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, deprecate
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
import numpy as np 

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
    
    def to_device(self, device):
        if self.model_args.use_lora:
            self.unet.to(device, dtype=self.weight_dtype)
        self.vae.to(device, dtype=self.weight_dtype)
        self.text_encoder.to(device, dtype=self.weight_dtype)
        if self.model_args.use_ema and (self.model_args.use_lora == False):
            self.ema_unet.to(device)


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
                images=self.images,
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


    def resume_from_path(self, resume_from_checkpoint, output_dir, gradient_accumulation_steps, num_update_steps_per_epoch, accelerator):
        
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
    # Deepanshu Comment: Please do check the return values
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

                return resume_from_checkpoint, global_step, first_epoch, resume_step

    def set_weight_dtype(self, mixed_precision):
        self.weight_dtype = torch.float32
        if mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def set_lora_attn_proccessor_to_unet(self):
        self.unet.requires_grad_(False)

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

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

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


    def final_inference(self, seed, output_dir, num_validation_images, validation_prompts, epoch, accelerator):
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_args.pretrained_model_name_or_path, revision=self.model_args.revision, torch_dtype=self.weight_dtype
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(output_dir)

        # run inference
        generator = torch.Generator(device=accelerator.device).manual_seed(seed)
        images = []
        for _ in range(num_validation_images):
            images.append(pipeline(validation_prompts, num_inference_steps=30, generator=generator).images[0])

        if accelerator.is_main_process:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {validation_prompts}")
                                for i, image in enumerate(images)
                            ]
                        }
                )
                        
    def log_validation(self, args, accelerator, epoch):
        if accelerator.is_main_process:
            self.images = []
            if self.model_args.use_ema and (self.model_args.use_lora == False):
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            if args.validation_prompts is not None and epoch % args.validation_epochs == 0:  
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompts}."
                )
                if self.model_args.use_lora:
                    pipeline = DiffusionPipeline.from_pretrained(
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
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=False)

                # if args.enable_xformers_memory_efficient_attention:
                #     pipeline.enable_xformers_memory_efficient_attention()

                if args.seed is None:
                    generator = None
                else:
                    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                self.images = []
                for i in range(len(args.validation_prompts)):
                    if self.model_args.use_lora:
                        with torch.autocast(device_type='cuda'):
                            self.images.append(pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0])
                    else:
                        self.images.append(pipeline(args.validation_prompts[i], num_inference_steps=30, generator=generator).images[0])

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in self.images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    elif tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                                    for i, image in enumerate(self.images)
                                ]
                            }
                        )
                    else:
                        logger.warn(f"image logging not implemented for {tracker.name}")

                del pipeline
                torch.cuda.empty_cache()

            if self.model_args.use_ema and (self.model_args.use_lora == False):
                # Switch back to the original UNet parameters.
                self.ema_unet.restore(self.unet.parameters())