#!/usr/bin/env python
# coding=utf-8
"""This script defines dataclasses: ModelArguments, InferenceAguments and DatasetArguments,
that contain the arguments for the model and dataset used in training.


"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

# from lmflow.args import ModelArguments,DatasetArguments, EvaluatorArguments, BenchmarkingArguments

from dataclasses import dataclass, field
from typing import Optional
from diffusers import StableDiffusionPipeline
import torch

@dataclass
class FinetunerArguments:

    pretrained_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    
    revision: Optional[str] = field(
        default=None, metadata={"help": "Revision of pretrained model identifier from huggingface.co/models."})
    
    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the Dataset (from the HuggingFace hub to train on."
                    " It can also be a path pointing to a local copy of a dataset in your filesystem,"
                    " or to a folder containing files that 🤗 Datasets can understand."
        })
    
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The config of the Dataset, leave as None if there's only one config."})
    
    train_data_dir: Optional[str] = field(
        default=None, metadata={
            "help": "A folder containing the training data. Folder contents must follow the structure described in"
                    " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
                    " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        })
    
    image_column: str = field(
        default="image", metadata={"help": "The column of the dataset containing an image."})
    
    caption_column: str = field(
        default="text", metadata={"help": "The column of the dataset containing a caption or a list of captions."})
    
    validation_prompt: Optional[str] = field(
        default=None, metadata={"help": "A prompt that is sampled during training for inference."})
    
    num_validation_images: int = field(
        default=4, metadata={"help": "Number of images that should be generated during validation with `validation_prompt`."})
    
    validation_epochs: int = field(
        default=1, metadata={"help": "Run fine-tuning validation every X epochs."})
    
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."})
    
    output_dir: str = field(
        default="sd-model-finetuned-lora", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory where the downloaded models and datasets will be stored."})
    
    seed: Optional[int] = field(
        default=None, metadata={"help": "A seed for reproducible training."})
    
    resolution: int = field(
        default=512, metadata={"help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"})
    
    center_crop: bool = field(
        default=False, metadata={"help": "Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping."})
    
    random_flip: bool = field(
        default=False, metadata={"help": "Whether to randomly flip images horizontally"})
    
    train_batch_size: int = field(
        default=16, metadata={"help": "Batch size (per device for the training dataloader."})
    
    num_train_epochs: int = field(
        default=100, metadata={"help": "Number of training epochs."})
    
    max_train_steps: Optional[int] = field(
        default=None, metadata={"help": "Total number of training steps to perform. Overrides num_train_epochs if provided."})
    
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass."})
    
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Initial learning rate to use."})
    
    scale_lr: bool = field(
        default=False, metadata={"help": "Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."})
    
    lr_scheduler: str = field(
        default="constant", metadata={"help": "The scheduler type to use."})
    
    lr_warmup_steps: int = field(
        default=500, metadata={"help": "Number of steps for the warmup in the lr scheduler."})
    
    snr_gamma: Optional[float] = field(
        default=None, metadata={"help": "SNR weighting gamma to be used if rebalancing the loss."})
    
    use_8bit_adam: bool = field(
        default=False, metadata={"help": "Whether or not to use 8-bit Adam from bitsandbytes."})
    
    allow_tf32: bool = field(
        default=False, metadata={"help": "Whether or not to allow TF32 on Ampere GPUs."})
    
    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of subprocesses to use for data loading."})
    
    adam_beta1: float = field(
        default=0.9, metadata={"help": "The beta1 parameter for the Adam optimizer."})
    
    adam_beta2: float = field(
        default=0.999, metadata={"help": "The beta2 parameter for the Adam optimizer."})
    
    adam_weight_decay: float = field(
        default=1e-2, metadata={"help": "Weight decay to use."})
    
    adam_epsilon: float = field(
        default=1e-08, metadata={"help": "Epsilon value for the Adam optimizer."})
    
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm."})
    
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to push the model to the Hub."})
    
    hub_token: Optional[str] = field(
        default=None, metadata={"help": "The token to use to push to the Model Hub."})
    
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."})
    
    logging_dir: str = field(
        default="logs", metadata={"help": "TensorBoard log directory."})
    
    mixed_precision: Optional[str] = field(
        default=None, metadata={"help": "Use mixed precision training using Apex. Options: '00', '01', '02', '03'."})
    
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to a checkpoint from which to resume training."})
    
    fp16_backend: str = field(
        default="auto", metadata={"help": "Mixed precision backend to use (auto, apex, amp, or torch."})
    
    fp16_opt_level: str = field(
        default="O1", metadata={"help": "Mixed precision optimization level (00, 01, 02, 03."})
    
    disable_tqdm: bool = field(
        default=False, metadata={"help": "Disable tqdm progress bars."})
    
    save_steps: int = field(
        default=1000, metadata={"help": "Number of steps between saving checkpoints."})
    
    save_total_limit: Optional[int] = field(
        default=None, metadata={"help": "If a value is passed, will limit the total amount of checkpoints."})
    
    eval_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the eval dataset to use."})
    
    tracker_project_name: str = field(
        default="text2image-fine-tune", metadata={"help": "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"})

@dataclass 
class ModelArguments:

    pretrained_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})

    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use lora."})
    
    torch_dtype: Optional[str] = field(default=None,metadata={"help": ("Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed,the "
        "dtype will be automatically derived from the model's weights."), "choices": ["auto", "bfloat16", "float16", "float32"]})


@dataclass
class InferenceArguments:  

    prompt: Optional[Union[str, List[str]]] = field(
        default=None, metadata={"help": "The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.instead."})

    height: Optional[int] = field(
        default=None, metadata={"help": "The height in pixels of the generated image."})


    width: Optional[int] = field(
        default=None, metadata={"help": "The height in pixels of the generated image."})


    num_inference_steps: int = field(
        default=50, metadata={"help": "The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."})


    guidance_scale: float = field(
        default=7.5, metadata={"help": '''Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).`guidance_scale` is defined as `w` of equation 2. of [ImagenPaper](https://arxiv.org/pdf/2205.11487.pdf). 
                Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.'''})

    negative_prompt: Optional[Union[str, List[str]]] = field(
        default=None, metadata={"help": '''The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).'''})


    num_images_per_prompt: Optional[int] = field(
        default=1, metadata={"help": "The number of images to generate per prompt."})


    eta: float = field(
        default=0.0, metadata={"help": '''Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.'''})


    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = field(
        default=None, metadata={"help": ''' One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.'''})


    latents: Optional[torch.FloatTensor] = field(
        default=None, metadata={"help": '''Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.'''})


    prompt_embeds: Optional[torch.FloatTensor] = field(
        default=None, metadata={"help": '''Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.'''})


    negative_prompt_embeds: Optional[torch.FloatTensor] = field(
        default=None, metadata={"help": '''Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. 
            If not provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.'''})


    output_type: Optional[str] = field(
        default="pil", metadata={"help": '''The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.'''})

    return_dict: bool = field(
        default=True, metadata={"help": "Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a plain tuple."})

    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = field(
        default=None, metadata={"help": '''A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.'''})


    callback_steps: int = field(
        default=1, metadata={"help": '''The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.'''})


    cross_attention_kwargs: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": '''A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).'''})


@dataclass
class RaftAlignterArguments:

    topk: int = field(
        default=1, metadata={"help": ""})
