#!/usr/bin/env python
# coding=utf-8
"""This script defines dataclasses: ModelArguments, InferenceArguments, FinetunerArguments and DatasetArguments,
that contain the arguments for the model and dataset used in training.


"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import torch

@dataclass
class FinetunerArguments:
    last_epoch: bool= field(
        default=True, metadata={"help":""}
    )
    prediction_type: str= field(
        default=None, metadata={"help": "The prediction_type that shall be used for training."
                                "Choose between 'epsilon' or 'v_prediction' or leave `None`. "
                                "If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",})
    validation_epochs: int = field(
        default=1, metadata={"help": "Run fine-tuning validation every X epochs."})
    
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."})
    
    output_dir: str = field(
        default="sd-model-finetuned-lora", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    
    
    seed: Optional[int] = field(
        default=None, metadata={"help": "A seed for reproducible training."})
    
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
    
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Path to a checkpoint from which to resume training."})
    
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

    checkpoints_total_limit: Optional[int] = field(
        default= None, metadata= {"help":"Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"})
    
    validation_prompts: Optional[List[str]] = field(
        default= None, metadata= {"help": "A set of prompts evaluated every `validation_epochs` and logged to `report_to`."})
    
    
    enable_xformers_memory_efficient_attention: bool= field(
        default= False, metadata={"help": "Whether or not to use xformers." })
    
    
    noise_offset: float= field(
        default=0, metadata={"help": "The scale of noise offset"})
    
    input_perturbation: float= field(
        default=0, metadata={"help": "The scale of input perturbation. Recommended 0.1."})    
    
    checkpointing_steps: int= field(
        default=500, metadata={"help":"Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."})
    
    report_to: str = field(
        default= "tensorboard", metadata= {"help": 'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'})
    
    
@dataclass 
class ModelArguments:

    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use lora."})
    
    # torch_dtype: Optional[str] = field(default=None,metadata={"help": ("Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed,the "
    #     "dtype will be automatically derived from the model's weights."), "choices": ["auto", "bfloat16", "float16", "float32"]})

    pretrained_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    
    revision: Optional[str] = field(
        default=None, metadata={"help": "Revision of pretrained model identifier from huggingface.co/models."})

    non_ema_revision: Optional[str]= field(
        default=None, metadata= {"help": "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."})
    
    use_ema: bool= field(
        default=False, metadata={"help": "Whether to use EMA model."})

    rank: int=field(
        default=4, metadata={'help':"The dimension of the LoRA update matrices."})
    

@dataclass
class InferenceArguments:  
    pretrained_model_name_or_path: Optional[str]= field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})

    prompt: Union[str, List[str]] = field(
        default='Raft', metadata={"help": "The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.instead."})

    height: int = field(
        default=512, metadata={"help": "The height in pixels of the generated image."})


    width: int = field(
        default=512, metadata = {"help": "The height in pixels of the generated image."})


    num_inference_steps: int = field(
        default=50, metadata={"help": "The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."})


    guidance_scale: float = field(
        default=7.5, metadata={"help": '''Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).`guidance_scale` is defined as `w` of equation 2. of [ImagenPaper](https://arxiv.org/pdf/2205.11487.pdf). 
                Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.'''})

    # negative_prompt: Optional[Union[str, List[str]]] = field(
    #     default=None, metadata={"help": '''The prompt or prompts not to guide the image generation. If not defined, one has to pass
    #             `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
    #             less than `1`).'''})


    num_images_per_prompt: Optional[int] = field(
        default=1, metadata={"help": "The number of images to generate per prompt."})

    save_format: str=field(
        default= "png", metadata={"help":"File format of save images"
                                  "Reference: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html", 
                                  "choices":["png", "jpeg", "ppm", "gif", "tif", "bmp"]})
    
    seed: Optional[int]= field(
        default=None, metadata={"help":"Used for generating similar images"})

    grid: bool= field(
        default=False,metadata={"help":"Save images as a grid"})

    save_image_dir: Optional[str] = field(
        default=None, metadata={"help":"Path to save generated images"})

    enable_xformers_memory_efficient_attention: bool= field(
        default= False, metadata={"help": "Whether or not to use xformers." })
    
    use_lora: bool= field(
        default=False, metadata={"help":"Whethere or not to use lora"}
    )
    
@dataclass
class DatasetArguments:

    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the Dataset (from the HuggingFace hub to train on."
                    " It can also be a path pointing to a local copy of a dataset in your filesystem,"
                    " or to a folder containing files that 🤗 Datasets can understand."})
    
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The config of the Dataset, leave as None if there's only one config."})
    
    train_data_dir: Optional[str] = field(
        default=None, metadata={
            "help": "A folder containing the training data. Folder contents must follow the structure described in"
                    " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
                    " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."})
        
    #Deepanshu Comments: Should this be in finetunerArgs or DatasetArgs
    dataloader_num_workers: int = field(
        default=0, metadata={"help": "Number of subprocesses to use for data loading."})

    resolution: int = field(
        default=512, metadata={"help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"})
    
    center_crop: bool = field(
        default=False, metadata={"help": "Whether to center crop the input images to the resolution."
                                 "If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping."})
    
    random_flip: bool = field(
        default=False, metadata={"help": "Whether to randomly flip images horizontally"})

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory where the downloaded models and datasets will be stored."})

    train_batch_size: int = field(
        default=16, metadata={"help": "Batch size (per device for the training dataloader."})
    
    overrode_init_dataset: bool= field(
        default= True, metadata={"help":"Set to true to use RAFT"})

    image_column: str= field(
        default = "image", metadata={"help":"The column of the datatset containing an image"})

    caption_column: str=field(
        default="text", metadata={"help":"The column of the datset containing an image"})
    
@dataclass
class RaftFinetunerArguments:

    clip_model_pretrained_or_path: str = field(
        default= 'ViT-L-14', metadata={"help":  "Path to pretrained clip model or clip model identifier from huggingface.co/model"})
                                        # "choices":{'aesthetic':['ViT-B-32','ViT-B-16','ViT-L-14']}} )

    topk: int = field(
        default=1, metadata={"help": "Select number of samples to select for finetuning"})
    
    raft_batch_size: int = field(
    default=16, metadata={"help": "Batch size (per device for the training dataloader."})

    score_model: str= field(
        default="aesthetic", metadata={"help": "Score model to ...", "choices":['aesthetic','clip','pick']})
    
    pickscore_processor_name_or_path: Optional[str] = field(
        default= None, metadata={"help": "Path to pretrained autoprocessor model or model identifier from huggingface.co/models for pickscore."})
    
    inference_batch_size: int = field(
        default=3, metadata={'help':''})

    num_images_per_prompt: int= field(
        default=2, metadata={"help":""})

    max_workers: int= field(
        default=1, metadata={"help":"Define max_wokers for mutliprocessing"})

    epochs: int= field(
        default=3, metadata={"help":""})

    save_finetune_images: bool=field(
        default=False, metadata={"help":"Save images generated by finetuned model"})

    num_inference_steps: int = field(
        default=50, metadata={"help": "The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."})
    
    guidance_scale: float = field(
        default=7.5, metadata={"help": '''Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).`guidance_scale` is defined as `w` of equation 2. of [ImagenPaper](https://arxiv.org/pdf/2205.11487.pdf). 
                Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.'''})

    grid: bool=field(
        default=False,metadata={"help":"Save images as a grid"})

    save_format: str=field(
        default= "png", metadata={"help":"File format of save images"
                                  "Reference: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html", 
                                  "choices":["png", "jpeg", "ppm", "gif", "tif", "bmp"]})
    
    negative_prompt: Optional[str] = field(
        default=None, metadata={"help": '''The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).'''})

    pipeline_scheduler: Optional[str]= field(
        default= None, metadata={"help":"", "choices": ['DDPMScheduler', 'PNDMScheduler', 'LMSDiscreteScheduler', 'EulerDiscreteScheduler', 'EulerAncestralDiscreteScheduler', 'DPMSolverMultistepScheduler']}
    )