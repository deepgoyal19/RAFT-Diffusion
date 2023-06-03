#!/usr/bin/env python
# coding=utf-8
"""This script defines dataclasses: ModelArguments and DatasetArguments,
that contain the arguments for the model and dataset used in training.


"""

from dataclasses import dataclass, field
from typing import Optional, List

from transformers.utils.versions import require_version

from transformers import (
    TrainingArguments,
)

# from lmflow.args import ModelArguments,DatasetArguments, EvaluatorArguments, BenchmarkingArguments


from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class FinetunerArguments(TrainingArguments):
    """
    Adapt transformers.TrainingArguments
    """
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
    

@dataclass 
class ModelArguments:

    pretrained_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})

    use_lora: bool = field(
        default=False, metadata={"help": "Whether to lora."})
    
    torch_dtype: Optional[str] = field(default=None,metadata={"help": ("Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed,the "
        "dtype will be automatically derived from the model's weights."), "choices": ["auto", "bfloat16", "float16", "float32"]})


@dataclass
class InferenceArguments:


@dataclass
class RaftAlignterArguments:
    
