#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import sys
import os
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))

from lmflow_aigen.args import (
    ModelArguments,
    DatasetArguments,
    FinetunerArguments,
)

from lmflow_aigen.datasets.dataset import Dataset
from lmflow_aigen.models.diffusion_finetuner import DiffusionFinetuner
from lmflow_aigen.pipeline.diffusion_model import DiffusionModel


def main():
    finetuner_args = FinetunerArguments(

        enable_xformers_memory_efficient_attention=True,
        lr_scheduler='constant',
        learning_rate=1e-6,
        lr_warmup_steps=0,
        seed=123,
        max_train_steps=60,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        train_batch_size=1,
        max_grad_norm=1,
        mixed_precision="fp16",
        output_dir='/home/deepanshu/LMFlow-diffusion/Model',
        accelerate_device= 'cuda',
        validation_prompt='pikachu',
        resume_from_checkpoint='latest',
        hub_token='[REDACTED]',
        hub_model_id='new_model',
        num_validation_images=1,
        push_to_hub=True,
        validation_epochs=60
    )

    model_args= ModelArguments(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        use_ema=True,
        use_lora= False)

    data_args=DatasetArguments(
        dataset_name="lambdalabs/pokemon-blip-captions"
    )

    finetuner = DiffusionFinetuner(finetuner_args=finetuner_args, model_args=model_args, data_args=data_args)
    model = DiffusionModel(model_args)
    dataset= Dataset(data_args)

    # Finetuning
    tuned_model = finetuner.tune(model=model, dataset=dataset)


if __name__ == '__main__':
    main()
