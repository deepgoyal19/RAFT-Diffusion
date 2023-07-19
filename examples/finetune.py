# !/usr/bin/env python
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
import os
import sys
from transformers import HfArgumentParser

from lmflow_aigen.args import (
    ModelArguments,
    DatasetArguments,
    FinetunerArguments,
    RaftFinetunerArguments,
    )

from lmflow_aigen.datasets.dataset import ImageDataset
from lmflow_aigen.pipeline.diffusion_finetuner import RaftFinetuner
from lmflow_aigen.models.diffusion_model import DiffusionModel


def main():

    parser= HfArgumentParser((FinetunerArguments, ModelArguments, DatasetArguments, RaftFinetunerArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, model_args, data_args, raft_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, model_args, data_args, raft_args = parser.parse_args_into_dataclasses()

    finetuner_args = FinetunerArguments(
        output_dir='/home/deepanshu/LMFlow-diffusion-main/examples/model',
        enable_xformers_memory_efficient_attention=True,
        gradient_accumulation_steps=1,
        lr_scheduler='constant',
        learning_rate=9e-6,
        lr_warmup_steps=0,
        max_train_steps=50,
        gradient_checkpointing=True,
        max_grad_norm=1,
        tracker_project_name="text2image-fine-tune",
        checkpointing_steps=40
    )

    model_args= ModelArguments(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        use_ema= False,
        use_lora= True)

    data_args=DatasetArguments(
        resolution=256,
        dataset_name='/home/deepanshu/LMFlow-diffusion-main/a',
        train_batch_size=8
    )

    raft_args=RaftFinetunerArguments(
        topk=8,
        inference_batch_size=10,
        epochs=2,
        num_images_per_prompt=10,
        clip_model_pretrained_or_path='ViT-L/14',
        score_model='clip',
        save_finetune_images=True,
        grid=True
    )

    # inference_args= InferenceArguments(
    #     grid= True,
    #     num_images_per_prompt=5,
    #     num_inference_steps= 100,
    #     guidance_scale= 6,
    #     save_format = 'jpg'
    #     seed= 1232
    # )


    # Initialization
    raft_finetuner = RaftFinetuner(raft_args=raft_args, data_args=data_args, model_args=model_args, finetuner_args=finetuner_args)
    model = DiffusionModel(model_args)
    dataset= ImageDataset(data_args)

    # Finetuning
    tuned_model = raft_finetuner.raft_finetune(model=model,dataset=dataset)

if __name__ == '__main__':
    main()

