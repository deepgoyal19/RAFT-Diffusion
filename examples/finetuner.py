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
    FinetunerArguments
    )

from lmflow_aigen.datasets.dataset import ImageDataset
from lmflow_aigen.pipeline.diffusion_finetuner import DiffusionFinetuner
from lmflow_aigen.models.diffusion_model import DiffusionModel


def main():

    parser= HfArgumentParser((FinetunerArguments, ModelArguments, DatasetArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        finetuner_args, model_args, data_args= parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        finetuner_args, model_args, data_args= parser.parse_args_into_dataclasses()


    # Initialization
    diffusion_finetuner = DiffusionFinetuner(finetuner_args=finetuner_args, data_args=data_args, model_args=model_args)
    model = DiffusionModel(model_args)
    dataset= ImageDataset(data_args)

    # Finetuning
    tuned_model = diffusion_finetuner.finetune(model=model,dataset=dataset)

if __name__ == '__main__':
    main()

