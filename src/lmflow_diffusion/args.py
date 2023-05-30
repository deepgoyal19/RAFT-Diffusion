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

from lmflow.args import ModelArguments,DatasetArguments, EvaluatorArguments, BenchmarkingArguments


