#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""

import copy
import logging
import os
import sys

import datasets
import transformers
import evaluate
from itertools import chain
from transformers import (
    Trainer,
    default_data_collator,
    set_seed,
)
from copy import deepcopy
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.base_tuner import BaseTuner


logger = logging.getLogger(__name__)


class DiffusionFinetuner(BaseTuner):
    pass