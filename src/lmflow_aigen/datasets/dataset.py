#!/usr/bin/env python
# coding=utf-8
"""This Python code defines a class Dataset with methods for initializing, loading,
and manipulating datasets from different backends such as Hugging Face and JSON.
 
The `Dataset` class includes methods for loading datasets from a dictionary and a Hugging
Face dataset, mapping datasets, and retrieving the backend dataset and arguments.
"""



# Importing necessary libraries and modules
import os
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset
from torchvision import transforms

class ImageDataset:
    r"""
    Initializes the Dataset object with the given parameters.

    Parameters
    ------------
    data_args : DatasetArguments object.
        Contains the arguments required to load the dataset.
    
    args : Optional.
        Positional arguments.
    
    kwargs : Optional.
        Keyword arguments.
    """
    def __init__(self, data_args=None):
        self.data_args=data_args

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            # Downloading and loading a dataset from the hub.
            self.dataset = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.data_args.cache_dir,
            )
        elif  self.data_args.train_data_dir is not None:
            if self.data_args.overrode_init_dataset:
                self.dataset = load_dataset(
                    "text",
                    data_dir=self.data_args.train_data_dir,
                    cache_dir=self.data_args.cache_dir,
                    )
            else:
                data_files = {} 
                data_files["train"] = os.path.join(self.data_args.train_data_dir, "**")
                self.dataset = load_dataset(
                    "imagefolder",
                    data_files=data_files,
                    cache_dir=self.data_args.cache_dir,
                )
        else:
                raise ValueError('Pease specify the name of the dataset or provide the path to a folder that includes a text file.')
        
        if not self.data_args.overrode_init_dataset:
            self.prepare_plain_finetuner_dataset()


    def raft_dataloader(self,batch_size):
        dataloader=torch.utils.data.DataLoader(
            self.dataset["train"],
            shuffle=False,
            batch_size=batch_size)
        return dataloader
    
    def prepare_raft_finetuner_dataset(self, images, texts):
        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        self.dataset=Dataset.from_dict({"image": images, "text": texts})

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.  

        column_names = self.dataset.column_names

        self.image_column = column_names[0]

        self.caption_column = column_names[1]

        # Preprocessing the datasets.
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(self.data_args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.data_args.resolution) if self.data_args.center_crop else transforms.RandomCrop(self.data_args.resolution),
                transforms.RandomHorizontalFlip() if self.data_args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def prepare_plain_finetuner_dataset(self):
        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        DATASET_NAME_MAPPING = {
            "lambdalabs/pokemon-blip-captions": ("image", "text"),
        }   
        column_names = self.dataset["train"].column_names
        self.dataset = self.dataset["train"]

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(self.data_args.dataset_name, None)

        if self.data_args.image_column is None:
            self.image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            self.image_column = self.data_args.image_column
            if self.image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{self.data_args.image_column}' needs to be one of: {', '.join(column_names)}"
                )

        if self.data_args.caption_column is None:
            self.caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            self.caption_column = self.data_args.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{self.data_args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Preprocessing the datasets.
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(self.data_args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.data_args.resolution) if self.data_args.center_crop else transforms.RandomCrop(self.data_args.resolution),
                transforms.RandomHorizontalFlip() if self.data_args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        

    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples[self.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions, max_length =self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(self, examples):
        images = [image.convert("RGB") for image in examples[self.image_column]]
        examples["pixel_values"] = [self.train_transforms(image) for image in images]
        examples["input_ids"] = self.tokenize_captions(examples)
        return examples
    
    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    def train_dataset(self, accelerator, seed, max_train_samples,tokenizer):
        self.tokenizer=tokenizer

        with accelerator.main_process_first():
            if max_train_samples is not None:
                self.dataset = self.dataset.shuffle(seed=seed).select(range(max_train_samples))
            # Set the training transforms
            self.training_dataset = self.dataset.with_transform(self.preprocess_train)
            return self.training_dataset, accelerator

    def train_dataloader(self, train_batch_size):
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            self.training_dataset,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=train_batch_size,
            num_workers=self.data_args.dataloader_num_workers,
        )
        return train_dataloader