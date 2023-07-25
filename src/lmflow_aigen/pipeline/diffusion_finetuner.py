#!/usr/bin/env python
# coding=utf-8
"""The Finetuner class simplifies the process of running finetuning process on a language model for a TunableModel instance with given dataset. 
"""

import logging
import math
import os
import accelerate
import datasets
import concurrent
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import  UNet2DConditionModel
from packaging import version
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
import diffusers
from diffusers import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
from PIL import Image
import shutil

logger = logging.getLogger(__name__)

class Finetuner:

    def compute_snr(self, timesteps, noise_scheduler):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr


    def save_model_hook(self, models, weights, output_dir):
        if self.model_args.use_ema:
            self.model.ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.model_args.use_ema:
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
            self.model.ema_unet.load_state_dict(load_model.state_dict())
            self.model.ema_unet.to(self.accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model
    
    def deepspeed_zero_init_disabled_context_manager(self):
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

class DiffusionFinetuner(Finetuner):
    # Parameters
    # ------------
    # finetuner_args : FinetunerArguments object.
    #     Contains the arguments required to perform finetuning.

    # kwargs : Mandatory argument required to perform finetuning without RAFT
    #     Keyword arguments.

    def __init__(self, finetuner_args, **kwargs):
        
        self.finetuner_args = finetuner_args

        if 'model_args' and 'data_args' in kwargs:
            self.model_args = kwargs['model_args']
            self.data_args = kwargs['data_args']


        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        # If passed along, set the training seed now.
        if self.finetuner_args.seed is not None:
            set_seed(self.finetuner_args.seed)

        logging_dir = os.path.join(self.finetuner_args.output_dir, self.finetuner_args.logging_dir)

        self.accelerator_project_config = ProjectConfiguration(project_dir=self.finetuner_args.output_dir, logging_dir=logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.finetuner_args.gradient_accumulation_steps,
            log_with=self.finetuner_args.report_to,
            project_config=self.accelerator_project_config)
                

        if self.finetuner_args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            self.optimizer_cls = bnb.optim.AdamW8bit
        else:
            self.optimizer_cls = torch.optim.AdamW
        
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.finetuner_args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.global_step = 0
        self.first_epoch = 0

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.finetuner_args))
            tracker_config.pop("num_train_epochs")
            tracker_config.pop("max_train_steps")
            
            self.accelerator.init_trackers(self.finetuner_args.tracker_project_name, config=tracker_config,init_kwargs={"wandb":{"allow_val_change":True}})

        # logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.finetuner_args.output_dir is not None:
                os.makedirs(self.finetuner_args.output_dir, exist_ok=True)
 
    def finetune(self, **kwargs):


        if self.finetuner_args.overrode_finetuner is False:
            if 'model' and 'dataset' in kwargs:
                self.model = kwargs['model']
                self.dataset = kwargs['dataset']
                # For mixed precision training we cast the text_encoder and vae weights to half-precision
                # as these models are only used for inference, keeping weights in full precision is not required.
                self.model.set_weight_dtype(self.accelerator.mixed_precision)

                # Set device 
                self.model.to_device(self.accelerator.device)

            with ContextManagers(self.deepspeed_zero_init_disabled_context_manager()):
                if self.model_args.use_ema and (self.model_args.use_lora == False):
                    self.model.vae=self.model.vae
                    self.model.text_encoder=self.model.text_encoder

            if self.model_args.use_lora :
                self.model.set_lora_attn_proccessor_to_unet()
            else: 
                if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
                    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
                    self.accelerator.register_save_state_pre_hook(self.save_model_hook)
                    self.accelerator.register_load_state_pre_hook(self.load_model_hook)
                
            if self.finetuner_args.enable_xformers_memory_efficient_attention:
                self.model.use_xformers()


        if self.model_args.use_lora :
            self.lora_layers = AttnProcsLayers(self.model.unet.attn_processors)
            self.lora_layers.to(self.accelerator.device)

        if self.finetuner_args.scale_lr:
            self.finetuner_args.learning_rate = (
                self.finetuner_args.learning_rate * self.finetuner_args.gradient_accumulation_steps * self.data_args.train_batch_size * self.accelerator.num_processes
            )
                
        if self.finetuner_args.gradient_checkpointing and (not self.model_args.use_lora):
            self.model.unet.enable_gradient_checkpointing()
            
        if self.model_args.use_lora:
            self.optimizer = self.optimizer_cls(
                self.lora_layers.parameters(),
                lr=self.finetuner_args.learning_rate,
                betas=(self.finetuner_args.adam_beta1, self.finetuner_args.adam_beta2),
                weight_decay=self.finetuner_args.adam_weight_decay,
                eps=self.finetuner_args.adam_epsilon,
            )
        else:
            self.optimizer = self.optimizer_cls(
                self.model.unet.parameters(),
                lr=self.finetuner_args.learning_rate,
                betas=(self.finetuner_args.adam_beta1, self.finetuner_args.adam_beta2),
                weight_decay=self.finetuner_args.adam_weight_decay,
                eps=self.finetuner_args.adam_epsilon,
            )

        # Get training dataset and training dataloader
        train_dataset,self.accelerator = self.dataset.train_dataset(self.accelerator, self.finetuner_args.seed, self.finetuner_args.max_train_samples, self.model.tokenizer)
        train_dataloader = self.dataset.train_dataloader(self.data_args.train_batch_size)

        # # Scheduler and math around the number of training steps.
        self.overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.finetuner_args.gradient_accumulation_steps)
        if self.finetuner_args.max_train_steps is None:
            self.finetuner_args.max_train_steps = self.finetuner_args.num_train_epochs * num_update_steps_per_epoch
            self.overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            self.finetuner_args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.finetuner_args.lr_warmup_steps * self.finetuner_args.gradient_accumulation_steps,
            num_training_steps=self.finetuner_args.max_train_steps * self.finetuner_args.gradient_accumulation_steps,
        )   

        self.accelerator._models = []
        self.accelerator._optimizers = []
        self.accelerator._schedulers = []
        if self.model_args.use_lora: 
            # Prepare everything with our `accelerator`.
            self.lora_layers, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.lora_layers, self.optimizer, train_dataloader, self.lr_scheduler
            )
        else:
            # Prepare everything with our `accelerator`.
            self.model.unet, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.model.unet, self.optimizer, train_dataloader, self.lr_scheduler
            )
 
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.finetuner_args.gradient_accumulation_steps)
        if self.overrode_max_train_steps:
            self.finetuner_args.max_train_steps = self.finetuner_args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.finetuner_args.num_train_epochs = math.ceil(self.finetuner_args.max_train_steps / num_update_steps_per_epoch)
        
        # Train!
        total_batch_size = self.data_args.train_batch_size * self.accelerator.num_processes * self.finetuner_args.gradient_accumulation_steps
        # print(self.accelerator._schedulers)
        logger.info("***** Running training *****")
        if self.finetuner_args.overrode_finetuner is False:
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {self.finetuner_args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.data_args.train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.finetuner_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.finetuner_args.max_train_steps}")


        # # Potentially load in the weights and states from a previous save
        if self.finetuner_args.resume_from_checkpoint is not None:
            self.resume_from_checkpoint, self.global_step, self.first_epoch, self.resume_step = self.model.resume_from_path(
                                                                self.finetuner_args.resume_from_checkpoint, 
                                                                self.finetuner_args.output_dir,
                                                                self.finetuner_args.gradient_accumulation_steps,
                                                                num_update_steps_per_epoch,
                                                                self.accelerator)
        else: 
            resume_global_step = self.global_step * self.finetuner_args.gradient_accumulation_steps
            self.first_epoch = self.global_step // num_update_steps_per_epoch
            self.resume_step = resume_global_step % (num_update_steps_per_epoch * self.finetuner_args.gradient_accumulation_steps)
            self.resume_from_checkpoint= None
        
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.global_step, self.finetuner_args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(self.first_epoch, self.finetuner_args.num_train_epochs):
            self.model.unet.train()
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if self.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                    if step % self.finetuner_args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                
                with self.accelerator.accumulate(self.model.unet):
                    # Convert images to latent space
                    latents = self.model.vae.encode(batch["pixel_values"].to(dtype=self.model.weight_dtype)).latent_dist.sample()
                    latents = latents * self.model.vae.config.scaling_factor
                    # latents = latents.to(self.accelerator.device)
                    
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if self.finetuner_args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self.finetuner_args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    if self.finetuner_args.input_perturbation and (not self.use_lora):
                        new_noise = noise + self.finetuner_args.input_perturbation * torch.randn_like(noise)
                    
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.model.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if self.finetuner_args.input_perturbation and (not self.use_lora):
                        noisy_latents = self.model.noise_scheduler.add_noise(latents, new_noise, timesteps)
                    else:
                        noisy_latents = self.model.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = self.model.text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if self.finetuner_args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        self.model.noise_scheduler.register_to_config(prediction_type=self.finetuner_args.prediction_type)
                    if self.model.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.model.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.model.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.model.noise_scheduler.config.prediction_type}")
                    # Predict the noise residual and compute loss
                    model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self.finetuner_args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = self.compute_snr(timesteps, self.model.noise_scheduler)
                        mse_loss_weights = (
                            torch.stack([snr, self.finetuner_args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.data_args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.finetuner_args.gradient_accumulation_steps

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        if self.model_args.use_lora:
                            params_to_clip = self.lora_layers.parameters()
                            self.accelerator.clip_grad_norm_(params_to_clip, self.finetuner_args.max_grad_norm)
                        else:
                            self.accelerator.clip_grad_norm_(self.model.unet.parameters(), self.finetuner_args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the self.accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.model_args.use_ema and (self.model_args.use_lora is False):
                        self.model.ema_unet.step(self.model.unet.parameters())
                    progress_bar.update(1)
                    self.global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                    train_loss = 0.0

                    if self.global_step % self.finetuner_args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self.finetuner_args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(os.path.join(self.finetuner_args.output_dir))
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self.finetuner_args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - self.finetuner_args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(self.finetuner_args.output_dir,removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)
                            save_path = os.path.join(self.finetuner_args.output_dir,f"checkpoint-{self.global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if self.global_step >= self.finetuner_args.max_train_steps:
                    break
                
            # Make a validation log
            if self.finetuner_args.validation_prompts is not None:
                self.model.log(self.finetuner_args,self.accelerator,self.global_step, self.data_args.resolution, 'validation')
            
            self.finetuner_args.learning_rate = self.lr_scheduler.get_last_lr()[0]

        if self.finetuner_args.last_epoch:
            # Create the pipeline using the trained modules and save it.
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:         
                # save model
                self.model.save(self.finetuner_args.output_dir, self.accelerator)

                # Push model to hub
                if self.finetuner_args.push_to_hub:
                    self.model.push_to_hub(
                        self.finetuner_args.hub_model_id,
                        self.finetuner_args.output_dir, 
                        self.finetuner_args.hub_token, 
                        self.data_args.dataset_name
                        )
        
            #Final Inference
            if self.finetuner_args.validation_prompts is not None:
                self.model.log(self.finetuner_args,self.accelerator,self.global_step, self.data_args.resolution, 'test')
            
            torch.cuda.empty_cache()

        


class RaftFinetuner(DiffusionFinetuner):
    # Parameters
    # ------------
    # model_args : ModelArguments object.
    #     Contains the arguments required to load the model.

    # data_args : DatasetArguments object.
    #     Contains the arguments required to load the dataset.

    # raft_args : RaftArguments object.
    #     Contains the arguments required to perform RAFT algorithm

    def __init__(self, model_args, data_args, finetuner_args, raft_args):
        self.raft_args = raft_args
        self.data_args = data_args
        self.model_args = model_args 
        super().__init__(finetuner_args)        

        
        if self.finetuner_args.resume_from_checkpoint:
            if self.finetuner_args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.finetuner_args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.finetuner_args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                self.global_step = int(path.split("-")[1])
                resume_global_step = self.global_step * self.finetuner_args.gradient_accumulation_steps
                self.resume_from_raft_epoch = int(int(path.split("-")[1]) / self.finetuner_args.max_train_steps)
                self.raft_epoch = self.resume_from_raft_epoch 
                self.raft_args.epochs+= self.raft_epoch
                self.finetuner_args.max_train_steps+= resume_global_step
            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.finetuner_args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.raft_epoch = 0
                self.resume_from_raft_epoch = 0
        else:
            self.raft_epoch = 0
            self.resume_from_raft_epoch = 0

        self.training_steps_per_epoch=self.finetuner_args.max_train_steps
        self.finetuner_args.resume_from_checkpoint = None
        self.finetuner_args.last_epoch  = False

        if os.path.exists(os.path.join(self.finetuner_args.output_dir,'raft_images')):
                        shutil.rmtree(os.path.join(self.finetuner_args.output_dir,'raft_images'))
        os.makedirs(os.path.join(self.finetuner_args.output_dir,'raft_images'))
        
    def raft_finetune(self, model, dataset):
        self.model=model
        self.dataset=dataset

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.model.set_weight_dtype(self.accelerator.mixed_precision)

        # Set device 
        self.model.to_device(self.accelerator.device)

        # Load Score Model
        self.model.load_score_model(self.raft_args.clip_model_pretrained_or_path,self.raft_args.score_model, self.accelerator.device)

        # Get dataloader to perform RAFT
        raft_dataloader=self.dataset.raft_dataloader(self.raft_args.inference_batch_size)
        
        generator = torch.Generator(device=self.accelerator.device)
        if self.finetuner_args.seed is not None:
            generator = generator.manual_seed(self.finetuner_args.seed)
        
        # Load pipeline scheduler
        if self.raft_args.pipeline_scheduler:
            self.model.import_pipeline_scheduler(self.raft_args.pipeline_scheduler)

        # Run RAFT 
        for self.raft_epoch in range(self.resume_from_raft_epoch,self.raft_args.epochs):
            

            # Get score model to device
            self.model.score_model.to(self.accelerator.device)
            # self.model.score_amodel.to(self.accelerator.device)

            # Load Diffusion Pipeline
            pipeline = self.model.load_model_pipeline(self.accelerator)

            # Use xformers to save memory
            if self.finetuner_args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            # Load pipeline scheduler
            if self.raft_args.pipeline_scheduler:
                pipeline.scheduler = self.model.load_pipeline_scheduler(pipeline)
            
            # Use Raft Algorithm 
            training_prompts=[]
            for step, prompts in enumerate(raft_dataloader):

                # Get negative prompts for raft pipeline
                if self.raft_args.negative_prompt:
                    negative_prompts = [self.raft_args.negative_prompt for _ in range(len(prompts['text']))]
                else:   
                    negative_prompts = None

                # Generating Images
                prompts=prompts['text']
                if self.model_args.use_lora :
                    images=pipeline( 
                            prompts,
                            num_images_per_prompt=self.raft_args.num_images_per_prompt,
                            width=self.data_args.resolution,
                            height=self.data_args.resolution,
                            num_inference_steps=self.raft_args.num_inference_steps,
                            negative_prompt= negative_prompts,
                            generator=generator).images
                else:
                    with torch.autocast(device_type='cuda'):
                        images=pipeline( 
                            prompts,
                            num_images_per_prompt=self.raft_args.num_images_per_prompt,
                            width=self.data_args.resolution,
                            height=self.data_args.resolution,
                            num_inference_steps=self.raft_args.num_inference_steps,
                            negative_prompt= negative_prompts,
                            generator=generator).images
                images_list=[]
            
                # Delete Stable pipeline to save memory
                del pipeline 

                # Appending images to the images_list
                for i in range(int(len(images)/self.raft_args.num_images_per_prompt)):
                    images_list.append(images[i*self.raft_args.num_images_per_prompt:(i+1)*self.raft_args.num_images_per_prompt])
                torch.cuda.empty_cache()

                # Get Aesthetic scores, Pick scores and CLIP scores of images
                with concurrent.futures.ThreadPoolExecutor(max_workers= self.raft_args.max_workers) as executor:
                    score_index=executor.map(self.model.preprocess_image,images_list,prompts)

                iterator=0
                for max_scores in score_index:
                    training_prompts.append([max_scores[0],images_list[iterator][max_scores[1]],prompts[iterator]])
                    iterator+=1

            # Load model to cpu to save GPU memory space
            self.model.score_model.to('cpu')
            # self.model.score_amodel.to('cpu')
            

            # Get images wit top scores
            training_prompts=[row[1:3] for row in sorted(training_prompts,key=lambda x: (x[0]),reverse=True)[:self.raft_args.topk]]

            # Store texts and images
            images = [row[0] for row in training_prompts]
            texts = [row[1] for row in training_prompts]

            if self.raft_args.save_raft_images:
                for i in range(len(images)):
                    if os.path.exists(f"{os.path.join(self.finetuner_args.output_dir,'raft_images')}/raft_epoch-{self.raft_epoch}"):
                        shutil.rmtree(f"{os.path.join(self.finetuner_args.output_dir,'raft_images')}/raft_epoch-{self.raft_epoch}")
                    os.makedirs(f"{os.path.join(self.finetuner_args.output_dir,'raft_images')}/raft_epoch-{self.raft_epoch}")
                    images[i].save(f"{os.path.join(self.finetuner_args.output_dir,'raft_images')}/raft_epoch-{self.raft_epoch}/{i}.png")

            # Prepare dataset for finetuning
            self.dataset.prepare_raft_finetuner_dataset(images, texts)

            # Finetune
            logger.info(f"  Raft Epoch = {self.raft_epoch}")
            self.finetuner_args.max_train_steps = self.training_steps_per_epoch*(self.raft_epoch+1)
            if self.raft_epoch==self.raft_args.epochs-1:
                self.finetuner_args.last_epoch = True
            if self.raft_epoch>0:
                self.finetuner_args.overrode_finetuner = True

            self.finetune()  
            torch.cuda.empty_cache()
        
        # Save images generated by finetuned model
        if self.raft_args.save_finetune_images:
            images_list=[]
            self.model.to_device(self.accelerator.device)
            finetuned_model_pipeline = self.model.load_model_pipeline(self.accelerator)
            finetuned_model_pipeline.enable_xformers_memory_efficient_attention()
            if self.raft_args.pipeline_scheduler:
                finetuned_model_pipeline.scheduler = self.model.load_pipeline_scheduler(finetuned_model_pipeline)
            self.accelerator.free_memory()
            for step, prompts in enumerate(raft_dataloader):
                prompts=prompts['text']
                if self.model_args.use_lora :
                    images=finetuned_model_pipeline( 
                            prompts,
                            num_images_per_prompt=self.raft_args.num_images_per_prompt,
                            width=self.data_args.resolution,
                            height=self.data_args.resolution,
                            num_inference_steps=self.raft_args.num_inference_steps,
                            negative_prompt= negative_prompts,
                            generator=generator).images
                else:
                    with torch.autocast(device_type='cuda'):
                        images=finetuned_model_pipeline( 
                            prompts,
                            num_images_per_prompt=self.raft_args.num_images_per_prompt,
                            width=self.data_args.resolution,
                            height=self.data_args.resolution,
                            num_inference_steps=self.raft_args.num_inference_steps,
                            negative_prompt= negative_prompts,
                            generator=generator).images
                images_list.extend(images)

            # Save images as a grid
            if self.raft_args.grid:
                rows=int(len(images_list)/self.raft_args.num_images_per_prompt)
                cols=self.raft_args.num_images_per_prompt
                assert len(images_list) == rows*cols
                w, h = images_list[0].size
                grid = Image.new('RGB', size=(cols*w, rows*h))
                
                for i, img in enumerate(images_list):
                    grid.paste(img, box=(i%cols*w, i//cols*h))
                
                if self.finetuner_args.output_dir:
                    grid.save(f"{self.finetuner_args.output_dir}/raft_images/output.{self.raft_args.save_format}")
            else:
                # Save all images
                if self.finetuner_args.output_dir:
                    for len_images in range(len(images)):
                        images[len_images].save(f"{self.finetuner_args.output_dir}/raft_images/{len_images}.{self.raft_args.save_format}")

        # Return Finetuned Model
        return finetuned_model_pipeline
