output_dir=${proj}
accelerate launch examples/finetuner.py \
        --output_dir '/home/deepanshu/LMFlow-diffusion-main/examples/model' \
        --enable_xformers_memory_efficient_attention True \
        --gradient_accumulation_steps 1 \
        --lr_scheduler 'constant' \
        --learning_rate 9e-6 \
        --lr_warmup_steps 0 \
        --max_train_steps 1 \
        --gradient_checkpointing True \
        --max_grad_norm 1 \
        --tracker_project_name "text2image-fine-tune" \
        --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
        --use_ema False \
        --use_lora True \
        --resolution 256 \
        --dataset_name '/home/deepanshu/LMFlow-diffusion-main/a' \
        --train_batch_size 5 \
        --inference_batch_size 10 \
        --save_finetune_images True \
        --grid True \
        --noise_offset -0.1 \

