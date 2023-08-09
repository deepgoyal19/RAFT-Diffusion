accelerate launch --config_file configs/accelerator_singlegpu_config.yaml examples/inferencer.py \
        --save_image_dir '/home/deepanshu/LMFlow-diffusion-main/examples/model' \
        --enable_xformers_memory_efficient_attention True \
        --height 512 \
        --width 512 \
        --grid False \
        --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
        --prompt 'car' \
        --num_images_per_prompt 5 \

