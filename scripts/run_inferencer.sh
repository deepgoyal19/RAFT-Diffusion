accelerate launch examples/inferencer.py \
        --save_image_dir '/home/deepanshu/LMFlow-diffusion-main/examples/model' \
        --enable_xformers_memory_efficient_attention True \
        --height 512 \
        --width 512 \
        --grid True \
        --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
        --prompt 'car'

