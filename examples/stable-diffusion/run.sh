model=/software/users/mengni/Wan2.2-T2V-A14B-Diffusers/
port=30088
rank=4
out_dir=14b_720p_cp$rank 

PT_HPU_LAZY_MODE=1 deepspeed --num_nodes 1 \
    --num_gpus $rank \
    --no_local_rank \
    --master_port $port \
     wan_t2v_quantization.py \
    --model_name_or_path $model \
    --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --num_videos_per_prompt 1 \
    --use_habana \
    --seed 108 \
    --height 720 \
    --width 1280 \
    --num_frames 81 \
    --num_inference_steps 20 \
    --guidance_scale 5.0 \
    --output_type mp4 \
    --video_save_dir $out_dir \
    --dtype bf16 \
    --context_parallel_size $rank \
    --quant_mode "measure" \
    --quant_config "quantization/wan/measure_config.json" \
    --quant_config_2 "quantization/wan/measure_config_2.json"

