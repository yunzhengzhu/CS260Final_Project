python run_generation.py \
--model_type gpt2-m \
--model_name_or_path models_dir/$1 \
--length 128 \
--stop_token "<|endoftext|>" \
--k 50
