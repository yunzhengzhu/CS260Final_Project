export DATA_TYPE=comve
export ROOT_PATH=..
export DEVICE=1
CUDA_VISIBLE_DEVICES=${DEVICE} \
python3 main.py \
--train_data_file ${ROOT_PATH}/data/${DATA_TYPE}/train \
--dev_data_file ${ROOT_PATH}/data/${DATA_TYPE}/dev \
--test_data_file ${ROOT_PATH}/data/${DATA_TYPE}/test \
--graph_path 2hops_100_directed_triple_filter.json \
--output_dir ${ROOT_PATH}/models/${DATA_TYPE}/grf-${DATA_TYPE} \
--source_length 32 \
--target_length 32 \
--model_type gpt2 \
--model_name_or_path ${ROOT_PATH}/models/gpt2-small \
--do_train \
--save_last \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--num_train_epochs 3 \
--learning_rate 1e-5 \
--aggregate_method max \
--alpha 3 \
--beta 5 \
--gamma 0.5 \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--logging_steps 20 \
