. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate DeepSpeed



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
deepspeed --master_port 25555 --num_gpus=1 /path/llm_train/finetuning_dpo.py \
    --model_name_or_path "/path/model_outputs/sft/sft_long_2/5epoch" \
    --data_path "/path/data/train_data/dpo_data/dpo_1151_long2.jsonl" \
    --deepspeed "/path/configs/ds_config_zero2.json" \
    --output_dir "/path/model_outputs/dpo/dpo_long2" \
    --fp16 False \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --logging_steps 1 \
    --learning_rate 5e-6 \
    --max_grad_norm 0.3 \
    --rpo_alpha 0.2 \
    --warmup_ratio 0.03 \
    --dpo_beta 0.2 \
    --report_to "wandb" \
    --max_length 2048 \
    --gradient_checkpointing True \
    --use_lora True \
    --project_name "project_name" \
    --run_name "run_name"


