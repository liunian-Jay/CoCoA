. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate DeepSpeed


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
deepspeed --master_port 25558 --num_gpus=1 /path/llm_train/finetuning.py \
    --model_name_or_path "/path/models/Meta-Llama-3.1-8B-Instruct" \
    --data_path "/path/data/train_data/sft_data/long_6888.jsonl" \
    --deepspeed "/path/configs/ds_config_zero2.json" \
    --output_dir "/path/model_outputs/sft/sft_long_1" \
    --fp16 False \
    --bf16 True \
    --tf32 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --logging_steps 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.99 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --model_max_length 2560 \
    --gradient_checkpointing True \
    --use_lora True \
    --lazy_preprocess False \
    --project_name "AP-RAG" \
    --run_name "longCogRAG-SFT-long-1"


