import os
import json
import wandb
from warnings import warn
import fire
import torch
import transformers
from transformers import set_seed
from transformers import TrainingArguments
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from peft import LoraConfig, PeftModel
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer,DPOConfig

############################
from datasets import Dataset
import pandas as pd

from train_utils import print_model_size
from arguments_dpo import (
    LoraArguments, DataArguments, TrainingArguments, ModelArguments,
    quantization_config as QUANTIZATION_CONFIG,
)


def read_jsonl(file_path, return_df=True):
    # turn jsonl to dataframe
    data = []
    print(f"Reading {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    if return_df:
        return pd.DataFrame(data)
    else:
        return data
    

def main():
    parser = transformers.HfArgumentParser( (ModelArguments, DataArguments, TrainingArguments, LoraArguments) )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Ensure repeatability
    set_seed(training_args.seed)

    # This serves for single-gpu qlora.
    from accelerate.utils import DistributedType
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # torchrun specific
    rank = int(os.environ["RANK"])
    if rank == 0 and 'wandb' in training_args.report_to:
        print(training_args)
        wandb.login(key="xxxx")              #TODO delete
        wandb.init(
            project = training_args.project_name,  
            name = training_args.run_name,          
        )

    is_chat_model = 'instruct' in model_args.model_name_or_path.lower()
    if (training_args.use_lora and is_deepspeed_zero3_enabled() and not is_chat_model ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")
    

    # setting quantization configs
    quantization_config = None
    if training_args.quantization is not None:
        if type(training_args.quantization) == type(True):
            warn( "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.", FutureWarning,)
            training_args.quantization = "8bit"
        quant_config = QUANTIZATION_CONFIG()
        quantization_config = quant_config.create_bnb_config(training_args.quantization)
    print("quantization_config:", quantization_config)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    print(config)
    # config.rope_scaling.factor = config.rope_scaling.factor
    config.use_cache = False

    # Load model and tokenizer  
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
        config=config,
        device_map =  "auto" if quantization_config else None,
        quantization_config = quantization_config,
        torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
    )
    # Reference model
    # ref_model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     load_in_4bit=True,
    #     device_map=device_map,
    # )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.max_length,
        padding_side="right",
        truncation_side ="right",
        # use_fast=False,
        trust_remote_code=True,
    )
    print(tokenizer)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix, throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
    print_model_size(model, model_args, rank)

    # Lora
    if training_args.use_lora:

        if is_chat_model:# TODO
            modules_to_save = None # instruct
            # modules_to_save = ["wte", "lm_head"] 
        else:
            modules_to_save = ["wte", "lm_head"] 


        # Load the pre-trained peft model checkpoint and setup its configuration
        if training_args.from_peft_checkpoint:
            model = PeftModel.from_pretrained(
                model, training_args.from_peft_checkpoint, is_trainable=True
            )
            peft_config = model.peft_config
        # Generate the peft config and start fine-tuning from original model
        else:
            def find_all_linear_names(args, model):
                import bitsandbytes as bnb
                cls = bnb.nn.Linear4bit if args.load_in_4bit == 4 else (
                    bnb.nn.Linear8bitLt if args.load_in_8bit == 8 else torch.nn.Linear)
                lora_module_names = set()
                for name, module in model.named_modules():
                    if isinstance(module, cls):
                        names = name.split('.')
                        lora_module_names.add(names[0] if len(names) == 1 else names[-1])
                if 'lm_head' in lora_module_names:  # needed for 16-bit
                    lora_module_names.remove('lm_head')
                return list(lora_module_names)
            
            if lora_args.lora_target_modules is None:
                lora_args.lora_target_modules = find_all_linear_names(lora_args, model)
            
            peft_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save  # This argument serves for adding new tokens.
            )

            if training_args.quantization:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )



    #################################################
    #################################################
    train_data = read_jsonl(data_args.data_path)
    dev_data = read_jsonl(data_args.eval_data_path) if data_args.eval_data_path else None

    # shuffle data
    train_data = train_data.sample(frac = 1, random_state=2025)
    dev_data = dev_data.sample(frac = 1, random_state=2025) if dev_data is not None else None

    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data) if dev_data is not None else None
    #################################################
    #################################################

    # Define Trainer
    args = DPOConfig(
        deepspeed=training_args.deepspeed,
        output_dir=training_args.output_dir,

        fp16=training_args.fp16,
        bf16=training_args.bf16,
        tf32=training_args.tf32,

        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        evaluation_strategy=training_args.evaluation_strategy,
        eval_steps=training_args.eval_steps,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        logging_steps=training_args.logging_steps,

        learning_rate=training_args.learning_rate,
        max_grad_norm=training_args.max_grad_norm,
        optim="adamw_torch_fused",
        warmup_ratio=training_args.warmup_ratio,
        
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        beta=training_args.dpo_beta,
        rpo_alpha=training_args.rpo_alpha,

        report_to=training_args.report_to,
        run_name=training_args.run_name,
        # load_best_model_at_end=True,
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        # ref_model=ref_model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Fine-tune model with DPO
    with torch.autocast("cuda"):
        dpo_trainer.train()
    dpo_trainer.save_state()
    # dpo_trainer.model.save_pretrained(training_args.output_dir)

    model = dpo_trainer.model.merge_and_unload()
    print(model)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    wandb.finish()


if __name__ == '__main__':
    main()