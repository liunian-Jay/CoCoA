import os
from warnings import warn
import torch
import wandb
import transformers
from transformers import Trainer 
import torch.distributed as dist
# from transformers import deepspeed
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    LlamaForCausalLM,
    AutoModelForCausalLM
)

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from arguments import (
    LoraArguments, DataArguments, TrainingArguments, ModelArguments,
    quantization_config as QUANTIZATION_CONFIG,
)
from dataset import make_supervised_data_module
from train_utils import (
    safe_save_model_for_hf_trainer,
    set_seed,
    print_model_size
)




def train():
    parser = transformers.HfArgumentParser( (ModelArguments, DataArguments, TrainingArguments, LoraArguments) )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Ensure repeatability
    set_seed(training_args.seed)

    # This serves for single-gpu qlora.
    from accelerate.utils import DistributedType
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    # torchrun specific
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    if rank == 0 and 'wandb' in training_args.report_to:
        print(training_args)
        wandb.login(key="xxxxx")              #TODO delete
        wandb.init(
            project = training_args.project_name,   
            name = training_args.run_name,          
            # config = training_args                
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    print(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ####################################################
    ####################################################
    ####################################################
    # special_token_dict = {
    #     "additional_special_tokens": [
    #         "<Internal\>", "<\Internal>",
    #         "<External\>", "<\External>",
    #         "<Thinking\>", "<\Thinking>",
    #         "<Source\>", "<\Source>",
    # ]} 
    # num_added_tokens = tokenizer.add_special_tokens(special_token_dict)
    ####################################################
    ####################################################
    ####################################################
    
    # If there is a mismatch between tokenizer vocab size and embedding matrix, throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print('*'*100,)
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        print('Old vocab size: ', model.get_input_embeddings().weight.shape[0] )
        print('New vocab size: ', len(tokenizer) )
        model.resize_token_embeddings(len(tokenizer))
        print('*'*100,)
        print(model)
    print_model_size(model, model_args, rank)
 

    # 使用Lora微调
    if training_args.use_lora:
        if is_chat_model:
            modules_to_save = None # instruct  # TODO
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

            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save  # This argument serves for adding new tokens.
            )
            # lora_weight_path: str = ""
            # inference_mode: bool = False
            
            if training_args.quantization:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(lora_args.lora_target_modules)
        print(modules_to_save)
        print(model)
    
    # input require grads
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model = model, 
        tokenizer = tokenizer, 
        args = training_args, 
        **data_module
    )

    with torch.autocast("cuda"):
        trainer.train()
    trainer.save_state()

    # safe_save_model_for_hf_trainer(
    #     trainer=trainer, 
    #     output_dir=training_args.output_dir, 
    #     bias=lora_args.lora_bias
    # )
    model = trainer.model.merge_and_unload()
    print(model)
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    wandb.finish()



if __name__ == "__main__":
    train()