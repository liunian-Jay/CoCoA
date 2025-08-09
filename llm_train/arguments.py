import torch
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, List
import transformers
from transformers import BitsAndBytesConfig

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={ "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)." },
    )
    
    quantization: Optional[Literal[None, "4bit", "8bit"]] = field(default=None)
    report_to: str = 'none'
    use_lora: bool = False
    from_peft_checkpoint: str="" # if not empty and use_lora=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    project_name: str = '' # wandb
    run_name: str = '' # wandb

@dataclass
class LoraArguments:
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory = lambda: ['o_proj', 'k_proj', 'q_proj', 'v_proj'],
        metadata={ "help": "The available modules are: ['gate_proj', 'o_proj', 'k_proj', 'q_proj', 'up_proj', 'down_proj', 'v_proj']" }
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_dropout: float = 0.05
    # inference_mode: bool = False

    # q_lora: bool = False
    # load_in_4bit: bool = False
    # load_in_8bit: bool = False


@dataclass
class quantization_config:
    quant_type: str =  "fp4" # "fp4" or "nf4"
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = False
    quant_storage: torch.dtype = torch.bfloat16

    def create_bnb_config(self, quantization: str) -> BitsAndBytesConfig:
        if quantization not in {"4bit", "8bit"}:
            raise ValueError("quantization must be either '4bit' or '8bit'")

        if quantization == "4bit":
            config_params = {
                "bnb_4bit_quant_type": self.quant_type,
                "bnb_4bit_compute_dtype": self.compute_dtype,
                "bnb_4bit_use_double_quant": self.use_double_quant,
                "bnb_4bit_quant_storage": self.quant_storage,
            }
            
            return BitsAndBytesConfig(load_in_4bit=True, **config_params)
        else:
            return BitsAndBytesConfig(load_in_8bit=True)




