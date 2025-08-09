import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(lm_type):
    if lm_type == "Llama-2-7b-chat-hf":
        model_name_or_path = os.path.abspath("/home/share/models/Llama-2-7b-chat-hf")
    elif lm_type == "Llama-2-13b-chat-hf":
        model_name_or_path = os.path.abspath("/home/share/models/Llama-2-13b-chat-hf")
        
    elif lm_type == "Llama-3-8B-Instruct":
        model_name_or_path = os.path.abspath("/path/models/Meta-Llama-3-8B-Instruct")
    elif lm_type == "Llama-3-70B-Instruct":
        model_name_or_path = os.path.abspath("/path/models/Meta-Llama-3-70B-Instruct")
    elif lm_type == "Mistral-7B-Instruct-v0.2":
        model_name_or_path = os.path.abspath("/home/share/models/Mistral-7B-Instruct-v0.2")
    elif lm_type == "Mistral-7B-Instruct-v0.3":
        model_name_or_path = os.path.abspath("/home/share/models/Mistral-7B-Instruct-v0.3") 

    elif lm_type == "Llama-3.1-8B-Instruct":
        model_name_or_path = os.path.abspath("/home/share/models/Meta-Llama-3.1-8B-Instruct")
    elif lm_type == "Llama-3.1-70B-Instruct":
        model_name_or_path = os.path.abspath("/home/share/models/Meta-Llama-3.1-70B-Instruct") 

    elif lm_type == "gpt2":
        model_name_or_path = os.path.abspath("/home/share/models/gpt2")   
    else:
        model_name_or_path = os.path.abspath(lm_type) 
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer



def llm_response(inputs, model, tokenizer):
    """ 调用generate生成回复 """
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,

            eos_token_id = tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    outputs = tokenizer.batch_decode(generate_ids.sequences[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    for i, ans in enumerate(outputs):
        if ans.strip() == "":
            outputs[i] = 'N/A'
    return outputs

