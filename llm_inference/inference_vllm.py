import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer



def load_model_vllm(lm_type, max_model_len = 2048):
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
    elif lm_type == 'DeepSeek-R1-Distill-Qwen-7B':
        model_name_or_path = os.path.abspath("/home/share/models/DeepSeek-R1-Distill-Qwen-7B")  
    elif lm_type == 'DeepSeek-R1-Distill-Llama-8B':
        model_name_or_path = os.path.abspath("/home/share/models/DeepSeek-R1-Distill-Llama-8B")  
    else:
        model_name_or_path = os.path.abspath(lm_type) 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # llm = LLM(model=model_name_or_path, tensor_parallel_size=torch.cuda.device_count(), max_model_len = 2048, gpu_memory_utilization=0.85)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=torch.cuda.device_count(), max_model_len = max_model_len)
    return llm, tokenizer


def llm_response_vllm(llm, prompts, max_tokens=2048):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens = max_tokens)
    outputs = llm.generate(prompts, sampling_params, )

    res = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        res.append(generated_text)
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return res