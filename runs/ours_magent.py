external_answer_prompt = (
    "### Passages:\n {passages}\n\n"
    "### Instruction:\n Answer the question below concisely in a few words.\n\n"
    "### Input:\n{question}\n"
)
internal_answer_prompt = (
    "### Instruction:\n Answer the question below concisely in a few words.\n\n"
    "### Input:\n{question}\n"
)

external_summary_prompt = (
    "### Instruction:\n Refer to the provided passages to generate a summary that meets the following conditions:\n"
    "1. Cite and Write a passage that can support the prediction about the question only based on the provided passages.\n"
    "2. No more than 200 words.\n"
    "3. Do not respond with anything other than the \"Summary\".\n"

    "### Passages:\n {passages}\n\n"
    "### Question:\n {question}\n"
    "### Prediction:\n {answer}\n\n"

    "### Generate Format:\n"
    "### Summary: xxx\n"
)
internal_recall_prompt = (
    "### Instruction:\n Please provide background for the question that meets the following conditions:\n"
    "1. Write a passage that can support the prediction about the question only based on your knowledge.\n"
    "2. No more than 200 words.\n"
    "3. Do not respond with anything other than the \"Background\".\n"

    "### Question:\n {question}\n"

    "### Prediction:\n {answer}\n\n"

    "### Generate Format:\n"
    "### Background: xxx\n"
)


thinking2_prompt = (
    "### Internal Reasoning Path: \n {passage_in}\n\n ### Internal Prediction 1: \n {answer_in}\n\n \"\"\" "
    "### External Reasoning Path: \n {passage_out}\n\n ### External Prediction 2: \n {answer_out}\n\n \"\"\" "
    "### Instruction:\n "
    "Refer to the information from the above two sources, verify the accuracy of the facts and the consistency of the logic, and choose the best prediction. "
    
    "### Question:\n {question}\n"

    "### Generate Format:\n"
    "### Thingking: xxx (Please think step by step)\n"
    "### Short Answer: xxx (just in a few words)\n"
)


def build_external(item, k=5):
    evidence = '\n'.join([f'{passage["text"]}\n\n' for passage in item['ctxs'][:k]])
    prompt = external_answer_prompt.format_map({'question': item['question'], 'passages': evidence})
    return prompt
def build_summary(item, k=5):
    evidence = '\n'.join([f'{passage["text"]}\n\n' for i, passage in enumerate(item['ctxs'][:k])])
    prompt = external_summary_prompt.format_map({'question': item['question'], 'passages': evidence, 'answer': item['answer_out']})
    return prompt

def build_internal(item):
    prompt = internal_answer_prompt.format_map({'question': item['question']})
    return prompt
def build_recall(item):
    prompt = internal_recall_prompt.format_map({'question': item['question'], 'answer': item['answer_in']})
    return prompt

def build_thinking2(item):
    prompt = thinking2_prompt.format_map({
        'question': item['question'], 
        'passage_in': item['passage_in'], 
        'answer_in': item['answer_in'],
        'passage_out': item['passage_out'],
        'answer_out': item['answer_out'],
    })
    return prompt


from tqdm import tqdm
from llm_inference.build_prompts import llm_prompts
from llm_inference.inference_vllm import load_model_vllm, llm_response_vllm
from evaluation import get_evaluation
from utils import load_data, preprocess_ARC_Challenge, preprocess_PubHealth

def llm_generation(inputs, model, tokenizer, lm_type='Llama-3.1-8B-Instruct', batch_size = 256):
    results = []
    for start in tqdm(range(0, len(inputs), batch_size)):
        end = min(start+batch_size, len(inputs))
        batch_inputs = inputs[start:end]

        tokens = llm_prompts(lm_type, batch_inputs, tokenizer, tokenize=False)
        responses = llm_response_vllm(model, tokens, max_tokens=2048*2)
        results += responses
    return results


def external(data, model, tokenizer):
    inputs = [build_external(item) for item in data]
    results = llm_generation(inputs, model, tokenizer)
    for item, res in zip(data, results):
        item['answer_out'] = res

def internal(data, model, tokenizer):
    inputs = [build_internal(item) for item in data]
    results = llm_generation(inputs, model, tokenizer)
    for item, res in zip(data, results):
        item['answer_in'] = res


def summary(data, model, tokenizer):
    inputs = [build_summary(item) for item in data]
    results = llm_generation(inputs, model, tokenizer)
    for item, res in zip(data, results):

        if "### Summary:" in res:
            item['passage_out'] = res.split('### Summary:')[1].strip()
        elif "Summary:" in res:
            item['passage_out'] = res.split('Summary:')[1].strip()
        else:
            item['passage_out'] = res.strip()

def recall(data, model, tokenizer):
    inputs = [build_recall(item) for item in data]
    results = llm_generation(inputs, model, tokenizer)
    for item, res in zip(data, results):
        if "### Background:" in res:
            item['passage_in'] = res.split('### Background:')[1].strip()
        elif "Background:"  in res:
            item['passage_in'] = res.split('Background:')[1].strip()
        else:
            item['passage_in'] = res.strip()


def thinking2(data, model, tokenizer):
    inputs = [build_thinking2(item) for item in data]
    results = llm_generation(inputs, model, tokenizer)

    error_num = 0
    for item, res in zip(data, results):
        print(res)
        try:
            # item['thinking2'] = res.split("### Thinking:")[1].split("### Short Answer:")[0].strip()
            # item['final_answer'] = res.split("### Short Answer:")[1].split("### Source:")[0].strip()

            if "### Short Answer:" in res:
                item['final_answer'] = res.split("### Short Answer:")[1].strip()
            
            elif "short answer:" in res:
                item['final_answer'] = res.split("short answer:")[1].strip()
            
            else:
                item['thinking2'] = res.split("### Thinking:")[1].split("### Short Answer:")[0].strip()
                item['final_answer'] = res.split("### Short Answer:")[1].strip()

        except:
            item['final_answer'] = res
            error_num += 1
    print(error_num)



def main(data_path, model, tokenizer):
    data = load_data(data_path) # [:5]

    data = preprocess_PubHealth(data)

    external(data, model, tokenizer)
    print('1'*100)
    summary(data, model, tokenizer)
    print('2'*100)
    internal(data, model, tokenizer)
    print('3'*100)
    recall(data, model, tokenizer)
    print('4'*100)
    thinking2(data, model, tokenizer)
    print('5'*100)
 
    EM_scores, F1_scores = 0,0
    for item in data:
        em,f1 = get_evaluation('', item['final_answer'], item['answers'])
        EM_scores += em
        F1_scores += f1

    print(data_path)
    print('Results:', 'EM:', round(EM_scores / len(data) * 100, 3), 'F1:', round(F1_scores / len(data) * 100, 3))


from transformers import set_seed
set_seed(2025)
if __name__ == '__main__':
    lm_type = 'Llama-3.1-8B-Instruct'
    # lm_type = 'Llama-3.1-70B-Instruct'
    model, tokenizer = load_model_vllm(lm_type,max_model_len = 2048*4)
    # for data_name in ['HotpotQA','2WikiMultiHopQA','PopQA_longtail','WebQuestions','NaturalQA','TriviaQA']:  
    # for data_name in ['2WikiMultiHopQA','HotpotQA','WebQuestions']:  

    for data_name in ['PubHealth']:   
    # for data_name in ['ARC_Challenge']:          
        data_path = f'/path/data/RAG/my_ctx_contriever/{data_name}/{data_name}.jsonl'
        main(data_path, model, tokenizer)
 