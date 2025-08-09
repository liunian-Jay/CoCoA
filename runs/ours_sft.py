from tqdm import tqdm
from llm_inference.build_prompts import llm_prompts
from llm_inference.inference_vllm import load_model_vllm, llm_response_vllm
from evaluation import get_evaluation
from utils import load_data, preprocess_ARC_Challenge, preprocess_PubHealth


input_prompt = (
    "### Instruction:\n"
    "1. First, provide background for the question. Write a passage that relevant about the question only based on your knowledge.\n"
    "2. Sencond, refer to the provided passages to generate a summary. Cite and Write a passage that relevant the question only based on the provided passages.\n"
    "3. Third, refer to the information from the above two sources, verify the accuracy of the facts and the consistency of the logic, and predict the final answer. "


    # "Is the following statement correct or not? Say true if it's correct; otherwise, say false. Don't capitalize or add periods, just say \"true\" or \"false\"."
    # "Given four answer candidates, A, B, C and D, choose the best answer choice. Please answer with the capitalized alphabet only, without adding any extra phrase or period."
    

    "### Passages:\n {passages}\n\n"
    "### Question:\n {question}\n"

    "### Generate Format:\n"
    "<Internal>\nxxx (your background based on your knowledge)\n<\\Internal>\n"
    "<External>\nxxx (your summary based on the provided passages)\n<\\External>\n"
    "<Thinking>\nxxx\n<\\Thinking>\n"
    "<Answer>\nxxx (your short answer consisting of only a few words)<\\Answer>\n"
)


def build_prompts(item, k=5):
    evidence = '\n'.join([f'{passage["text"]}\n\n' for i, passage in enumerate(item['ctxs'][:k])])
    prompt = input_prompt.format_map({'question': item['question'], 'passages': evidence})
    return prompt


def llm_generation(inputs, model, tokenizer, lm_type='Llama-3.1-8B-Instruct', batch_size = 512):
    results = []
    for start in tqdm(range(0, len(inputs), batch_size)):
        end = min(start+batch_size, len(inputs))
        batch_inputs = inputs[start:end]

        tokens = llm_prompts(lm_type, batch_inputs, tokenizer, tokenize=False)
        # responses = llm_response_vllm(model, tokens, max_tokens=2048)
        responses = llm_response_vllm(model, tokens, max_tokens=4096)

        results += responses
    return results


def main(data_path, model, tokenizer, lm_type):
    k=5 # TODO
    data = load_data(data_path)#[:2]

    # data = preprocess_PubHealth(data)
    data = preprocess_ARC_Challenge(data)



    inputs = [build_prompts(item,k=k) for item in data]    
    results = llm_generation(inputs, model, tokenizer, lm_type)
    error_num = 0
    for item, res in zip(data, results):
        # print(res)
        try:
            item['final_answer'] = res.split("<Answer>")[1].split("<\\Answer>")[0].strip()
        except:
            item['final_answer'] = res
            error_num += 1
    print(error_num)
    print('l'*100)  

    EM_scores, F1_scores = 0,0
    for item in data:
        em,f1 = get_evaluation('ARC_Challenge', item['final_answer'], item['answers'])
        EM_scores += em
        F1_scores += f1

    print(data_path)
    print('Results:', 'EM:', round(EM_scores / len(data) * 100, 3), 'F1:', round(F1_scores / len(data) * 100, 3))



from transformers import set_seed
set_seed(2025)
if __name__ == '__main__':
    # lm_type = '/project_path/model_outputs/sft/sft_long_2/5epoch'
    lm_type = '/project_path/model_outputs/dpo/dpo_long2'
    # lm_type = 'Llama-3.1-8B-Instruct'
    model, tokenizer = load_model_vllm(lm_type,max_model_len=4096*2)
    # for data_name in ['2WikiMultiHopQA','HotpotQA','WebQuestions','NaturalQA','PopQA_longtail','TriviaQA','SQuAD']:
    # for data_name in ['PubHealth']:   
    for data_name in ['ARC_Challenge']:             
        data_path = f'/path/data/RAG/my_ctx_contriever/{data_name}/{data_name}.jsonl'
        main(data_path, model, tokenizer, lm_type)
        break