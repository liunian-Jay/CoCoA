import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(results, path_output):
    with open(path_output, "w", encoding="utf-8") as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + "\n")
            
def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r", encoding="utf-8") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


import re

def postprocess_answer(answer):
    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")
    
    answer = answer.strip('"')
    answer = answer.strip('\'')
    # answer = answer.strip('.')
    answer = answer.strip('[')
    answer = answer.strip(']')
    answer = answer.strip('*')
    return answer

def extract_by_tags(text, tags):
    """
    Extract segments from text based on a list of tags.
    Returns: list of str: Extracted content between each pair of tags or between a tag and the end of text.
    """
    contents = []
    for i in range(len(tags)):
        if i < len(tags) - 1:
            pattern = fr'\[?{tags[i]}\]?:\s*(.*?)(?=\s*\[?{tags[i + 1]}\]?:|$)'
        else:
            pattern = fr'\[?{tags[i]}\]?:\s*(.*)'
        match = re.search(pattern, text, re.DOTALL)
        contents.append(match.group(1) if match else '[N/A]')
    return contents

def extract_by_tags(text, tags):
    """
    Extract segments from text based on a list of tags.
    Returns: list of str: Extracted content between each pair of tags or between a tag and the end of text.
    """
    contents = []
    for i in range(len(tags)):
        if i < len(tags) - 1:
            pattern = fr'\[?{tags[i]}\]?:\s*(.*?)(?=\s*\[?{tags[i + 1]}\]?:|$)'
        else:
            pattern = fr'\[?{tags[i]}\]?:\s*(.*)'
        match = re.search(pattern, text, re.DOTALL)
        contents.append(match.group(1) if match else '[N/A]')
    return contents




########################################################
########################################################
########################################################

def preprocess_ARC_Challenge(data):
    choice_dict = {'A':'A','B':'B','C':'C','D':'D','E':'E','1':'A','2':'B','3':'C','4':'D','5':'E'}
    new_data = []
    for item in data:
        choice_text = ''
        choices = item["choices"]
        for i in range(len(choices["label"])):
            answer_key = choices["label"][i]
            answer_key = choice_dict[answer_key]
            text = choices["text"][i]
            choice_text += "\n{0}: {1}".format(answer_key, text)

        item["question"] = item["question"] + choice_text
        item["answers"] = [item["answerKey"]]
        new_data.append(item)
    return new_data

def preprocess_PubHealth(data):
    new_data = []
    for item in data:
        answer = 'true' if item['label']=='SUPPORTS' else 'false'
        item["question"] = item["claim"]
        item["answers"] = [answer]
        new_data.append(item)
    return new_data




