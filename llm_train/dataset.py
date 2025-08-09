import os
import json
import torch
import transformers
# from transformers import DataCollatorForLanguageModeling
from typing import Dict
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def rank0_print(*args):
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print(*args)


class BaseDataset(Dataset):
    """Dataset for supervised fine-tuning LLama3.1, main difference is preprocessing. """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(BaseDataset, self).__init__()
        ...

    def __len__(self):
        ...

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ...
    
    def _preprocess(self, 
            sources, 
            tokenizer: transformers.PreTrainedTokenizer, 
            max_len: int,
            system_message: str = "You are a helpful assistant. "
    ) -> Dict:
        begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
        start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
        end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
        eot_id = tokenizer.get_vocab()["<|eot_id|>"]
        # nl_tokens = tokenizer('\n').input_ids
        nl_tokens = tokenizer('\n\n').input_ids

        _system = tokenizer('system').input_ids
        _user = tokenizer('user').input_ids
        _assistant = tokenizer('assistant').input_ids

        # Apply prompt templates
        input_ids, targets = [], []
        for _ , source in enumerate(sources):
            input_id, target = [], []

            system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(system_message).input_ids + [eot_id]
            input_id += system
            target += [IGNORE_TOKEN_ID] * len(input_id)

            assert len(input_id) == len(target)
            for _ , sentence in enumerate(source):
                role = sentence["from"]
                value = sentence["value"]
                if role == 'user':
                    _input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(value).input_ids + [eot_id]
                    _target = [IGNORE_TOKEN_ID] * len(_input_id)
                elif role == 'assistant':
                    _input_id = (
                        [start_header_id] + _assistant + [end_header_id] +  nl_tokens + 
                        tokenizer(value).input_ids + [eot_id]
                    )
                    _target = (
                        [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(_assistant) +  [IGNORE_TOKEN_ID] +  [IGNORE_TOKEN_ID] * len(nl_tokens) + 
                        tokenizer(value).input_ids  + [eot_id]
                    )

                else:
                    raise NotImplementedError
                input_id += _input_id
                target += _target

            assert len(input_id) == len(target)
            input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            target += [IGNORE_TOKEN_ID] * (max_len - len(target))
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        return dict(
            input_ids = input_ids,
            labels = targets,
            attention_mask = input_ids.ne(tokenizer.pad_token_id),
        )



class SupervisedDataset(BaseDataset):
    """Dataset for supervised fine-tuning. """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__(raw_data, tokenizer, max_len)

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = self._preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(BaseDataset):
    """Dataset for supervised fine-tuning. """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
    
        super(LazySupervisedDataset, self).__init__(raw_data, tokenizer, max_len)
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = self._preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret
        return ret


@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
            
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_TOKEN_ID)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        return batch
    

def make_supervised_data_module( tokenizer: transformers.PreTrainedTokenizer, data_args, max_len, ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    dataset_cls = ( LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset = train_dataset, 
        eval_dataset = eval_dataset,
        data_collator = data_collator
    )
