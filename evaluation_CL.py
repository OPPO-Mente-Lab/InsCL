# encoding: utf-8
'''
Copyright 2024 OPPO. All rights reserved.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
'''

import argparse
import os
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import copy
import logging
import jsonlines
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
# modify print function
import builtins as __builtin__
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import utils
from train import smart_tokenizer_and_embedding_resize, TrainingArguments,DataArguments, ModelArguments
from tqdm import tqdm
from codecs import open
from utils import compute_metric



IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n\n{input}"
    ),
    "prompt_no_input": (
        "{instruction}"
    ),
}


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = result_dir+'_%s_rank%d.json'%(filename, torch.distributed.get_rank())
    final_result_file = result_dir+'_%s.json'%filename
    json.dump(result,open(result_file,'w', encoding="utf-8"), ensure_ascii=False, indent=4)
    dist.barrier()
    if dist.get_rank() == 0: 
        # combine results from all processes
        result = []
        for rank in range(dist.get_world_size()):
            result_file = result_dir + '_%s_rank%d.json'%(filename,rank)
            res = json.load(open(result_file,'r', encoding="utf-8"))
            result += res
        # unique all
        final_result = {e['idx']: e for e in result}
        final = list(final_result.values())
        final.sort(key=lambda e: int(e["idx"]))
        print("length is:", len(final))
        json.dump(final, open(final_result_file,'w', encoding="utf-8"), ensure_ascii=False, indent=4)            
        print('result file saved to %s'%final_result_file)
    return final_result_file


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess_batch(source, target, tokenizer):
    # examples, sources = [source + target], [source]
    # examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    sources_tokenized = _tokenize_fn([source], tokenizer)
    input_ids = sources_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    # for i, (label, source_len) in enumerate(zip(labels, sources_tokenized["input_ids_lens"])):
    #     labels[i][:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels, input_ids_lens=sources_tokenized["input_ids_lens"])

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer: transformers.PreTrainedTokenizer, mode="chat"):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.mode = mode
        logging.info("Data convert mode: {}".format(mode))
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        print("Formatting inputs...Skip in lazy mode")
        self.raw_data = list_data_dict
        # self.cached_data_dict = {}
        self.prompt_input, self.prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # if i in self.cached_data_dict:
        #     return self.cached_data_dict[i]
        example = self.raw_data[i]
        source = self.prompt_input.format_map(example) if example.get("input", "") != "" else self.prompt_no_input.format_map(example)
        target = "{}{}".format(example.get('output', ""), self.tokenizer.eos_token)
        if self.mode == "instruction":
            pass  # default
        elif self.mode == "chat":
            if "Human" not in source and "Assistant" not in source:
                source = "Human: {} \n\nAssistant: ".format(source)

        
        ret = preprocess_batch(source, target, self.tokenizer)
        ret = dict(
            uid = str(i),
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0]
            # input_ids_lens=ret["input_ids_lens"][0],
        )
        # self.cached_data_dict[i] = ret
        return ret


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        uids = [instance["uid"] for instance in instances]
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = utils.pad_sequence(
            input_ids, padding_value=self.tokenizer.pad_token_id, padding_left=self.tokenizer.padding_side=='left'
        )
        labels = utils.pad_sequence(labels, padding_value=IGNORE_INDEX)
        return dict(
            uids=uids,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

@torch.no_grad()
def evaluate(model, data_loader, device, tokenizer):
    # evaluate
    model.eval() 
    # config
    num_beams = 5
    max_length = 2048
    do_sample = False
    targets = []
    sources = []
    idxs = []
    for i, batch in enumerate(data_loader):
        uids = batch.get("uids")
        input_ids, attention_mask = batch.get("input_ids").cuda(), batch.get("attention_mask").cuda()
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=False, top_k=5, top_p=0.3, temperature=0.9, min_length=1, max_length=2048)
        sequence_lens = attention_mask.sum(dim=1)
        input_size = input_ids.size()[1]
        sources += [tokenizer.decode(e[:input_size], skip_special_tokens=True) for idx, e in enumerate(output)]
        targets += [tokenizer.decode(e[input_size:], skip_special_tokens=True) for idx, e in enumerate(output)]
        idxs += uids
        if i%10 == 0 and dist.get_rank()==0:
            print("current step:{} ".format(i))
    output = [{"source": src, "target": tgt, "idx": idx} for src, tgt, idx in zip(sources, targets, idxs)]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='./checkpoints')
    parser.add_argument('--cache_dir', default='./')
    parser.add_argument('--input_file', default='input.txt')
    parser.add_argument('--output_merge_dir', default='test_tmp')  
    parser.add_argument('--output_file', default='output.txt')        
    parser.add_argument('--output_dir', default='tmp')        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--model_max_length', default=2048, type=int)
    parser.add_argument('--mode', default="instruction", type=str, choices=["chat", "instruction"])
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')    
    parser.add_argument('--batch_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    # dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400)) 
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.distributed.init_process_group(backend="NCCL", init_method=args.dist_url, \
                                         world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=5400))

    # builtin_print = __builtin__.print
    # def print(*args, **kwargs):
    #     force = kwargs.pop('force', False)
    #     if torch.distributed.get_rank()==0 or force:
    #         builtin_print(*args, **kwargs)
    # __builtin__.print = print
    if torch.distributed.is_available() and dist.get_rank() == 0:
        print("distribution initlized, world_size-{}".format(torch.distributed.get_world_size()))
    
    device = torch.device(local_rank)
    torch.cuda.set_device(device)
    torch.distributed.barrier()

    # fix the seed for reproducibility
    if dist.get_rank() == 0:
        print("seed everything")
    seed = args.seed + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # model input
    print("load model ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )
    model.config.use_cache=True
    print("load tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.add_eos_token=False
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.pad_token=tokenizer.eos_token

    #### Dataset #### 
    print("Creating dataset")
    test_dataset = LazySupervisedDataset(args.input_file, tokenizer, args.mode)
    print('dataset length:', len(test_dataset.raw_data))
    if args.distributed:
        num_tasks = world_size
        global_rank = torch.distributed.get_rank()            
        sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler = None
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=world_size,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
            collate_fn=data_collator,
            drop_last=False,
        )      

    print('dataset length:', len(test_loader.dataset.raw_data))
    model = model.half().to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.distributed.get_rank()])
        model_without_ddp = model.module    
    
    test_result = evaluate(model_without_ddp, test_loader, device, tokenizer)  
    if not os.path.exists(args.output_merge_dir):
        os.mkdir(args.output_merge_dir)
    
    # model dir
    output_dir_path = os.path.join(args.output_merge_dir, args.output_dir)
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    output_file_path = os.path.join(output_dir_path, args.output_file)
    test_result_file = save_result(test_result, output_file_path, 'test')

    dist.barrier()
    # torch.distributed.barrier()
    if dist.get_rank() == 0:
        predict_datas = json.load(open(test_result_file, 'r', encoding="utf-8"))
        golden_datas = utils.jload(args.input_file)
        assert len(predict_datas) == len(golden_datas)
        for i, (p, g) in enumerate(zip(predict_datas, golden_datas)):
            predict_datas[i]["std_answer"] = g["output"]
        with open(os.path.join(output_dir_path, f"{args.output_file}_total.output.json"), 'w', encoding="utf-8") as f:
            json.dump(predict_datas, f, indent=4, ensure_ascii=False)

    if dist.get_rank() == 0:
        rouge_result = utils.compute_metric(test_result_file, args.input_file)
        rouge_result_path = f'{args.output_merge_dir}/output_state.jsonl'
        item = {}
        with jsonlines.open(rouge_result_path, mode='a') as file:
            item['data_category'] = args.output_file
            item['model_category'] = args.output_dir
            item['rouge-1'] = rouge_result['rouge-1']
            item['rouge-2'] = rouge_result['rouge-2']
            item['rouge-l'] = rouge_result['rouge-l']
            jsonlines.Writer.write(file, item)
        print(f"save rouge result to: {args.output_merge_dir}/output_state.jsonl")
        print(f"{args.output_file}_rouge_result with {args.output_dir}: {rouge_result}")

        # setlog.set_logger(f"{output_dir_path}/{args.output_dir}_state.log")
        # print(f"save rouge result to: {output_dir_path}/{args.output_dir}_state.log")
        # logging.info(f"{args.output_file}_rouge_result with {args.output_dir}: {rouge_result}")
        # print("rouge_result: ", rouge_result)
        

if __name__ == '__main__':
    main()
