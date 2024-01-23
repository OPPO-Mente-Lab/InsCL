'''
Copyright 2024 OPPO. All rights reserved.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
'''

from sklearn.model_selection import train_test_split
import jsonlines
import os
from tqdm import tqdm
import json
import torch
from sentence_transformers import SentenceTransformer
import random
import numpy as np
import pickle
import ot
from tag_utils import get_InsInfo_dic
import argparse


def create_folder(folderpath):
    """
    create a new folder
    """
    if not os.path.exists(folderpath):
        print("folderpath not exist, so create the dir")
        os.mkdir(folderpath)

def clear_folder(filepath):
    """
    Delete all files or folders in a directory
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def write_jsonl(target_path, data):
    """
    Write the elements in the list to the jsonl file line by line
    """
    for d in tqdm(data):
        with jsonlines.open(target_path, mode='a') as c_f:
            jsonlines.Writer.write(c_f, d)

def normalization(data):
    """
    Normalize a list
    """
    sum_value = sum(data)
    norm_data = [ i/sum_value for i in data]
    return norm_data

def pW_cal(a, b, p=1, metric='cosine'):
    """ 
    calculate Wasserstein Distance with uniform distribution assumption
    Args:
        a, b: samples sets drawn from α,β respectively
        p: the coefficient in the OT cost (i.e., the p in p-Wasserstein)
        metric: the metric to compute cost matrix, 'euclidean' or 'cosine'
    """
    # cost matrix
    M = ot.dist(a, b, metric=metric)
    M = pow(M, p)
    # uniform distribution assumption
    alpha = ot.unif(len(a))
    beta = ot.unif(len(b))
    # p-Wasserstein Distance
    pW = ot.emd2(alpha, beta, M, numItermax=100000)
    pW = pow(pW, 1/p)
    return pW

def pW_cal_dist(a, b, a_dist, b_dist, p, metric='euclidean'):
    """ 
    calculate Wasserstein Distance with real distribution
    Args:
        a, b: samples sets drawn from α,β respectively
        a_dist, b_dist: the real distribuution of samples sets
        p: the coefficient in the OT cost (i.e., the p in p-Wasserstein)
        metric: the metric to compute cost matrix, 'euclidean' or 'cosine'
    """
    # cost matrix
    M = ot.dist(a, b, metric=metric)
    M = pow(M, p)
    # real distribution
    alpha = a_dist
    beta = b_dist
    # p-Wasserstein Distance
    pW = ot.emd2(alpha, beta, M, numItermax=100000)
    pW = pow(pW, 1/p)
    return pW

def initial_replay_num(replay_num, loop_list):
    """
    get the number of replay data corresponding to each task
    Args:
        replay_nun: initialize a fixed number of replaying
        loop_list: sequential task list
    """
    initial_num_dic = {}
    for i in range(0, len(loop_list)):
        initial_num_dic[loop_list[i]] = replay_num
    return initial_num_dic

def cal_dynamic_num(emb_file_path, cur_idx, initial_num_dic, loop_list, style):
    """
    calculate dynamic replay number for previous tasks
    Args:
        emb_file_path: file that stores original instructions, embeddings and distributions
        cur_idx: current task index
        initial_num_dic: a dic stores fixed replay number for each task
        loop_list: sequential task list
        style: the replay style
    """
    # calculate the total amount of replay data in previous tasks
    pre_total_num = 0
    for c in loop_list[:cur_idx]:
        pre_total_num += initial_num_dic[c]

    # calculate Wasserstein Distance with instruction embeddings and distributions
    # here we load prepared file that stores original instructions, embeddings and distributions  
    pW_ls = []
    with open(emb_file_path, mode='rb') as file:
        emb_dic = pickle.load(file)
    a = emb_dic[loop_list[cur_idx]]['encoded']
    for i in range(cur_idx):
        b = emb_dic[loop_list[i]]['encoded']
        # Wasserstein Distance
        if 'dist' in style:
            # calculate Wasserstein Distance with real distribution
            a_dist = emb_dic[loop_list[cur_idx]]['distribution']
            b_dist = emb_dic[loop_list[i]]['distribution']
            pW = pW_cal_dist(a, b, a_dist, b_dist, p=1, metric='cosine')
        else:
            # calculate Wasserstein Distance with uniform distribution assumption
            pW = pW_cal(a, b, p=1, metric='cosine')
        pW_ls.append(pW)
    norm_pW = normalization(pW_ls)
    dynamic_num_ls = [int(w*pre_total_num) for w in norm_pW]
    sorted_norm_pW = sorted(norm_pW)
    # make sure the totals consistent with the baseline
    # delete
    if sum(dynamic_num_ls) > pre_total_num:
        del_num = sum(dynamic_num_ls) - pre_total_num
        for w in sorted_norm_pW[:del_num]:
            del_idx = norm_pW.index(w)
            dynamic_num_ls[del_idx] = dynamic_num_ls[del_idx]-1
    # add
    if sum(dynamic_num_ls) < pre_total_num:
        add_num = pre_total_num - sum(dynamic_num_ls)
        for w in sorted_norm_pW[-add_num:]:
            add_idx = norm_pW.index(w)
            dynamic_num_ls[add_idx] = dynamic_num_ls[add_idx]+1
    assert sum(dynamic_num_ls) == pre_total_num, 'inconsistent total number!'
    print('sum_dynamic_num', sum(dynamic_num_ls), 'dynamic_num_ls', dynamic_num_ls)
    return dynamic_num_ls


def sample_example(ins_tag_dic, task, data_list, num):
    """
    sample the specified amount of corresponding instruction data
    Args:
        ins_tag_dic: store 'ins_list' and InsInfo 'score_list'
        task: task name
        data_list: a list of specified task data
        num: respective replay number of the specified task
    """
    ins_ls = ins_tag_dic[task]['ins_list']          # instruction list
    InsInfo_rate = ins_tag_dic[task]['score_list']  # normalized InsInfo score list, arranged from large to small
    single_ins_ls = []                              # store instructions containing only one piece of data
    example = []                                    # store sampled data
    
    # when the number of instructions is larger than input num
    # select the top-num instructions ranked by InsInfo
    if num < len(ins_ls):
        top_ins = []
        for ins in ins_ls[:num]:
            top_ins.append(ins)
        random.shuffle(data_list)
        for d in data_list:
            if d['instruction'][0] in top_ins:
                top_ins.remove(d['instruction'][0])
                example.append(d)       
    else:
        for i in range(len(ins_ls)):
            InsInfo_num = int(InsInfo_rate[i]*num)
            if InsInfo_num < 1:
                InsInfo_num = 1
                single_ins_ls.append(ins_ls[i])     # instruction with only one data
            ins_data = [item for item in data_list if item['instruction'][0] == ins_ls[i]]
            random.shuffle(ins_data)
            ins_example = random.sample(ins_data, InsInfo_num)
            example.extend(ins_example)
        # make sure the totals consistent with the respective replay number
        print('current length:', len(example))
        random.shuffle(data_list)
        # add
        if len(example) < num:
            print('start add data!')
            for d in data_list:
                if len(example) == num:
                    break
                if (d not in example) and (d['instruction'][0] in ins_ls): # Make sure the supplement is in instruction list
                    example.append(d)
        # delete
        random.shuffle(example)
        del_idx = 0
        if len(example) > num:
            print('start delete data!')
            for i in range(len(example)):
                e = example[del_idx]
                if len(example) == num:
                    break
                else:
                    if e['instruction'][0] not in single_ins_ls:
                        example.remove(e)
                        ins_data = [item for item in example if item['instruction'][0] == e['instruction'][0]]
                        # do not delete only one piece of corresponding instruction data
                        if len(ins_data) == 1:
                            single_ins_ls.append(e['instruction'][0])
                    else:
                        del_idx += 1  
    assert len(example) == num, f'inconsistent total number for {len(example)} examples with {num} you need!'
    return example

def InsCL(emb_file_path, random_seed, replay_num, src_dir_path, loop_list, style, eps, min_samples):
    """
    dynamic replay with InsInfo-guided sampling
    Args:
        radom_seed: seed for random sampling
        raplay_num: initialized fixed number of replaying
        src_dir: the folder path of initial trainig data
        loop_list: sequential task list
        style: define the replay sequence and method
        eps: the parameter of DBSCAN, find the points in the ε (eps) neighborhood of every point
        min_samples: the parameter of DBSCAN, identify the core points with more than min_samples neighbors
    """
    model = SentenceTransformer('whaleloops/phrase-bert')    # load from https://huggingface.co/whaleloops/phrase-bert
    initial_num_dic = initial_replay_num(replay_num, loop_list)     # task_name: initial replay number
    target_dir = f'replay_data/InsCL_{style}_seed{random_seed}_num{replay_num}_eps{eps}_{min_samples}'
    create_folder(target_dir)
    clear_folder(target_dir)
    for cur_idx in range(1, len(loop_list)):
        dynamic_num_ls = cal_dynamic_num(cur_idx, initial_num_dic, loop_list, style)
        # calculate the InsInfo score for each instruction
        ins_tag_dic = get_InsInfo_dic(loop_list, cur_idx, model, eps, min_samples)
        target_path = f'{target_dir}/{loop_list[cur_idx]}.jsonl' # target path
        example = []
        for i in range(cur_idx):
            master_category = f'{loop_list[i]}.jsonl'
            master_category_path = src_dir_path+'/' + master_category
            with open(master_category_path, 'r') as f:
                data = jsonlines.Reader(f)
                data_list = []
                for d in data:
                    data_list.append(d)
            num = dynamic_num_ls[i]     # the replay number of the corresponding task
            sub_example = sample_example(ins_tag_dic, loop_list[i], data_list, num)
            example.extend(sub_example)
        write_jsonl(target_path, example)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InsCL sampling')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--replay_num', type=int, default=200, help='the amount of replay data for each task')
    parser.add_argument('--src_train_path', type=str, default='train', help='the original training data path')
    parser.add_argument('--eps', type=float, default=0.1, help='the parameter of DBSCAN, find the points in the ε (eps) neighborhood of every point')
    parser.add_argument('--min_samples', type=int, default=5, help='the parameter of DBSCAN, identify the core points with more than min_samples neighbors')
    parser.add_argument('--style', type=str, default="curriculum_pWdist",
                        choices=['curriculum', 'curriculum_pWdist'], help="end up with pWdist if calculate Wasserstein Distance with real distribution")
    parser.add_argument('--emb_file_path', type=str, default='encoded_ins_dist.pkl', help='file that stores original instructions, embeddings and distributions')
    args = parser.parse_args()

    random_seed = args.seed
    print("Random Seed: ", random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    replay_num = args.replay_num
    src_dir_path = args.src_train_path
    eps = args.eps
    min_samples = args.min_samples
    style = args.style # end with pWdist to use the real distribution
    if 'random' in style:
        loop_list = ['Text_Quality_Evaluation', 'Mathematics', 'Dialogue', 'Summarization', 'Extraction', 'Misc', 'Closed_QA', 'Code', 'Rewriting', 'Detection', 'Comprehension', 'Sentiment_Analysis', 'Open_QA', 'Program_Execution', 'Generation', 'Classification']
    if 'curriculum' in style:
        loop_list = ['Classification', 'Text_Quality_Evaluation', 'Code', 'Detection', 'Sentiment_Analysis', 'Comprehension', 'Closed_QA', 'Extraction', 'Dialogue', 'Program_Execution', 'Rewriting', 'Open_QA', 'Misc', 'Generation', 'Summarization', 'Mathematics']
    if 'difficult' in style:
        loop_list = ['Mathematics', 'Summarization', 'Generation', 'Misc', 'Open_QA', 'Rewriting', 'Program_Execution', 'Dialogue', 'Extraction', 'Closed_QA', 'Comprehension', 'Sentiment_Analysis', 'Detection', 'Code', 'Text_Quality_Evaluation', 'Classification']
    emb_file_path = args.emb_file_path 

    InsCL(emb_file_path, random_seed, replay_num, src_dir_path, loop_list, style, eps, min_samples)