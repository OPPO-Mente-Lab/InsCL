'''
Copyright 2024 OPPO. All rights reserved.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
'''

import json
import pickle
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import math

# obtain pos tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def norm_phrase(phrase):
    tokens = word_tokenize(phrase)      # split phrase into words
    tagged_sent = pos_tag(tokens)       # obtain pos tags
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
    new_string = ' '.join(lemmas_sent)
    return new_string

def normalization(data):
    """
    Normalize a list
    """
    sum_value = sum(data)
    norm_data = [ i/sum_value for i in data]
    return norm_data

def tag_dedup(loop_list, label_tag_info, ins_tag):
    """
    deduplicate the tag corresponding to the specified label
    Args:
        loop_list: sequential task list
        label_tag_info: tags with the corresponding clustering label and frequncy
        ins_tag: a dictionary that stores the instructions and corresponding tags for each task
    """
    ins_tag_dedup = {}
    label_tag_info = sorted(label_tag_info, key = lambda x: x['count'], reverse=True)     # arrange in order from largest to smallest frequency
    label_tag_ls = [i['original'] for i in label_tag_info]
    for task in loop_list:
        new_item_ls = []
        for item in ins_tag[task]:
            instruction = item['instruction']
            tag_ls = item['tag_list']
            new_tag_ls = []
            for tag in tag_ls:
                if tag in label_tag_ls:
                    tag = label_tag_ls[0]   # For the current clustering label, use the tag with the highest frequency to replace the corresponding instance.
                if tag not in new_tag_ls:
                    new_tag_ls.append(tag)
            new_item_ls.append({"instruction": instruction, "tag_list": new_tag_ls})
        ins_tag_dedup[task] = new_item_ls
    return ins_tag_dedup


def get_InsInfo_dic(loop_list, cur_step, model, eps, min_samples):
    """
    return normalized InsInfo scores for the current training stage
    Args:
        loop_list: sequential task list
        cur_step: the order of current task (start from 0)
        model: the model for encoding tags
        eps: the parameter of DBSCAN, find the points in the ε (eps) neighborhood of every point
        min_samples: the parameter of DBSCAN, identify the core points with more than min_samples neighbors
    """
    ins_tag={}  # uniform case and delete tags with special characters
    for task in tqdm(loop_list): 
        # use gpt-4 to generate tags and store {'instruction': instruction, 'tag': tag list} in a list
        pw_file = f'tag_pkl/{task}_tag.pkl'
        with open(pw_file, mode='rb') as file:
            pw_info = pickle.load(file)
        norm_ins_tag_ls = []
        for i in pw_info[task]:
            instruction = i['instruction']
            api_output = i['api_output']
            norm_tag_ls = []
            for a in api_output:
                info_ls = []
                for v in a.values():
                    info_ls.append(v)
                tag = info_ls[0].lower()    # uniform case
                tag = re.sub(r"[^A-Za-z]+", " ", tag)   # remove special symbols
                tag = norm_phrase(tag)         
                norm_tag_ls.append(tag.strip())
            norm_ins_tag_ls.append({'instruction': instruction, 'tag_list': norm_tag_ls})         
        ins_tag[task] = norm_ins_tag_ls

    # tag_ls stores the information of tags in the current task and previous tasks  
    tag_ls = []
    for i in range(cur_step+1):
        for item in ins_tag[loop_list[i]]:
            tag_ls.extend(item['tag_list'])
    c = Counter(tag_ls)
    tag_pool = list(c)  # get tag pool
    phrase_embs = model.encode(tag_pool)   # get embeddings of current tag pool

    # DBSCAN clustering to remove duplicates
    clustering = DBSCAN(eps = eps, min_samples = min_samples, metric = 'cosine').fit(phrase_embs)
    # get the information of each tag
    tag_info = []
    for idx in range(len(phrase_embs)):
        tag_info.append({"original": tag_pool[idx], "label": clustering.labels_[idx], "count": c[tag_pool[idx]]})
    # deduplicate tags based on clustering results
    for label in range(max(clustering.labels_)):
        label_tag_info = [i for i in tag_info if i['label'] == label]   # tags with the corresponding clustering label
        ins_tag = tag_dedup(loop_list, label_tag_info, ins_tag)   # iteratively modify ins_tag

    total_num = 0       # statistics of the total amount of instructions
    dedup_tag_ls = []   # deduplicated tag list
    for i in range(cur_step+1):
        total_num += len(ins_tag[loop_list[i]])
        for item in ins_tag[loop_list[i]]:
            dedup_tag_ls.extend(item['tag_list'])
    dedup_c = Counter(dedup_tag_ls)

    # use idf to generate the InsInfo score corresponding to each instruction for previous tasks.
    ins_tag_dic = {}
    for i in range(cur_step):
        task = loop_list[i]
        sub_ins_tag = []
        for item in ins_tag[task]:
            new_tag_ls = []
            score = 0
            for tag in item['tag_list']:
                # new_tag_ls.append({'tag': tag, 'count': dedup_c[tag]})
                score += math.log(total_num/dedup_c[tag])      # idf
            # sub_ins_tag.append({'instruction': item['instruction'], 'tag_list': new_tag_ls, 'score': score})
            sub_ins_tag.append({'instruction': item['instruction'], 'score': score})

        sub_ins_tag = sorted(sub_ins_tag, key = lambda x: x['score'], reverse=True)
        ins_ls = [i['instruction'] for i in sub_ins_tag]
        score_ls = [i['score'] for i in sub_ins_tag]
        norm_score_ls = normalization(score_ls)
        ins_tag_dic[task] = {'ins_list': ins_ls, 'score_list': norm_score_ls}
    return ins_tag_dic
