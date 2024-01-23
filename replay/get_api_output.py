'''
Copyright 2024 OPPO. All rights reserved.

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
'''

from tqdm import tqdm

def get_prompt(query):
    """
    generate the prompt to call the api according to the template
    """
    prompt = 'You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. Below is an instruction:\n[begin]\n' + query + '\n[end]\nPlease provide coarse-grained tags, such as "Spelling and Grammar Check" and "Cosplay", to identify main intentions of above instruction. Your answer should be a list including titles of tags and a brief explanation of each tag. Your response have to strictly follow this JSON format: [{"tag": str, "explanation": str}]. Please response in English.'
    return prompt

for i in range(len(loop_list)):
    tag_dic = {}
    item_ls = []
    query_ls = task_instructions    # replace with instructions for current task
    print('task:', loop_list[i])
    for q in tqdm(query_ls):
        prompt = get_prompt(q)
        output = generate(prompt)  # please replace the function calling the api when using it
        json_output = json.loads(output)
        item = {'instruction': q, 'api_output': json_output, 'tag_num': len(json_output)}
        item_ls.append(item)
    tag_dic[loop_list[i]] = item_ls

    tag_file = f'tag_pkl/{loop_list[i]}_tag.pkl'
    with open(tag_file, mode='wb') as f:
        pickle.dump(tag_dic, f)