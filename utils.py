import os
import io
import json
import tqdm
import torch
from rouge_metric import PyRouge

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding="utf-8")
    return f

def jdump(obj, f, mode="w", indent=4, default=str):
    """
    Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    name = f 
    f = _make_r_io_base(f, mode)
    jdict = [json.loads(l) for l in f] if name.endswith("jsonl") else json.load(f)
    jdict_new = []
    print("replace source target with instrcution and output ...")
    for i in tqdm.tqdm(range(len(jdict))):
        if "source" in jdict[i] and 'target' in jdict[i]:
            jdict[i]['instruction'] = jdict[i].pop('source') if 'source' in jdict[i] else jdict[i]['instruction']
            jdict[i]['input'] = ''
            jdict[i]['output'] = jdict[i].pop('target') if 'target' in jdict[i] else jdict[i]['output']
        if "input" in jdict[i] and jdict[i]['input'].strip():
            item = {}
            item["instruction"] = jdict[i]['instruction'][0] + "\n\n" + jdict[i]['input']
            item["input"] = ""
            if 'test' in name:
                item["output"] = jdict[i]["output"]
                jdict_new.append(item)
            else:
                for g_truth in jdict[i]['output']:
                    item["output"] = g_truth
                    jdict_new.append(item)
    f.close()
    return jdict_new

def pad_sequence(sequences,  padding_left=False, padding_value=0):
    bs = len(sequences)
    max_len = max([len(e) for e in sequences])
    output_tensor = torch.ones(bs, max_len) * int(padding_value)
    for i, s in enumerate(sequences):
        l = len(s)
        if padding_left:
            output_tensor[i][-l:] = s
        else:
            output_tensor[i][:l] = s 
    return output_tensor.long()

def compute_rouge_metric(preds, labels):
    rouge = PyRouge(rouge_n=(1, 2), rouge_l=True)
    assert len(preds) == len(labels), "preds lengths, labels length: {}, {}".format(len(preds), len(labels))
    rouge_1, rouge_2, rouge_l = 0, 0, 0
    total = 0
    empty_cnt = 0
    scores = rouge.evaluate(preds, labels)
    return scores

def compute_metric(pred_file, refs_file):
    with open(pred_file, 'r', encoding="utf-8") as f:
        pred_datas = [e["target"] for e in json.load(f)]
    refs_datas = [e["output"] for e in jload(refs_file)]
    result = compute_rouge_metric(pred_datas, refs_datas)
    return result
