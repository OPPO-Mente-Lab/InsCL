# InsCL: Instruction-based Continual Learning
This repository contains code for the paper "[InsCL: A Data-efficient Continual Learning Paradigm for Fine-tuning Large Language Models with instructions](https://openreview.net/forum?id=c2YfFbNHax3)".
We propose a novel paradigm called Instruction-based Continual Learning (InsCL). 
InsCL dynamically replays previous data based on task similarity, calculated by Wasserstein Distance with instructions. 
And we further introduce an Instruction Information Metric (InsInfo) to quantify the complexity and diversity of instructions. According to InsInfo, InsCL guides the replay process more inclined to high-quality data.

## Installation

To install the experiment, please install the pip file. We chiefly just need pytorch and transformers package from huggingface. It might be a good idea to create a conda environment.
```bash
pip install -r requirements.txt
```

## Create training data with InsCL

We obtain 16 categories by integrating English tasks in [SuperNI](https://arxiv.org/abs/2204.07705) dataset (loaded from https://github.com/allenai/natural-instructions), and conduct further experiments based on 16 reallocated tasks.
The details of task composition are shown in the Appendix A.2 of [InsCL](https://openreview.net/forum?id=c2YfFbNHax3).
We randomly hold out 20% instances on each task to test the LLM on different training stages, and store the train and test set in the _'dataset'_ folder.

### Obtain tags

High-performing open-source LLMs demonstrate the ability to annotate queries with tag entities, and the precision and consistency are proven through manual annotation. Consequently, we employ [GPT-4](https://arxiv.org/abs/2303.08774) (OpenAI, 2023) as an intention tagger and clean the raw tags, representing instructions at a fine-grained entity level.
```bash
cd replay
python get_api_output.py
```
You can define your own function to call the api and store output tags.

### InsCL replay

After annotating the instructions, run the following code to sample replay data in the _'replay/replay_data'_ folder.

```bash
python InsCL_sampling.py --emb_file_path encoded_ins_dist.pkl --style curriculum_pWdist 
```
here we load prepared file that stores original instructions, embeddings and distributions from `emb_file_path`.
Please generate corresponding instruction embeddings and modify the file path as needed.
And `style` control the training order of tasks and the calculation method of Wasserstein Distance.
When the real distribution of instructions can not be obtained, remove the '__pWdist_' at the end of style.

To merge the replay data with source training data, you can run:
```bash
bash merge_data.sh
```

## Train and Evaluate

We call the training function in a sequential loop in the script to simulate incremental learning of staged fine-tuning. 
Here we define `data_dir` as _'dataset/train'_ and `root_model` as _'llama_7B'_ (loaded from https://huggingface.co/baffo32/decapoda-research-llama-7B-hf).
You can modify the dataset path and model path as needed.
```bash
cd ..
bash run_train.sh
```

Run the script to evaluate model with [Rouge-L](https://aclanthology.org/W04-1013/).
```bash
bash run_evaluate.sh
```

## Acknowledgements

This repo relies on the [POT](https://pythonot.github.io/) packages for calculating Wasserstein Distance. We are grateful to the authors and maintainers of the project.
