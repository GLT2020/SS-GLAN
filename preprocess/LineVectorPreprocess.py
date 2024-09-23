import json
import os
import re
from os.path import join

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

def read_contract(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()

    original_lines = source_code.split('\n')
    return original_lines



if __name__ == '__main__':
    ROOT = '../data/ge-sc-data/source_code'
    # bug_type = {'access_control': 57*2, 'arithmetic': 60*2, 'denial_of_service': 46*2,
    #           'front_running': 44*2, 'reentrancy': 71*2, 'time_manipulation': 50*2,
    #           'unchecked_low_level_calls': 95*2}

    # TODO: change type!
    bug_type = {'unchecked_low_level_calls': 95*2}

    for bug, counter in bug_type.items():
        source = f'{ROOT}/{bug}/all_clean'
        output = f'{ROOT}/{bug}/LineVector_codeBert.json'
        pre_model_path = '../PretrainModel/CodeBert'

        tokenizer = AutoTokenizer.from_pretrained(pre_model_path)
        model = AutoModel.from_pretrained(pre_model_path)
        smart_contracts = [join(source, f) for f in os.listdir(source) if f.endswith('.sol')]

        files_vector_dict = {}

        for files in tqdm(smart_contracts):
            file_name = files.split('/')[-1]
            files_vector_dict[file_name] = []
            file_content = read_contract(files)

            print(file_name)

            for content_item in file_content:
                # print(content_item)
                nl_tokens = tokenizer.tokenize(content_item)
                tokens = [tokenizer.cls_token] + nl_tokens  + [tokenizer.eos_token]
                # Convert tokens to ids
                tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

                context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0].mean(1).flatten()
                context_embeddings = context_embeddings.detach().numpy().tolist()

                # print(context_embeddings)
                files_vector_dict[file_name].append(context_embeddings)

        # 将文件行向量字典写入文件
        bug_name = list(bug_type.keys())[0]
        write_path = f'../data/ge-sc-data/source_code/{bug_name}/LineVectorList.json'
        print(f'写入路径:{write_path}')
        with open(write_path, 'w') as f:
            f.write(json.dumps(files_vector_dict, ensure_ascii=False))
            print("写入完成...")
