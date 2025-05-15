# -*- coding: utf-8 -*-
# @Time    : 4/10/23 5:05 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : eval_llm_cla.py

# evaluation classification based on gpt/bert embeddings
import os
import os.path

import openai
import numpy as np
import math
import json
import string
import torch
import numpy as np
from collections import OrderedDict
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report
from stats import calculate_stats
import argparse
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='+', required=True,
                        help='List of space-separated json file names')
parser.add_argument('--vs_class_labels', type=str, default='./labels/class_labels_indices_vs.csv')
parser.add_argument('--text_embed_setting', type=str, default='gpt', choices=['gpt', 'bert'])
parser.add_argument('--max_openai_batch_size', type=int, default=2000)
parser.add_argument('--num_os_processes', type=int, default=40)
args = parser.parse_args()

eval_file_list = args.files

dataset = 'vs'
llm_task = 'caption'
text_embed_setting = args.text_embed_setting

for x in eval_file_list:
    assert os.path.exists(x) == True

num_class = 6
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_mdl_size = 'bert-large-uncased'
bert_tokenizer = ""
bert_model = ""

for eval_file in eval_file_list:

    eval_file_folder = os.path.dirname(os.path.abspath(eval_file))

    def get_bert_embedding(input_text):
        input_text = remove_punctuation_and_lowercase(input_text)
        #print(input_text)
        inputs = bert_tokenizer(input_text, return_tensors="pt")
        if inputs['input_ids'].shape[1] > 512:
            inputs['input_ids'] = inputs['input_ids'][:, :512]
            inputs['token_type_ids'] = inputs['token_type_ids'][:, :512]
            inputs['attention_mask'] = inputs['attention_mask'][:, :512]
            #print('trim the length')
            #print(inputs['input_ids'].shape)
        outputs = bert_model(**inputs.to(device))
        last_hidden_states = torch.mean(outputs.last_hidden_state[0], dim=0).cpu().detach().numpy()
        return last_hidden_states

    def get_gpt_embedding(input_text_list, openai_model='text-embedding-ada-002'):
        # TODO: change to your openai key
        client = openai.OpenAI()
        response = client.embeddings.create(
            input=input_text_list,
            model=openai_model,
            encoding_format='float'
        )
        # embeddings = response['data'][0]['embedding']
        embeddings = [response.data[i].embedding for i in range(len(response.data))]
        return embeddings

    def cosine_similarity(vector1, vector2):
        # dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        # magnitude1 = math.sqrt(sum(v1 ** 2 for v1 in vector1))
        # magnitude2 = math.sqrt(sum(v2 ** 2 for v2 in vector2))
        # return dot_product / (magnitude1 * magnitude2)
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def remove_punctuation_and_lowercase(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text

    label_list = np.loadtxt(args.vs_class_labels, delimiter=',', dtype=str, skiprows=1)

    label_embed_cache_file = '{:s}/label_embed_dict/{:s}_{:s}.json'.format(eval_file_folder, dataset, text_embed_setting)
    if os.path.exists(label_embed_cache_file):
        with open(label_embed_cache_file, 'r') as f:
            json_str = f.read()
        label_dict = json.loads(json_str, object_pairs_hook=OrderedDict)
    else:
        label_dict = OrderedDict()
        if text_embed_setting == 'gpt':
            gpt_embeddings_list = get_gpt_embedding(['sound of ' + class_name[1:-1].replace('_', ' ').lower() for class_name in label_list[:, 2]])
        for i in range(label_list.shape[0]):
            class_code = label_list[i, 1]
            class_name = label_list[i, 2][1:-1]
            if text_embed_setting == 'gpt':
                # label_dict[class_name] = get_gpt_embedding('sound of ' + class_name.replace('_', ' ').lower())
                label_dict[class_name] = gpt_embeddings_list[i]
            elif text_embed_setting == 'bert':
                label_dict[class_name] = get_bert_embedding('sound of ' + class_name.replace('_', ' ').lower())

        os.makedirs(os.path.dirname(label_embed_cache_file), exist_ok=True)
        with open(label_embed_cache_file, 'w') as f:
            json_str = json.dumps(label_dict)
            f.write(json_str)

    with open(eval_file, 'r') as fp:
        eval_data = json.load(fp)

    save_cache_path = '{:s}/embedding_cache/{:s}_{:s}_{:s}.json'.format(eval_file_folder, dataset, llm_task, text_embed_setting)
    if os.path.exists(save_cache_path) == True:
        with open(save_cache_path, 'r') as f:
            embed_cache = f.read()
        embed_cache = json.loads(embed_cache)
    else:
        embed_cache = {}
        if text_embed_setting == 'gpt':
            print('Embedding cache not found, creating new cache')
            if llm_task == 'cla':
                list_of_preds_for_cache = [x['pred'].split(':')[-1].split('.')[0][1:].split(';') for x in eval_data]
                list_of_preds_for_cache = ['sound of ' + x.lower().lstrip() for x in list_of_preds_for_cache]
            elif llm_task == 'caption':
                list_of_preds_for_cache = [x['pred'].split(':')[-1][1:] for x in eval_data]
                list_of_preds_for_cache = ['sound of ' + x.lower() for x in list_of_preds_for_cache]
            total_num_preds = len(list_of_preds_for_cache)
            
            list_of_embeds_for_cache = []
            samples_done = 0
            while samples_done < total_num_preds:
                do_until = min(samples_done + args.max_openai_batch_size, total_num_preds)
                list_of_embeds_for_cache += get_gpt_embedding(list_of_preds_for_cache[samples_done:do_until])
                samples_done = do_until
                print(f'\rDone getting {samples_done}/{total_num_preds} embeddings..', end='')
            print()
            for i in range(len(list_of_preds_for_cache)):
                embed_cache[list_of_preds_for_cache[i]] = list_of_embeds_for_cache[i]
            embed_cache_json = json.dumps(embed_cache)

            os.makedirs(os.path.dirname(save_cache_path), exist_ok=True)
            with open(save_cache_path, 'w') as f:
                f.write(embed_cache_json)
            print('Embedding cache created')

    def get_pred(cur_pred_list, label_dict, mode='max'):
        # at beginning, all zero scores
        score = np.zeros(num_class)
        label_embed_list = list(label_dict.values())
        # pred might not be a single text
        for cur_pred in cur_pred_list:
            if cur_pred in embed_cache:
                cur_pred_embed = embed_cache[cur_pred]
            else:
                if text_embed_setting == 'gpt':
                    # cur_pred_embed = get_gpt_embedding(cur_pred)
                    raise Exception('Should have cached gpt embedding')
                else:
                    cur_pred_embed = get_bert_embedding(cur_pred)
                embed_cache[cur_pred] = cur_pred_embed
            for i in range(num_class):
                if mode == 'accu':
                    score[i] = score[i] + cosine_similarity(cur_pred_embed, label_embed_list[i])
                elif mode == 'max':
                    score[i] = max(score[i], cosine_similarity(cur_pred_embed, label_embed_list[i]))
        cur_pred = np.argmax(score)
        return cur_pred

    num_sample = len(eval_data)
    print('number of samples {:d}'.format(num_sample))
    all_pred = np.zeros([num_sample, num_class])
    all_truth = np.zeros([num_sample, num_class])
    current_batch_size = 0
    current_batch_input = []
    for i in range(num_sample):
        cur_audio_id = eval_data[i]['audio_id']
        if llm_task == 'cla':
            cur_pred_list = eval_data[i]['pred'].split(':')[-1].split('.')[0][1:].split(';')
            cur_pred_list = ['sound of ' + x.lower().lstrip() for x in cur_pred_list]
        elif llm_task == 'caption':
            cur_pred_list = eval_data[i]['pred'].split(':')[-1][1:]
            cur_pred_list = ['sound of ' + cur_pred_list.lower()]

        cur_truth = eval_data[i]['ref'].split(':')[-1]

        cur_truth_idx = list(label_dict.keys()).index(cur_truth)
        all_truth[i, cur_truth_idx] = 1.0

        current_batch_input.append(cur_pred_list)
        current_batch_size += 1

        if current_batch_size == args.num_os_processes or i == num_sample-1:
            def get_pred_for_one_sample(x):
                return get_pred(x, label_dict)
            pool = mp.Pool(args.num_os_processes)
            all_pred_batch_output = pool.map(get_pred_for_one_sample, current_batch_input)
            pool.close()
            for j in range(len(all_pred_batch_output)):
                all_pred[i + 1 - current_batch_size + j, all_pred_batch_output[j]] = 1.0
            current_batch_input = []
            current_batch_size = 0

        # cur_pred_idx = get_pred(cur_pred_list, label_dict)

        # all_pred[i, cur_pred_idx] = 1.0
        print(f'\rDone with {i+1}/{num_sample} samples..', end='')
    print()

    save_fold = "{:s}/{:s}_{:s}_{:s}_cla_report".format(eval_file_folder, '.'.join(eval_file.split('/')[-1].split('.')[:-1]), llm_task, text_embed_setting)
    if os.path.exists(save_fold) == False:
        os.makedirs(save_fold)

    np.save(save_fold + '/all_pred.npy', all_pred)
    np.save(save_fold + '/all_truth.npy', all_truth)
    stats = calculate_stats(all_pred, all_truth)

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    acc = stats[0]['acc']

    np.savetxt(save_fold + '/result_summary.csv', [mAP, mAUC, acc], delimiter=',')

    sk_acc = accuracy_score(all_truth, all_pred)
    print('vocal sound accuracy: ', acc, sk_acc)

    report = classification_report(all_truth, all_pred, target_names=list(label_dict.keys()))
    with open(save_fold + "/cla_summary.txt", "w") as f:
        f.write(report)

