import pandas as pd
import json
from joblib import dump, load
from operator import itemgetter
from os import environ
from tqdm import tqdm
from transformers import T5Tokenizer, T5Model, T5EncoderModel
import torch
import re
import random

RESIDUE_3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                  'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                  'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                  'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                  'UNK': 'X'}
RESIDUE_1_to_3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                  'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                  'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                  'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
                  'X': 'UNK'}      # UNK for unknown amino acid
RESIDUE_1 = 'ARNDCQEGHILKMFPSTWYVX'  # X for unknown amino acid

def add_space(s):
    s1 = ''
    for i in s:
        s1 += (i + ' ')
    return s1[:-1]

def check_mutation(file_mapping, mut_chain, mut_pos, mut_res):
    is_correct = 0
    with open(file_mapping, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            cur_res3 = line[0:3]
            cur_chain = line[4]
            cur_pos = int(line[5:9])
            if cur_pos == mut_pos and cur_chain == mut_chain:
                if RESIDUE_3_TO_1[cur_res3] != mut_res:
                    next_line = lines[i + 1]
                    next_pos = int(next_line[5:9])
                    if next_line[9] != ' ' and next_pos == mut_pos:  # Some sites have lettered designations: 60, 60A, 60B.....
                        continue
                    else:
                        is_correct = 0
                        break
                else:
                    is_correct = 1
                    break
    return is_correct

def get_chains_information(pdbNum, file_mapping, chainNames1, chainNames2, mut_chain, mutPoints):
    chains1, chains2 = {}, {}
    chains_mut = {}
    seq_mut_chains = ''
    mut_res = [mutPoints[0], mutPoints[-1]]
    mut_pos = int(mutPoints[1:-1])
    abs_mut_pos = None
    usefulChainNames = chainNames1 + chainNames2
    for chainName in chainNames1:
        chains1[chainName] = ''
    for chainName in chainNames2:
        chains2[chainName] = ''
    with open(file_mapping, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            cur_res3 = line[0:3]
            cur_chain = line[4]
            res = RESIDUE_3_TO_1[cur_res3] if cur_res3 in RESIDUE_3_TO_1 else 'X'
            if line[-6] == ' ' and line[-4] != ' ':   # num_residue >= 1000
                num = line[-5:-1]
            else:   # num_residue < 1000
                num = line[-4:-1]
            if cur_chain in usefulChainNames:
                num = int(num)
                if cur_chain in chainNames1:
                    chains1[cur_chain] += res
                    if num != len(chains1[cur_chain]):
                        print('error at insert residue to chains')
                if cur_chain in chainNames2:
                    chains2[cur_chain] += res
                    if num != len(chains2[cur_chain]):
                        print('error at insert residue to chains')
            if cur_chain == mut_chain:
                if mut_pos == int(line[5:9]) and res == mut_res[0]:
                    seq_mut_chains += mut_res[-1]
                    abs_mut_pos = num
                else:
                    seq_mut_chains += res
        # Processing multi-chain
        if pdbNum in ['1DAN', '2VIR', '3BN9', '3HI1']:
            if len(chainNames1) > 1:
                chains1[chainNames1] = chains1[chainNames1[0]] + '/' + chains1[chainNames1[1]]
                if mut_chain in chainNames1:
                    chainmutNames = chainNames1 + '_' + mut_chain + mutPoints
                    if mut_chain == chainNames1[0]:
                        chains_mut[chainmutNames] = seq_mut_chains + '/' + chains1[chainNames1[1]]
                    else:
                        chains_mut[chainmutNames] = chains1[chainNames1[0]] + '/' + seq_mut_chains
                chains1.pop(chainNames1[0])
                chains1.pop(chainNames1[1])
            if len(chainNames2) > 1:
                chains2[chainNames2] = chains2[chainNames2[0]] + '/' + chains2[chainNames2[1]]
                if mut_chain in chainNames2:
                    chainmutNames = chainNames2 + '_' + mut_chain + mutPoints
                    if mut_chain == chainNames2[0]:
                        chains_mut[chainmutNames] = seq_mut_chains + '/' + chains2[chainNames2[1]]
                    else:
                        chains_mut[chainmutNames] = chains2[chainNames2[0]] + '/' + seq_mut_chains
                chains2.pop(chainNames2[0])
                chains2.pop(chainNames2[1])
        elif pdbNum in ['4JRA', '5XWT']:
            chains1.pop(chainNames1[1])
            chains2.pop(chainNames2[1])
            chainNames1, chainNames2 = chainNames1[0], chainNames2[0]
        else:
            if len(chainNames1) > 1:
                if mut_chain in chainNames1:
                    chains1 = {mut_chain: chains1[mut_chain]}
                    chainNames1 = mut_chain
                else:
                    chains1 = {chainNames1[0]: chains1[chainNames1[0]]}
                    chainNames1 = chainNames1[0]
            if len(chainNames2) > 1:
                if mut_chain in chainNames2:
                    chains2 = {mut_chain: chains2[mut_chain]}
                    chainNames2 = mut_chain
                else:
                    chains2 = {chainNames2[0]: chains2[chainNames2[0]]}
                    chainNames2 = chainNames2[0]
        if len(chains_mut) == 0:
            if mut_chain == chainNames1:
                chainmutNames = chainNames1 + '_' + mut_chain + mutPoints
            elif mut_chain == chainNames2:
                chainmutNames = chainNames2 + '_' + mut_chain + mutPoints
            chains_mut[chainmutNames] = seq_mut_chains
        # Cut off length
        if pdbNum == '1DE4':
            if abs_mut_pos < 400:
                chains2[chainNames2 + '1'] = chains2[chainNames2][:500]
                chains2.pop(chainNames2)
                chainNames2 = chainNames2 + '1'
                chains_mut[chainNames2 + '_' + mut_chain + mutPoints] = chains_mut[chainmutNames][:500]
                chains_mut.pop(chainmutNames)
                chainmutNames = chainNames2 + '_' + mut_chain + mutPoints
            else:
                chains2[chainNames2 + '2'] = chains2[chainNames2][-500:]
                chains2.pop(chainNames2)
                chainNames2 = chainNames2 + '2'
                chains_mut[chainNames2 + '_' + mut_chain + mutPoints] = chains_mut[chainmutNames][-500:]
                chains_mut.pop(chainmutNames)
                chainmutNames = chainNames2 + '_' + mut_chain + mutPoints
        elif pdbNum in ['1SUV', 'A', 'T']:
            if len(chains1[chainNames1]) > 500:
                chains1[chainNames1] = chains1[chainNames1][-500:]
            if len(chains2[chainNames2]) > 500:
                chains2[chainNames2] = chains2[chainNames2][-500:]
            if len(chains_mut[chainmutNames]) > 500:
                chains_mut[chainmutNames] = chains_mut[chainmutNames][-500:]
        elif pdbNum == 'U':
            chains1[chainNames1] = chains1[chainNames1][1000:1500]
            chains_mut[chainmutNames] = chains_mut[chainmutNames][1000:1500]
        elif pdbNum == 'Z':
            chains2[chainNames2] = chains2[chainNames2][500:1000]
            chains_mut[chainmutNames] = chains_mut[chainmutNames][500:1000]

    if mut_chain in chainNames2:
        return [pdbNum+'_'+chainNames1, pdbNum+'_'+chainNames2], [chains1, chains2],\
               [pdbNum+'_'+chainNames1, pdbNum+'_'+chainmutNames], [chains1, chains_mut]
    elif mut_chain in chainNames1:
        return [pdbNum+'_'+chainNames2, pdbNum+'_'+chainNames1], [chains2, chains1], \
               [pdbNum+'_'+chainNames2, pdbNum+'_'+chainmutNames], [chains2, chains_mut]
    else:
        print('error')


def generateSequenceData(csvPath, MappingPath, savePath):
    with open(csvPath, 'rb') as f:
        dataAll = pd.read_excel(f)
        total_num = len(dataAll)
        SequenceData = []
        for idx in tqdm(range(total_num)):
            data = dataAll.iloc[idx]
            ddg = data[-1]
            pdbNum = data[2][:4]
            chainNames1, chainNames2 = data[10].replace(', ', ''), data[11].replace(', ', '')
            if 'Uniprot' in data[12]:
                data[12] = data[12].replace(' (Based on Uniprot)', '')
            if len(pdbNum) < 4:  # for Uniprot
                chainNames1, chainNames2 = 'A', 'B'
                mut_chain = data[12][:6]
                pdb_mut = data[12].replace(': ', '')[6:]
                mut_chain = 'A' if mut_chain == data[3] else 'B'
                seq_mut = data[13]
                if pdb_mut != seq_mut:  # for Uniprot, when pdb_mut_pos != seq_mut_pos, use seq_mut_pos
                    pdb_mut = seq_mut
            else:
                mut_chain = data[12][0]
                pdb_mut = data[12].replace(': ', '')[1:]
            mut_res = pdb_mut[0]
            mut_pos = int(pdb_mut[1:-1])
            file_mapping = MappingPath + '/' + pdbNum + '.mapping'

            mut_correct = check_mutation(file_mapping, mut_chain, mut_pos, mut_res)
            if mut_correct == 0:
                print(data[0])
                continue
            nameOfParts, Chains, nameOfParts_mut, Chains_mut = get_chains_information(pdbNum, file_mapping, chainNames1, chainNames2, mut_chain, pdb_mut)
            SequenceData.append([pdbNum, nameOfParts, Chains, nameOfParts_mut, Chains_mut, ddg])
        dump(SequenceData, savePath)


def generate_embedding_dict(SequenceData_path, embedding_dict_path):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    embedding_dict = {}  # store labels and the corresponding hidden layer representations
    chain_sequence_dict = {}  # store labels and the corresponding sequences
    SequenceData = load(SequenceData_path)
    for data in SequenceData:
        for part in data[2]:
            for key in part.keys():
                if (data[0] + '_' + key) not in chain_sequence_dict:
                    chain_sequence_dict[data[0] + '_' + key] = part[key]
        for part in data[4]:
            for key in part.keys():
                if (data[0] + '_' + key) not in chain_sequence_dict:
                    chain_sequence_dict[data[0] + '_' + key] = part[key]
    tokenizer = T5Tokenizer.from_pretrained('./protein_llm', do_lower_case=False)
    print('Loading pre-trained model.')
    model = T5Model.from_pretrained('./protein_llm').to(device)
    print('Finish loading.')
    for chain in tqdm(chain_sequence_dict.keys()):
        sequence = add_space(chain_sequence_dict[chain])
        if '/' in sequence:
            sequence = sequence.replace('/', '<extra_id_1>')  # for multi-chains, add token 'SEP', i.e. <extra_id_1>
        sequence = ['<extra_id_0> ' + re.sub(r'[UZOB]', 'X', sequence)]  # add token 'CLS', i.e. <extra_id_0>
        ids = tokenizer.batch_encode_plus(sequence, max_length=500, add_special_tokens=True, padding='max_length', truncation=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding_dict[chain] = torch.squeeze(embedding.last_hidden_state).cpu()
    dump(embedding_dict, embedding_dict_path)


def generate_sorted_10_fold(SequenceData_path, root_path, postfix=''):
    SequenceData = load(SequenceData_path)
    chain_DDG_list = []
    for i in range(len(SequenceData)):
        cur_pair = SequenceData[i]
        chainA, chainB = cur_pair[1][0], cur_pair[1][1]
        chainBmut = cur_pair[3][1]
        ddg = float(cur_pair[-1])
        chain_DDG_list.append([chainA, chainB, chainBmut, ddg])
    chain_DDG_list_folds = []
    chain_DDG_list_train, chain_DDG_list_test = [], []
    sorted_chain_DDG_list = sorted(chain_DDG_list, key=lambda x: (x[-1]))
    for fold_i in range(10):
        chain_DDG_list_folds.append(itemgetter(*list(range(fold_i, len(chain_DDG_list), 10)))(sorted_chain_DDG_list))
    for fold_i in range(10):
        chain_DDG_list_test.append(chain_DDG_list_folds[fold_i])
        chain_DDG_list_train.append([])
        for i in range(0, fold_i):
            chain_DDG_list_train[fold_i] += chain_DDG_list_folds[i]
        for i in range(fold_i + 1, 10):
            chain_DDG_list_train[fold_i] += chain_DDG_list_folds[i]
    dump(chain_DDG_list_train, root_path + 'chain_DDG_list_train' + postfix)
    dump(chain_DDG_list_test, root_path + 'chain_DDG_list_test' + postfix)

def generate_random_10_fold(SequenceData_path, root_path, postfix=''):
    SequenceData = load(SequenceData_path)
    chain_DDG_list = []
    for i in range(len(SequenceData)):
        cur_pair = SequenceData[i]
        chainA, chainB = cur_pair[1][0], cur_pair[1][1]
        chainBmut = cur_pair[3][1]
        ddg = float(cur_pair[-1])
        chain_DDG_list.append([chainA, chainB, chainBmut, ddg])
    chain_DDG_list_folds = []
    chain_DDG_list_train, chain_DDG_list_test = [], []
    random_chain_DDG_list = [chain_DDG_list[i] for i in random.sample(range(len(chain_DDG_list)), len(chain_DDG_list))]
    for fold_i in range(10):
        chain_DDG_list_folds.append(itemgetter(*list(range(fold_i, len(chain_DDG_list), 10)))(random_chain_DDG_list))
    for fold_i in range(10):
        chain_DDG_list_test.append(chain_DDG_list_folds[fold_i])
        chain_DDG_list_train.append([])
        for i in range(0, fold_i):
            chain_DDG_list_train[fold_i] += chain_DDG_list_folds[i]
        for i in range(fold_i + 1, 10):
            chain_DDG_list_train[fold_i] += chain_DDG_list_folds[i]
    dump(chain_DDG_list_train, root_path + 'chain_DDG_list_train' + postfix)
    dump(chain_DDG_list_test, root_path + 'chain_DDG_list_test' + postfix)

def get_tensor_DDG_list(chain_DDG_path, embedding_path, edge_path, lengh_path, dump_path):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    edge_dict = json.load(open(edge_path, 'r'))
    length_dict = json.load(open(lengh_path, 'r'))
    embedding_dict = load(embedding_path,)
    total_chain_ddg = load(chain_DDG_path)
    for fold_i in range(10):
        print(f'Processing fold_{fold_i} dataset.')
        chain_ddg = total_chain_ddg[fold_i]
        result = []
        for i in range(len(chain_ddg)):
            chain_r = []
            for j in range(3):
                embedding = embedding_dict[chain_ddg[i][j]]
                chain_r.append(embedding)
            edge_index = torch.tensor(edge_dict[chain_ddg[i][j]])
            length = torch.tensor(length_dict[chain_ddg[i][j]])
            ddg = torch.tensor(float(chain_ddg[i][3]), dtype=torch.float32)
            a = [torch.stack(chain_r, dim=0).to(device), length.to(device), edge_index.to(device), ddg.to(device)]
            result.append(a)
        dump(result, dump_path+'_'+str(fold_i))

def generate_chain_length(SequenceData_path):
    dataset_name = SequenceData_path.split('/')[-1].split('_')[-1]
    length_dict = {}
    SequenceData = load(SequenceData_path)
    for data in SequenceData:
        for part in data[2]:
            for key in part.keys():
                if (data[0] + '_' + key) not in length_dict:
                    length_dict[data[0] + '_' + key] = len(part[key])
        for part in data[4]:
            for key in part.keys():
                if (data[0] + '_' + key) not in length_dict:
                    length_dict[data[0] + '_' + key] = len(part[key])
    with open(f'../data/length_{dataset_name}.json', 'w') as f:
        json.dump(length_dict, f)

if __name__ == '__main__':
    environ['CUDA_VISIBLE_DEVICES'] = '0'   # select gpu
    # generateSequenceData('../data/MPAD_Clean.xlsx', '../data/mapping/mpad', '../data/MPAD/SequenceData_mpad')
    # generate_embedding_dict('../data/MPAD/SequenceData_mpad', f'../data/embedding_dict_mpda')
    # generate_chain_length('../data/MPAD/SequenceData_mpad')

    ## randomly generated
    # generate_random_10_fold('../data/MPAD/SequenceData_mpad', '../data/MPAD/', '_rand')
    # get_tensor_DDG_list('../data/MPAD/chain_DDG_list_train_rand', f'../data/embedding_dict_mpda',
    #                     '../data/MPAD/mpad_edge.json', '../data/length_mpad.json',
    #                     f'../data/MPAD/tensor_DDG_list_train')
    # get_tensor_DDG_list('../data/MPAD/chain_DDG_list_test_rand', f'../data/embedding_dict_mpda',
    #                     '../data/MPAD/mpad_edge.json', '../data/length_mpad.json',
    #                     f'../data/MPAD/tensor_DDG_list_test')


    ## processing dataset by sorting DDG
    # get_tensor_DDG_list('../data/MPAD/chain_DDG_list_train, f'../data/embedding_dict_mpda',
    #                     '../data/MPAD/mpad_edge.json', '../data/length_mpad.json',
    #                     f'../data/MPAD/tensor_DDG_list_train')
    # get_tensor_DDG_list('../data/MPAD/chain_DDG_list_test', f'../data/embedding_dict_mpda',
    #                     '../data/MPAD/mpad_edge.json', '../data/length_mpad.json',
    #                     f'../data/MPAD/tensor_DDG_list_test')



