import os
import re
import torch
import json
import pandas as pd
from model.utils import calc_std_dev
from copy import deepcopy
from joblib import dump, load
from operator import itemgetter
from tqdm import tqdm
from math import isnan, log
from transformers import T5Tokenizer, T5Model

RESIDUE_3_TO_1 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                  'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                  'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                  'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                  'UNK': 'X'}  # UNK for unknown amino acid
EXTERN_2I9B = [['2I9B',
                ['2I9B_A', '2I9B_E'],
                [{'A': 'SNELHQVPSNCDCLNGGTCVSNKYFSNIHWCNCPKKFGGQHCEIDKSKTCYEGNGHFYRGKASTDTMGRPCLPWNSATVLQQTYHAHRSDALQLGLGKHNYCRNPDNRRRPWCYVQVGLKPLVQECMVHDCADGKKPSSPPEE'},
                 {'E': 'LRCMQCKTNGDCRVEECALGQDLCRTTIVRLWEEGEELELVEKSCTHSEKTNRTLSYRTGLKITSLTEVVCGLDLCNQGNSGRAVTYSRSRYLECISCGSSDMSCERGRHQSLQCRSPEEQCLDVVTHWIQEGEEGRPKDDRHLRGCGYLPGCPGSNGFHNQDTFHFLKCCQTTKCNEGPILELENLPQNGRQCYSCKGQSTHGCSSEETFLIDCRGPMNQCLVATGTHEPKQQSYMVRGCATASMCQHAHLGDAFSMNHIDVSCCTKSGCNHPDLD'}],
                9.10E-10,
                ['2I9B_A', '2I9B_E_RE129A'],    # (R)ARG E 137 -> 129
                [{'A': 'SNELHQVPSNCDCLNGGTCVSNKYFSNIHWCNCPKKFGGQHCEIDKSKTCYEGNGHFYRGKASTDTMGRPCLPWNSATVLQQTYHAHRSDALQLGLGKHNYCRNPDNRRRPWCYVQVGLKPLVQECMVHDCADGKKPSSPPEE'},
                 {'E_RE129A': 'LRCMQCKTNGDCRVEECALGQDLCRTTIVRLWEEGEELELVEKSCTHSEKTNRTLSYRTGLKITSLTEVVCGLDLCNQGNSGRAVTYSRSRYLECISCGSSDMSCERGRHQSLQCRSPEEQCLDVVTHWIQEGEEGAPKDDRHLRGCGYLPGCPGSNGFHNQDTFHFLKCCQTTKCNEGPILELENLPQNGRQCYSCKGQSTHGCSSEETFLIDCRGPMNQCLVATGTHEPKQQSYMVRGCATASMCQHAHLGDAFSMNHIDVSCCTKSGCNHPDLD'}],
                5.60E-10,
                298],
               ['2I9B',
                ['2I9B_A', '2I9B_E'],
                [{'A': 'SNELHQVPSNCDCLNGGTCVSNKYFSNIHWCNCPKKFGGQHCEIDKSKTCYEGNGHFYRGKASTDTMGRPCLPWNSATVLQQTYHAHRSDALQLGLGKHNYCRNPDNRRRPWCYVQVGLKPLVQECMVHDCADGKKPSSPPEE'},
                 {'E': 'LRCMQCKTNGDCRVEECALGQDLCRTTIVRLWEEGEELELVEKSCTHSEKTNRTLSYRTGLKITSLTEVVCGLDLCNQGNSGRAVTYSRSRYLECISCGSSDMSCERGRHQSLQCRSPEEQCLDVVTHWIQEGEEGRPKDDRHLRGCGYLPGCPGSNGFHNQDTFHFLKCCQTTKCNEGPILELENLPQNGRQCYSCKGQSTHGCSSEETFLIDCRGPMNQCLVATGTHEPKQQSYMVRGCATASMCQHAHLGDAFSMNHIDVSCCTKSGCNHPDLD'}],
                9.10E-10,
                ['2I9B_A', '2I9B_E_KE131A'],  # (K)LYS E 139 -> 131
                [{'A': 'SNELHQVPSNCDCLNGGTCVSNKYFSNIHWCNCPKKFGGQHCEIDKSKTCYEGNGHFYRGKASTDTMGRPCLPWNSATVLQQTYHAHRSDALQLGLGKHNYCRNPDNRRRPWCYVQVGLKPLVQECMVHDCADGKKPSSPPEE'},
                 {'E_KE131A': 'LRCMQCKTNGDCRVEECALGQDLCRTTIVRLWEEGEELELVEKSCTHSEKTNRTLSYRTGLKITSLTEVVCGLDLCNQGNSGRAVTYSRSRYLECISCGSSDMSCERGRHQSLQCRSPEEQCLDVVTHWIQEGEEGRPADDRHLRGCGYLPGCPGSNGFHNQDTFHFLKCCQTTKCNEGPILELENLPQNGRQCYSCKGQSTHGCSSEETFLIDCRGPMNQCLVATGTHEPKQQSYMVRGCATASMCQHAHLGDAFSMNHIDVSCCTKSGCNHPDLD'}],
                2.84E-09,
                298]]

def add_space(s):
    s1 = ''
    for i in s:
        s1 += (i + ' ')
    return s1[:-1]

def generateProcessedCSV(ori_skempi2_csv, processed_csv):
    with open(ori_skempi2_csv, 'r', encoding='UTF-8') as f:
        dataAll = pd.read_csv(f, sep=';', dtype=str, keep_default_na=False)
        poplist = ['Mutation(s)_PDB', 'iMutation_Location(s)', 'Hold_out_type',
                   'Hold_out_proteins', 'Affinity_mut (M)', 'Affinity_wt (M)', 'Reference',
                   'Protein 1', 'Protein 2', 'kon_mut (M^(-1)s^(-1))', 'kon_mut_parsed',
                   'kon_wt (M^(-1)s^(-1))', 'kon_wt_parsed', 'koff_mut (s^(-1))',
                   'koff_mut_parsed', 'koff_wt (s^(-1))', 'koff_wt_parsed',
                   'dH_mut (kcal mol^(-1))', 'dH_wt (kcal mol^(-1))', 'dS_mut (cal mol^(-1) K^(-1))',
                   'dS_wt (cal mol^(-1) K^(-1))', 'Notes', 'Method', 'SKEMPI version']
        for column_name in poplist:
            dataAll.pop(column_name)
        temperatures = dataAll['Temperature']
        for i in range(len(temperatures)):
            temperatures[i] = temperatures[i].replace('(assumed)', '')
            # if temperatures[i]=='':
            #     temperatures[i]='298'
        dataAll.to_csv(processed_csv, index=False, sep=';')

def getChainsFromMappingFile(pdnNum, mapping_path):
    pdb, chainNames1, chainNames2 = pdnNum.split('_')
    usefulChainNames = chainNames1 + chainNames2
    chains1, chains2 = {}, {}
    for chainName in chainNames1:
        chains1[chainName] = ''
    for chainName in chainNames2:
        chains2[chainName] = ''
    with open(mapping_path + '/' + pdb + '.mapping', 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            res_3 = line[0:3]
            chainName = line[4]
            if line[-6] == ' ' and line[-4] != ' ':  # num_residue>=1000
                num = line[-5:-1]
            else:  # num_residue<1000
                num = line[-4:-1]
            if chainName in usefulChainNames:
                res = RESIDUE_3_TO_1[res_3]
                num = int(num)
                if chainName in chainNames1:
                    chains1[chainName] += res
                    if num != len(chains1[chainName]):
                        print('error at insert residue to chains')
                if chainName in chainNames2:
                    chains2[chainName] += res
                    if num != len(chains2[chainName]):
                        print('error at insert residue to chains')
        return pdb, [pdb + '_' + chainNames1, pdb + '_' + chainNames2], [chains1, chains2]


def getMutChains(nameOfParts, Chains, mutPoints: str, startResNums=None, exchangePlace=False):
    mut_Chains = deepcopy(Chains)
    newMutPoints = mutPoints.split(',')
    mutsNum = len(newMutPoints)
    newMutPoints = list(set(newMutPoints))
    if mutsNum != len(newMutPoints):
        print('repeated muts')
    for i in range(len(newMutPoints)):
        newMutPoints[i] = newMutPoints[i].replace(':', '')
        if exchangePlace:
            newMutPoints[i] = newMutPoints[i][1] + newMutPoints[i][0] + newMutPoints[i][2:]
    chainNameDict = {}
    for chainName in mut_Chains[0]:
        chainNameDict[chainName] = chainName
    for chainName in mut_Chains[1]:
        chainNameDict[chainName] = chainName
    for mutPoint in newMutPoints:
        chain = mutPoint[1]
        chainNameDict[chain] += '_' + mutPoint
    nameOfParts_mut = [nameOfParts[0], nameOfParts[1]]
    for mutPoint in newMutPoints:
        chain = mutPoint[1]
        oldRes = mutPoint[0]
        if startResNums is None:
            startResNum = 1
        else:
            startResNum = startResNums[chain]
        mutPosition = int(mutPoint[2:-1]) - startResNum
        newRes = mutPoint[-1]
        if chain in mut_Chains[0]:
            mutPart = 0
        else:
            mutPart = 1
        if mut_Chains[mutPart][chain][mutPosition] != oldRes:
            print("Error!!! The mutation does not fit the origin res.")
            raise
        mut_Chains[mutPart][chain] = mut_Chains[mutPart][chain][0:mutPosition] + newRes + mut_Chains[mutPart][chain][mutPosition + 1:]
        nameOfParts_mut[mutPart] += '_' + mutPoint
    for chainName in chainNameDict:  # change name of key to add mut positions
        newName = chainNameDict[chainName]
        if chainName != newName:
            if chainName in mut_Chains[0]:
                mutPart = 0
            else:
                mutPart = 1
            mut_Chains[mutPart][newName] = mut_Chains[mutPart][chainName]
            del mut_Chains[mutPart][chainName]
    return nameOfParts_mut, mut_Chains

def generateSequenceData(csvPath, MappingPath, savePath):
    with open(csvPath, 'r', encoding='UTF-8') as f:
        dataAll = pd.read_csv(f, sep=';')
        total_num = len(dataAll)
        SequenceData = []
        for idx in range(total_num):
            print('Preprocessing: ' + str(idx))
            data = dataAll.iloc[idx]
            # get pdb_num, names of the two non-mutated parts and their sequence representations
            pdbNum, nameOfParts, Chains = getChainsFromMappingFile(data[0], MappingPath)
            nameOfParts_mut, Chains_mut = getMutChains(nameOfParts, Chains, data[1])
            affinity = float(data[3])
            affinity_mut = float(data[2])
            temperature = float(data[4])
            SequenceData.append([pdbNum, nameOfParts, Chains, affinity, nameOfParts_mut, Chains_mut, affinity_mut, temperature])
    SequenceData += EXTERN_2I9B
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
        for part in data[5]:
            for key in part.keys():
                if (data[0] + '_' + key) not in chain_sequence_dict:
                    chain_sequence_dict[data[0] + '_' + key] = part[key]
    tokenizer = T5Tokenizer.from_pretrained('./protein_llm/', do_lower_case=False)
    print('Loading pretrained model.')
    model = T5Model.from_pretrained('./protein_llm').to(device)  # Needs about 11800M VRAM to load the protein language model
    print('Finish loading.')
    for chain in tqdm(chain_sequence_dict.keys()):
        sequence = add_space(chain_sequence_dict[chain])
        sequence = ['<extra_id_0> '+re.sub(r'[UZOB]', 'X', sequence)]  # add token 'CLS', i.e. <extra_id_0>
        ids = tokenizer.batch_encode_plus(sequence, max_length=500, add_special_tokens=True, padding='max_length', truncation=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding_dict[chain] = torch.squeeze(embedding.last_hidden_state).cpu()
    dump(embedding_dict, embedding_dict_path)

def get_raw_chain_DDG_list_skempi1131(filename):
    chain_DDG_list=[]
    f = open(filename, 'r', encoding='UTF-8')
    lines = f.readlines()
    for line in lines:
        words = line.split()
        if words[3] == '1':
            if words[2] == words[5][1]:
                chainA = words[0]+'_'+words[1]
                chainB = words[0]+'_'+words[2]
                chainBmut = words[0]+'_'+words[2]+'_'+words[5]
            else:
                chainA = words[0]+'_'+words[2]
                chainB = words[0]+'_'+words[1]
                chainBmut = words[0]+'_'+words[1]+'_'+words[5]
            ddg = float(words[4])
            chain_DDG_list.append([chainA, chainB, chainBmut, ddg])
    return chain_DDG_list

def get_raw_chain_DDG_list_skempi2398(SequenceData_path):
    SequenceData = load(SequenceData_path)
    chain_DDG_list = []
    chain_DDG_dict, chain_to_sequence = {}, {}
    for data in tqdm(SequenceData[:-2]):   # The last two instances are 2I9B added to S1131
        if len(data[2][0]) > 1 or len(data[2][1]) > 1:
            continue  # in SKEMPI_2.0, only single strand is left
        part0, part1, part1_mut = [], [], []
        for chain in data[2][0]:
            part0.append(data[0]+'_'+chain)
        for chain in data[2][1]:
            part1.append(data[0]+'_'+chain)
        mutPart = 1
        for chain in data[5][0]:
            if len(chain) > 1:
                mutPart = 0
        if mutPart == 0:
            tmp = part0
            part0 = part1
            part1 = tmp
        if len(list(data[5][mutPart].keys())[0].split('_')) > 2 or len(list(data[5][1-mutPart].keys())[0]) > 1:
            continue    # Discard, if there are more than 1 mutations in chain B or also in chain A
        if isnan(data[3]) or isnan(data[6]) or isnan(data[7]):  # Affinity values before and after mutation need to be recorded
            continue
        if len(list(data[2][0].values())[0]) < 20 or len(list(data[2][1].values())[0]) < 20:  # Retain chain length >= 20
            continue
        for chain in data[2][mutPart]:
            for chain_mut in data[5][mutPart]:
                if chain_mut[0:len(chain)] == chain:
                    part1_mut.append(data[0]+'_'+chain_mut)
        if int(chain_mut[4:-1]) > 500:  # Reserve the chains longer than 20
            continue
        for part in data[2]:
            for key in part.keys():
                if (data[0] + '_' + key) not in chain_to_sequence:
                    chain_to_sequence[data[0] + '_' + key] = part[key]
        for part in data[5]:
            for key in part.keys():
                if (data[0] + '_' + key) not in chain_to_sequence:
                    chain_to_sequence[data[0] + '_' + key] = part[key]
        Kd_origin = data[3]
        Kd_mut = data[6]
        temperature = data[7]
        DG_origin = (8.314/4184) * temperature * log(Kd_origin)
        DG_mut = (8.314/4184) * temperature * log(Kd_mut)
        DDG = DG_mut - DG_origin
        key = part0[0] + ';' + part1[0] + ';' + part1_mut[0]
        value = chain_DDG_dict.get(key)
        if value is None:
            chain_DDG_dict[key] = [DDG]
        else:
            chain_DDG_dict[key].append(DDG)
    for chain_DDG in chain_DDG_dict.items():
        chain_3, DDG_list = chain_DDG
        A, B, B_mut = chain_3.split(';')
        if calc_std_dev(DDG_list) > 1:
            continue  # big_deviation, discard
        chain_DDG_list.append([A, B, B_mut, sum(DDG_list) / len(DDG_list)])
    return chain_DDG_list

def generate_10_folds_dataset(chain_DDG_list, root_path, postfix=''):
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
        for i in range(fold_i+1, 10):
            chain_DDG_list_train[fold_i] += chain_DDG_list_folds[i]
    dump(chain_DDG_list_train, root_path+'/'+'chain_DDG_list_train'+postfix)
    dump(chain_DDG_list_test, root_path+'/'+'chain_DDG_list_test'+postfix)

def get_tensor_DDG_list(chain_DDG_path, embedding_path, edge_path, lengh_path, dump_path):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    edge_dist_dict = json.load(open(edge_path, 'r'))
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
            edge_index = torch.tensor(edge_dist_dict[chain_ddg[i][j]])
            length = torch.tensor(length_dict[chain_ddg[i][j]])
            ddg = torch.tensor(float(chain_ddg[i][3]), dtype=torch.float32)
            a = [torch.stack(chain_r, dim=0).to(device), length.to(device), edge_index.to(device), ddg.to(device)]
            result.append(a)
        dump(result, dump_path+'_'+str(fold_i))

def get_tensor_DDG_list_with_dist(chain_DDG_path, embedding_path, edge_path, lengh_path, dump_path):
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    edge_dist_dict = json.load(open(edge_path, 'r'))
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
            edge_index = torch.tensor(edge_dist_dict[chain_ddg[i][j]][:2])
            dist = torch.tensor(edge_dist_dict[chain_ddg[i][j]][-1])
            length = torch.tensor(length_dict[chain_ddg[i][j]])
            ddg = torch.tensor(float(chain_ddg[i][3]), dtype=torch.float32)
            a = [torch.stack(chain_r, dim=0).to(device), length.to(device), edge_index.to(device), dist.to(device), ddg.to(device)]
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
        for part in data[5]:
            for key in part.keys():
                if (data[0] + '_' + key) not in length_dict:
                    length_dict[data[0] + '_' + key] = len(part[key])
    with open(f'../data/length_{dataset_name}.json', 'w') as f:
        json.dump(length_dict, f)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    ''' processing embedding dict '''
    # generateProcessedCSV('../data/skempi_v2.csv', '../data/processed_v2.csv')
    # generateSequenceData('../data/processed_v2.csv', '../data/mapping/skempi', '../data/SequenceData_skempi')
    # generate_embedding_dict('../data/SequenceData_skempi', f'../data/embedding_dict_skempi')
    # generate_chain_length('../data/SequenceData_skempi')

    ''' processing skempi-1131 '''
    # chain_DDG_list = get_raw_chain_DDG_list_skempi1131('../data/skempi_1131')
    # generate_10_folds_dataset(chain_DDG_list, '../data/S1131')
    # get_tensor_DDG_list('../data/S1131/chain_DDG_list_train', '../data/embedding_dict_skempi',
    #                     '../data/S1131/s1131_edge.json', '../data/length_skempi.json',
    #                     '../data/S1131/tensor_DDG_list_train')
    # get_tensor_DDG_list('../data/S1131/chain_DDG_list_test', '../data/embedding_dict_skempi',
    #                     '../data/S1131/s1131_edge.json', '../data/length_skempi.json',
    #                     '../data/S1131/tensor_DDG_list_test')

    ''' processing skempi-2398 '''
    # chain_DDG_list = get_raw_chain_DDG_list_skempi2398('../data/SequenceData_skempi')
    # generate_10_folds_dataset(chain_DDG_list, '../data/S2398')
    # get_tensor_DDG_list('../data/S2398/chain_DDG_list_train', f'../data/embedding_dict_skempi',
    #                     '../data/S2398/s2398_edge.json', '../data/length_skempi.json',
    #                     f'../data/S2398/tensor_DDG_list_train')
    # get_tensor_DDG_list('../data/S2398/chain_DDG_list_test', f'../data/embedding_dict_skempi',
    #                     '../data/S2398/s2398_edge.json', '../data/length_skempi.json',
    #                     f'../data/S2398/tensor_DDG_list_test')