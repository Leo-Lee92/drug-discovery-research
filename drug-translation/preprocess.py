# %%
# 라이브러리 설치 
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_text as text
# %%
# 데이터 로드
raw_data = pd.read_csv('/home/messy92/Leo/Drug-discovery-research/data/BindingDB_BindingDB_Inhibition (prep).csv')
sample_data = raw_data.loc[:, ['Ligand SMILES', 'BindingDB Target Chain Sequence', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']]

# 단백질과 화합물의 시퀀스 길이 분포 추출
protein_len_distribution = raw_data.apply(lambda x : len(x['BindingDB Target Chain Sequence']), axis = 1)
compound_len_distribution = raw_data.apply(lambda x : len(x['Ligand SMILES']), axis = 1)

# 단백질의 경우 소수의 너무 긴 단백질이 존재하므로 길이 하위 80% 까지만 고려.
protein_q80 = protein_len_distribution.quantile(0.8)
truncated_protein_len_distribution = protein_len_distribution[protein_len_distribution.lt(protein_q80)]

# protein sequence는 길이 하위 80%에 해당하는 샘플만 고려하는 trunc_sample_data 만들어주기
trunc_sample_data = sample_data[protein_len_distribution.lt(protein_q80)]

# %%
# 토큰 딕셔너리 만들기
token_set = set(''.join(list(trunc_sample_data['Ligand SMILES'])))
num_tokens = len(token_set)
token_dict = dict(zip(list(range(num_tokens)), sorted(list(token_set))))

# protein sequence (Amino acids)와 compound sequence (SMILES) 데이터 토큰화
protein_max_len = max(trunc_sample_data['BindingDB Target Chain Sequence'].apply(lambda x : len(x)))
compound_max_len = max(trunc_sample_data['Ligand SMILES'].apply(lambda x : len(x)))

# %%
SMILES_dat = np.array(copy.deepcopy(trunc_sample_data['Ligand SMILES']))
for idx, SMILES in enumerate(trunc_sample_data['Ligand SMILES']):
    SMILES_dat[idx] = list(SMILES)

SMILES_dat = list(SMILES_dat)

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

t = Tokenizer(filters = ' ', lower = False)
t.fit_on_texts(SMILES_dat)
t.word_index
encoded_SMILES = t.texts_to_sequences(SMILES_dat)
padded_SMILES = pad_sequences(encoded_SMILES, maxlen = compound_max_len, padding = 'post')

# %%
# for key, val in token_dict.items():
#     target_seq = np.array(list(sample_data['Ligand SMILES'][0]))
    
#     = np.where(target_seq == val)[0]

# %%
# %%
