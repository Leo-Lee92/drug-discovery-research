# %%
# 라이브러리 설치 
import copy
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as text
# %%
# 데이터 로드
raw_data = pd.read_csv('/home/messy92/Leo/Drug-discovery-research/data/BindingDB_BindingDB_Inhibition (prep).csv')
sample_data = raw_data.loc[:, ['Ligand SMILES', 'BindingDB Target Chain Sequence', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']]

# %%
# 데이터 샘플링
# (1) 단백질과 화합물의 시퀀스 길이 분포 추출
protein_len_distribution = raw_data.apply(lambda x : len(x['BindingDB Target Chain Sequence']), axis = 1)
compound_len_distribution = raw_data.apply(lambda x : len(x['Ligand SMILES']), axis = 1)

# (2-1) 단백질의 경우 소수의 너무 긴 단백질이 존재하므로 길이 하위 80% 까지만 고려.
protein_q80 = protein_len_distribution.quantile(0.8)
truncated_protein_len_distribution = protein_len_distribution[protein_len_distribution.lt(protein_q80)]
# (2-2) protein sequence는 길이 하위 80%에 해당하는 샘플만 고려하는 trunc_sample_data 만들어주기
trunc_sample_data = sample_data[protein_len_distribution.lt(protein_q80)]

# %%
# 단백질 및 화합물의 최대 길이
protein_max_len = max(trunc_sample_data['BindingDB Target Chain Sequence'].apply(lambda x : len(x)))
compound_max_len = max(trunc_sample_data['Ligand SMILES'].apply(lambda x : len(x)))

# %%
# 데이터 구축
# 표적 단백질 시퀀스  (Amino acids sequence represented as FASTA)
FASTA_dat = np.array(copy.deepcopy(trunc_sample_data['BindingDB Target Chain Sequence']))
for idx, FASTA in enumerate(trunc_sample_data['BindingDB Target Chain Sequence']):
    FASTA_dat[idx] = list(FASTA)

FASTA_dat = list(FASTA_dat)

# 약물 시퀀스 (compound sequence represented as SMILES)
SMILES_dat = np.array(copy.deepcopy(trunc_sample_data['Ligand SMILES']))
for idx, SMILES in enumerate(trunc_sample_data['Ligand SMILES']):
    SMILES_dat[idx] = list(SMILES)

SMILES_dat = list(SMILES_dat)

# %%
# protein sequence (Amino acids)와 compound sequence (SMILES) 데이터 토큰화

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# (1) 단백질 시퀀스 정수 임베딩
protein_tokenizer = Tokenizer(filters = ' ', lower = False)
protein_tokenizer.fit_on_texts(FASTA_dat)
encoded_FASTA = protein_tokenizer.texts_to_sequences(FASTA_dat)
padded_FASTA = pad_sequences(encoded_FASTA, maxlen = protein_max_len, padding = 'post')

# (2) 약물 시퀀스 정수 임베딩
compound_tokenizer = Tokenizer(filters = ' ', lower = False)
compound_tokenizer.fit_on_texts(SMILES_dat)
encoded_SMILES = compound_tokenizer.texts_to_sequences(SMILES_dat)
padded_SMILES = pad_sequences(encoded_SMILES, maxlen = compound_max_len, padding = 'post')

# (3) 단백질 및 약물 사전 만들기
protein_dict = copy.deepcopy(protein_tokenizer.word_index)
protein_dict['BOS'] = len(protein_dict) + 1
protein_dict['EOS'] = len(protein_dict) + 1

compound_dict = copy.deepcopy(compound_tokenizer.word_index)
compound_dict['BOS'] = len(compound_dict) + 1
compound_dict['EOS'] = len(compound_dict) + 1

# (4) 단백질, 약물 array에 BOS, EOS 추가
beg_token_FASTA = np.repeat(protein_dict['BOS'], padded_FASTA.shape[0]).reshape((-1, 1))
end_token_FASTA = np.repeat(protein_dict['EOS'], padded_FASTA.shape[0]).reshape((-1, 1))
padded_FASTA = np.concatenate((beg_token_FASTA, padded_FASTA, end_token_FASTA), axis = 1)

beg_token_SMILES = np.repeat(compound_dict['BOS'], padded_SMILES.shape[0]).reshape((-1, 1))
end_token_SMILES = np.repeat(compound_dict['EOS'], padded_SMILES.shape[0]).reshape((-1, 1))
padded_SMILES = np.concatenate((beg_token_SMILES, padded_SMILES, end_token_SMILES), axis = 1)

# (5) 단백질, 약물 시퀀스의 최대길이
protein_maxlen = trunc_sample_data['BindingDB Target Chain Sequence'].apply(lambda x : len(x)).max()
compound_maxlen = trunc_sample_data['Ligand SMILES'].apply(lambda x : len(x)).max()

# (6) 데이터 내 전체 unique Protein, unique Compound 갯수 확인
_, unq_proteins = np.unique(padded_FASTA, axis = 0, return_counts = True)
_, unq_compounds = np.unique(padded_SMILES, axis = 0, return_counts = True)
print('단백질 종류 갯수 : {}'.format(len(unq_proteins)))
print('화합물 종류 갯수 : {}'.format(len(unq_compounds)))

# %%
# 데이터 셔플링해주기
index_list = np.arange(len(padded_FASTA))
np.random.seed(1234)
np.random.shuffle(index_list)
shuffled_FASTA = padded_FASTA[index_list, :]
shuffled_SMILES = padded_SMILES[index_list, :]

# 데이터셋을 training (4)과 test (1)로 5-fold split하기
from sklearn.model_selection import KFold
i = 0
num_splits = 5  # (4-1, 즉 80%-20%로 나누겠다는 뜻)
X_train_split_list = [None] * num_splits
X_test_split_list = [None] * num_splits
y_train_split_list = [None] * num_splits
y_test_split_list = [None] * num_splits

for train_index, test_index in KFold(num_splits).split(shuffled_FASTA):
    ## Train split 데이터 셋
    X_train_split_list[i] = shuffled_FASTA[train_index, :]
    y_train_split_list[i] = shuffled_SMILES[train_index, :]

    ## Test split 데이터 셋
    X_test_split_list[i] = shuffled_FASTA[test_index, :]
    y_test_split_list[i] = shuffled_SMILES[test_index, :]

    i += 1

    ## 각 list가 train의 경우 10485개 관측치, test의 경우 2622개 관측치가 들어가 있음.

# %%
# 각 split dataset 내 unique Protein & unique SMILES 확인
for i in range(num_splits):
    print('---------------------------')
    print('{}-th split'.format(i))
    print("Protein-Train-Size : {}".format(X_train_split_list[i].shape))
    print("Protein-Test-Size : {}".format(X_test_split_list[i].shape))

    X_train_unq, X_train_count = np.unique(X_train_split_list[i], axis = 0, return_counts = True)
    FASTA_train_len = len(X_train_count)    # unique Protein 갯수 (train)
    print('Train 데이터 unique Protein 시퀀스 갯수 : {}'.format(FASTA_train_len))    
    X_test_unq, X_test_count = np.unique(X_test_split_list[i], axis = 0, return_counts = True)
    FASTA_test_len = len(X_test_count)      # unique Protein 갯수 (test)
    print('Test 데이터 unique Protein 시퀀스 갯수 : {}'.format(FASTA_test_len))    
    FASTA_intersect_len = len(np.where((X_train_unq[:, None] == X_test_unq).all(-1).any(-1) == True)[0])
    print('Train-Test 데이터 겹치는 Protein 시퀀스 갯수 : {}'.format(FASTA_intersect_len))    


    print('')
    print("Compound-Train-Size : {}".format(y_train_split_list[i].shape))
    print("Compound-Test-Size : {}".format(y_test_split_list[i].shape))

    y_train_unq, y_train_count = np.unique(y_train_split_list[i], axis = 0, return_counts = True)
    SMILES_train_len = len(y_train_count)   # unique Compounds 갯수 (train)
    print('Train 데이터 unique Compounds 시퀀스 갯수 : {}'.format(SMILES_train_len))    
    y_test_unq, y_test_count = np.unique(y_test_split_list[i], axis = 0, return_counts = True)
    SMILES_test_len = len(y_test_count)     # unique Compounds 갯수 (test)
    print('Test 데이터 unique Compounds 시퀀스 갯수 : {}'.format(SMILES_test_len))    
    SMILES_intersect_len = len(np.where((y_train_unq[:, None] == y_test_unq).all(-1).any(-1) == True)[0])
    print('Train-Test 데이터 겹치는 SMILES 시퀀스 갯수 : {}'.format(SMILES_intersect_len))    

    print('')
 





# %%
## (중요) ---- 기록용
# np.all(), np.any(), array.all(), array.any() 차이 구별해서 쓰기
X_train_unq.shape   # Train 데이터 셋의 unique Protein 갯수
X_test_unq.shape    # Test 데이터 셋의 unique Protein 갯수

# Train과 Test 데이터 셋에 존재하는 unique Protein들을 글자단위에서 매칭시키기
# 차원 형태 = (train unique Protein 갯수 X test unique Protein 갯수 X Amino_acid 갯수)
(X_train_unq[:, None] == X_test_unq).shape

# Amino_axid가 열 방향 (-1)으로 모두 (all) match 된 경우 True, 아닌 경우 False 반환
# 차원 형태 = (train unique Protein 갯수 X test unique Protein 갯수) 가 됨. 즉 train-test Protein pair가 모두 매치된 경우를 알려주는 행렬 구축
(X_train_unq[:, None] == X_test_unq).all(-1).shape

# 열 방향으로 하나라도 True가 있다면 (즉 가능한 train-test Protein pair 중에 하나라도 True라면, True를 반환하도록 하여 행렬 구축)
# 차원 형태 = (train unique Protein 갯수 X test unique Protein 갯수)
(X_train_unq[:, None] == X_test_unq).all(-1).any(-1).shape

len(np.where((X_train_unq[:, None] == X_test_unq).all(-1).any(-1) == True)[0])





# %%
