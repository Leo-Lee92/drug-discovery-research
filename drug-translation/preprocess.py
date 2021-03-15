# %%
# 라이브러리 설치 
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.patches as mpatches
import tensorflow as tf
import tensorflow_text as text
# %%
# 데이터 로드
raw_data = pd.read_csv('/home/messy92/Leo/Drug-discovery-research/data/BindingDB_BindingDB_Inhibition (prep).csv')
sample_data = raw_data.loc[:, ['Ligand SMILES', 'BindingDB Target Chain Sequence', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']]

# %%
# 길이 분포 플롯팅하기
# (1) 단백질과 화합물의 시퀀스 길이 분포 추출
protein_len_distribution = raw_data.apply(lambda x : len(x['BindingDB Target Chain Sequence']), axis = 1)
compound_len_distribution = raw_data.apply(lambda x : len(x['Ligand SMILES']), axis = 1)

# (2-1) 단백질의 경우 소수의 너무 긴 단백질이 존재하므로 길이 하위 80% 까지만 고려.
protein_q80 = protein_len_distribution.quantile(0.8)
truncated_protein_len_distribution = protein_len_distribution[protein_len_distribution.lt(protein_q80)]
# (2-2) protein sequence는 길이 하위 80%에 해당하는 샘플만 고려하는 trunc_sample_data 만들어주기
trunc_sample_data = sample_data[protein_len_distribution.lt(protein_q80)]

# (3) 길이 데이터 만들기
# (3-1) protein 길이 데이터 만들기 (시퀀스 대상과 시퀀스 길이를 의미하는 변수로 구성된 2차원 데이터 프레임 만들기)
protein_len_data = pd.concat([pd.Series(np.repeat("protein", len(truncated_protein_len_distribution))), truncated_protein_len_distribution], axis = 1)
# (3-2) compound 길이 데이터 만들기 (시퀀스 대상과 시퀀스 길이를 의미하는 변수로 구성된 2차원 데이터 프레임 만들기)
compound_len_data = pd.concat([pd.Series(np.repeat("compound", len(compound_len_distribution))), compound_len_distribution], axis = 1)

# (4) 위에서 만든 두 개의 데이터 프레임을 행방향으로 합치기
sequence_len_dist_data = pd.concat([protein_len_data, compound_len_data], axis = 0)
sequence_len_dist_data.columns = ['sequence', 'length']

# (5) 풀롯팅 하기
sns.distplot(sequence_len_dist_data[sequence_len_dist_data.sequence == "protein"]['length'], label = "protein", kde = False, color = 'tab:red')
sns.distplot(sequence_len_dist_data[sequence_len_dist_data.sequence == "compound"]['length'], label = "compound", kde = False, color = 'tab:blue')
plt.style.use('ggplot')
plt.ylabel('Frequency')
plt.ylabel('Sequence length')
plt.legend(title = 'Sequence', facecolor = 'white')
# plt.title('Sequence Length Distribution')
plt.savefig('/home/messy92/Leo/Drug-discovery-research/images/sequence_length_distribution')

# %%
# 단백질-화합물 네트워크 데이터 (그래프 데이터) 구축
# trunc_sample_data['FASTA Category'] = trunc_sample_data['BindingDB Target Chain Sequence'].astype('category')
# trunc_sample_data['SMILES Category'] = trunc_sample_data['Ligand SMILES'].astype('category')
# trunc_sample_data['FASTA Category'] = pd.factorize(trunc_sample_data['BindingDB Target Chain Sequence'])[0]
# trunc_sample_data['SMILES Category'] = pd.factorize(trunc_sample_data['Ligand SMILES'])[0]

# (1) 단백질 시퀀스와 화합물 시퀀스를 카테고리 타입 (factorize)으로 변환후 컬럼 추가
trunc_sample_data['FASTA Category'] = np.array([str('P') + str(i) for i in pd.factorize(trunc_sample_data['BindingDB Target Chain Sequence'])[0].tolist()])
trunc_sample_data['SMILES Category'] = np.array([str('C') + str(i) for i in pd.factorize(trunc_sample_data['Ligand SMILES'])[0].tolist()])
trunc_sample_data['connection'] = 1

# (2-1) 단백질-화합물 network matrix (transaction matrix) 만들어주기
protein_network_matrix = pd.pivot_table(trunc_sample_data, values = 'connection', index = 'SMILES Category', columns = 'FASTA Category', fill_value = 0)
print(protein_network_matrix.sum(axis=0).sort_values(ascending = False))    # FASTA 별 관련 SMILES 갯수

# (2-2) 단백질 별 활성화 화합물 갯수 플로팅 (The number of bound ligands)
top_n = 30
protein_code = protein_network_matrix.sum(axis=0).sort_values(ascending = True).index.astype('str')[-top_n:]
transaction_freqs = protein_network_matrix.sum(axis=0).sort_values(ascending = True)[-top_n:]
plt.style.use('ggplot')
plt.yticks(fontsize = 300/top_n - 2.5)
plt.barh(protein_code, transaction_freqs, color = 'tab:red', alpha = 0.5)
# plt.title('Num of SMILES per FASTA')
plt.ylabel('Protein code (top ' + str(top_n) + ')')
plt.xlabel('The number of bound ligands')
plt.savefig('/home/messy92/Leo/Drug-discovery-research/images/num_bound_ligands')

# (3-1) 화합물-단백질 network matrix (transaction matrix) 만들어주기
compound_network_matrix = pd.pivot_table(trunc_sample_data, values = 'connection', index = 'FASTA Category', columns = 'SMILES Category', fill_value = 0)
print(compound_network_matrix.sum(axis=0).sort_values(ascending = False))    # FASTA 별 관련 SMILES 갯수

# (3-2) 화합물 별 표적 단백질 갯수 플로팅 (The number of target proteins)
top_n = 30
compound_code = compound_network_matrix.sum(axis=0).sort_values(ascending = True).index.astype('str')[-top_n:]
transaction_freqs = compound_network_matrix.sum(axis=0).sort_values(ascending = True)[-top_n:]
plt.style.use('ggplot')
plt.yticks(fontsize = 300/top_n - 2.5)
plt.barh(compound_code, transaction_freqs, color = 'tab:blue', alpha = 0.5)
# plt.title('Num of FASTA per SMILES')
plt.ylabel('Compound code (top ' + str(top_n) + ')')
plt.xlabel('The number of targets')
plt.savefig('/home/messy92/Leo/Drug-discovery-research/images/num_activating_targets')

# %%
# 네트워크 플롯팅
# 참고 1: https://www.python-graph-gallery.com/network-chart/
# 참고 2: https://networkx.org/documentation/latest/_downloads/networkx_reference.pdf
top_n = 30
protein_code = protein_network_matrix.sum(axis=0).sort_values(ascending = True).index.astype('str')[-top_n:]
compound_code = compound_network_matrix.sum(axis=0).sort_values(ascending = True).index.astype('str')[-top_n:]

all_idx = np.empty([])
all_code = np.empty([])

# for i in protein_code[:top_n]:
for i in compound_code[:top_n]:
    # idx = np.where(np.array(trunc_sample_data['FASTA Category']) == i)[0]
    idx = np.where(np.array(trunc_sample_data['SMILES Category']) == i)[0]
    code = np.repeat(i, len(idx))
    all_code = np.append(all_code, code)    # 코드의 str값을 담은 array
    all_idx = np.append(all_idx, idx)       # 각 코드에 해당하는 index값을 담은 array
all_code = all_code[1:]
all_idx = all_idx[1:].astype('int')

all_idx.sort()
np.unique(all_code)
len(all_idx)

trunc_data_small = trunc_sample_data[['FASTA Category', 'SMILES Category']].iloc[all_idx]

# node의 색깔을 결정하는 property dataframe 만들기
all_protein_code = np.array(trunc_data_small['FASTA Category'])
all_compound_code = np.array(trunc_data_small['SMILES Category'])
len_protein_codes = len(np.unique(all_protein_code))
len_compound_codes = len(np.unique(all_compound_code))

# 단백질, 화합물 코드 벡터 연결하기
all_code_ids = np.append(np.unique(all_protein_code), np.unique(all_compound_code))

# 단백질, 화합물 색값 벡터 연결 및 각 색깔 할당 (단백질에 'tab:red', 화합물에 'tab:blue' 값을 매핑)
all_code_color_values = copy.deepcopy(all_code_ids)
all_code_color_values[np.arange(len_protein_codes)] = 'tab:red'
all_code_color_values[np.arange(len_protein_codes, len(all_code_ids))] = 'tab:blue'

# 코드 벡터와 색값 벡터를 컬럼으로 하는 dataframe 생성
property_dat = pd.DataFrame([all_code_ids, all_code_color_values]).T
property_dat.columns = ['code', 'color']

# 네트워크 그리기
G = nx.from_pandas_edgelist(trunc_data_small, 'FASTA Category', 'SMILES Category')

# property dat ㅇㅇㅇ
property_dat = property_dat.set_index('code')
property_dat = property_dat.reindex(np.array(G.nodes()))

fig = plt.figure()
nx.draw(G, with_labels=True, node_size=75, node_color = property_dat['color'], edge_color='black', linewidths=1, font_size=5, pos = graphviz_layout(G, prog = 'neato'), alpha = 0.5)
red_patch = mpatches.Patch(color='tab:red', label='protein', alpha = 0.5)
blue_patch = mpatches.Patch(color='tab:blue', label='compound', alpha = 0.5)
plt.legend(handles=[red_patch, blue_patch], facecolor = 'white', title = "Sequence")
plt.axis('on')
plt.savefig('/home/messy92/Leo/Drug-discovery-research/images/protein-compound_network.png')
plt.show()


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
compound_dict = copy.deepcopy(compound_tokenizer.word_index)

# (4) 데이터 내 전체 unique Protein, unique Compound 갯수 확인
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
## (중요)
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
