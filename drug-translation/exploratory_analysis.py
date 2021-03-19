# %%
# Run this line only when you want to run whole file at once.
from preprocess import *

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.patches as mpatches

# %%
# raw data에 존재하는 unique 단백질, unique 약물 갯수 요약
len(np.unique(raw_data['BindingDB Target Chain Sequence']))
len(np.unique(raw_data['Ligand SMILES']))

# raw data에 존재하는 모든 단백질, 약물 길이 분포 통계량 요약
protein_len_rawdata = pd.concat([pd.Series(np.repeat("protein", len(protein_len_distribution))), protein_len_distribution], axis = 1)
compound_len_rawdata = pd.concat([pd.Series(np.repeat("compound", len(compound_len_distribution))), compound_len_distribution], axis = 1)
sequence_len_dist_rawdata = pd.concat([protein_len_rawdata, compound_len_rawdata], axis = 0)
sequence_len_dist_rawdata.columns = ['sequence', 'length']
sequence_len_dist_rawdata.groupby('sequence').describe()


# %%
# 길이 분포 플롯팅하기
# (3) 길이 데이터 만들기
# (3-1) protein 길이 데이터 만들기 (시퀀스 대상과 시퀀스 길이를 의미하는 변수로 구성된 2차원 데이터 프레임 만들기)
protein_len_data = pd.concat([pd.Series(np.repeat("protein", len(truncated_protein_len_distribution))), truncated_protein_len_distribution], axis = 1)
# (3-2) compound 길이 데이터 만들기 (시퀀스 대상과 시퀀스 길이를 의미하는 변수로 구성된 2차원 데이터 프레임 만들기)
compound_len_data = pd.concat([pd.Series(np.repeat("compound", len(compound_len_distribution))), compound_len_distribution], axis = 1)

# (4) 위에서 만든 두 개의 데이터 프레임을 행방향으로 합치기
sequence_len_dist_data = pd.concat([protein_len_data, compound_len_data], axis = 0)
sequence_len_dist_data.columns = ['sequence', 'length']
sequence_len_dist_data.groupby('sequence').describe()

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