# %%
# 라이브러리 설치 
import copy
import pandas as pd
import numpy as np
import tensorflow as tf
# import tensorflow_text as text

# # GPU 할당
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # 텐서플로가 첫 번째 GPU만 사용하도록 제한
#   try:
#     tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
#   except RuntimeError as e:
#     # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
#     print(e)
# %%
# Get functions from utils.py
from utils import data_loader, len_based_truncation, node_generation, data_tokenizer, check_node_num

# Load dataset
# Load dataset from given '/data' folder. task option indicates whether the current project executes 'drug discovery or article recommendation'. 
# In case you want to do 'drug discovery' then you need to input 'generation' as an argument, otherwise need to input 'recommendation'. 
sample_data = data_loader('../data/BindingDB_BindingDB_Inhibition (prep).csv', task = "generation")

# Preprocess dataset -- step 1. Data truncation.
truncated_data, A_Partition_len_distribution, truncated_A_Partition_len_distribution, B_Partition_len_distribution = len_based_truncation(sample_data, truncate_rate = 0.2)

# Preprocess dataset -- step 2. Node generation.
truncated_data, a_code, b_code = node_generation(truncated_data, task = "generation")

# Preprocess dataset -- step 3. Tokenizing.
a_sequence, b_sequence, a_dict, b_dict = data_tokenizer(truncated_data)

# Preprocess dataset -- step 4. Exploratory Data Analysis.
num_a_nodes, maxlen_a, num_b_nodes, maxlen_b = check_node_num(a_sequence, b_sequence, truncated_data)





# # %%
# '''
# 여기서부턴 좀 나중에 코드 정리 해주기
# '''
# # 데이터 셔플링해주기
# index_list = np.arange(len(padded_FASTA))
# np.random.seed(1234)
# np.random.shuffle(index_list)
# shuffled_FASTA = padded_FASTA[index_list, :]
# shuffled_SMILES = padded_SMILES[index_list, :]


# # 데이터셋을 training (4)과 test (1)로 5-fold split하기
# from sklearn.model_selection import KFold
# i = 0
# num_splits = 5  # (4-1, 즉 80%-20%로 나누겠다는 뜻)
# X_train_split_list = [None] * num_splits
# X_test_split_list = [None] * num_splits
# y_train_split_list = [None] * num_splits
# y_test_split_list = [None] * num_splits

# for train_index, test_index in KFold(num_splits).split(shuffled_FASTA):
#     ## Train split 데이터 셋
#     X_train_split_list[i] = shuffled_FASTA[train_index, :]
#     y_train_split_list[i] = shuffled_SMILES[train_index, :]

#     ## Test split 데이터 셋
#     X_test_split_list[i] = shuffled_FASTA[test_index, :]
#     y_test_split_list[i] = shuffled_SMILES[test_index, :]

#     i += 1

#     ## 각 list가 train의 경우 10485개 관측치, test의 경우 2622개 관측치가 들어가 있음.

# # %%
# # 각 split dataset 내 unique Protein & unique SMILES 확인
# for i in range(num_splits):
#     print('---------------------------')
#     print('{}-th split'.format(i))
#     print("Protein-Train-Size : {}".format(X_train_split_list[i].shape))
#     print("Protein-Test-Size : {}".format(X_test_split_list[i].shape))

#     X_train_unq, X_train_count = np.unique(X_train_split_list[i], axis = 0, return_counts = True)
#     FASTA_train_len = len(X_train_count)    # unique Protein 갯수 (train)
#     print('Train 데이터 unique Protein 시퀀스 갯수 : {}'.format(FASTA_train_len))    
#     X_test_unq, X_test_count = np.unique(X_test_split_list[i], axis = 0, return_counts = True)
#     FASTA_test_len = len(X_test_count)      # unique Protein 갯수 (test)
#     print('Test 데이터 unique Protein 시퀀스 갯수 : {}'.format(FASTA_test_len))    
#     FASTA_intersect_len = len(np.where((X_train_unq[:, None] == X_test_unq).all(-1).any(-1) == True)[0])
#     print('Train-Test 데이터 겹치는 Protein 시퀀스 갯수 : {}'.format(FASTA_intersect_len))    


#     print('')
#     print("Compound-Train-Size : {}".format(y_train_split_list[i].shape))
#     print("Compound-Test-Size : {}".format(y_test_split_list[i].shape))

#     y_train_unq, y_train_count = np.unique(y_train_split_list[i], axis = 0, return_counts = True)
#     SMILES_train_len = len(y_train_count)   # unique Compounds 갯수 (train)
#     print('Train 데이터 unique Compounds 시퀀스 갯수 : {}'.format(SMILES_train_len))    
#     y_test_unq, y_test_count = np.unique(y_test_split_list[i], axis = 0, return_counts = True)
#     SMILES_test_len = len(y_test_count)     # unique Compounds 갯수 (test)
#     print('Test 데이터 unique Compounds 시퀀스 갯수 : {}'.format(SMILES_test_len))    
#     SMILES_intersect_len = len(np.where((y_train_unq[:, None] == y_test_unq).all(-1).any(-1) == True)[0])
#     print('Train-Test 데이터 겹치는 SMILES 시퀀스 갯수 : {}'.format(SMILES_intersect_len))    

#     print('')





# # %%
# ## (중요) ---- 기록용
# # np.all(), np.any(), array.all(), array.any() 차이 구별해서 쓰기
# X_train_unq.shape   # Train 데이터 셋의 unique Protein 갯수
# X_test_unq.shape    # Test 데이터 셋의 unique Protein 갯수

# # Train과 Test 데이터 셋에 존재하는 unique Protein들을 글자단위에서 매칭시키기
# # 차원 형태 = (train unique Protein 갯수 X test unique Protein 갯수 X Amino_acid 갯수)
# (X_train_unq[:, None] == X_test_unq).shape

# # Amino_axid가 열 방향 (-1)으로 모두 (all) match 된 경우 True, 아닌 경우 False 반환
# # 차원 형태 = (train unique Protein 갯수 X test unique Protein 갯수) 가 됨. 즉 train-test Protein pair가 모두 매치된 경우를 알려주는 행렬 구축
# (X_train_unq[:, None] == X_test_unq).all(-1).shape

# # 열 방향으로 하나라도 True가 있다면 (즉 가능한 train-test Protein pair 중에 하나라도 True라면, True를 반환하도록 하여 행렬 구축)
# # 차원 형태 = (train unique Protein 갯수 X test unique Protein 갯수)
# (X_train_unq[:, None] == X_test_unq).all(-1).any(-1).shape

# len(np.where((X_train_unq[:, None] == X_test_unq).all(-1).any(-1) == True)[0])





# %%
