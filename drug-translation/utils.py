# %%
import copy
import pandas as pd
import numpy as np
import tensorflow as tf

'''
-- (1) Preprocessing 용 utils
'''

# 데이터 로더
def data_loader(data_address, task = "generation"):
  # data_address : csv 파일 주소
  if task == "generation":
    use_cols = ['Ligand SMILES', 'BindingDB Target Chain Sequence', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
    sample_data = pd.read_csv(data_address, usecols = use_cols) # use_cols에 해당하는 변수들만 read 하기
    sample_data = sample_data[['BindingDB Target Chain Sequence', 'Ligand SMILES', 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']] # 변수들의 순서 바꿔주기; 0번째 컬럼 = A_Partition, 1번째 컬럼 = B_Partition.

  elif task == "recommendation":
    print('Not yet')

  return sample_data

# 시퀀스 길이 기반 데이터 절삭기
'''
truncate_rate (절삭률) : 시퀀스 길이 상위 truncate_rate (%) 파라미터; 
시퀀스 길이 상위 truncate_rate (%) 이상인 시퀀스들은 절삭 (제거)
'''
def len_based_truncation(sample_data, truncate_rate = 0.2):

  # 데이터 샘플링
  # (1) 단백질과 화합물의 시퀀스 길이 분포 추출
  protein_len_distribution = sample_data.apply(lambda x : len(x['BindingDB Target Chain Sequence']), axis = 1)
  compound_len_distribution = sample_data.apply(lambda x : len(x['Ligand SMILES']), axis = 1)

  # (2-1) 단백질의 경우 소수의 너무 긴 단백질이 존재하므로 길이 하위 1 - truncate_rate (%) 까지만 고려.
  protein_quantile = protein_len_distribution.quantile(1 - truncate_rate)
  truncated_protein_len_distribution = protein_len_distribution[protein_len_distribution.lt(protein_quantile)]

  # (2-2) protein sequence는 길이 하위 1 - truncate_rate (%) 에 해당하는 샘플만 고려하는 truncated_data 만들어주기
  truncated_data = sample_data[protein_len_distribution.lt(protein_quantile)]

  return truncated_data, protein_len_distribution, truncated_protein_len_distribution, compound_len_distribution

# 노드 생성기
'''
각 시퀀스에 대응하는 노드를 의미하는 변수 (카테고리 및 코드) 생성
'''
def node_generation(truncated_data, task = "generation"):

  # A 파티션과 B 파티션 내 노드들의 카테고리 변수 추출후 해당 컬럼 추가
  if task == "generation":
    truncated_data.loc[:, 'A_Partition_Category'] = np.array([str('P') + str(i) for i in pd.factorize(truncated_data.iloc[:, 0])[0].tolist()])
    truncated_data.loc[:, 'B_Partition_Category'] = np.array([str('C') + str(i) for i in pd.factorize(truncated_data.iloc[:, 1])[0].tolist()])
  elif task == "recommendation":
    truncated_data.loc[:, 'A_Partition_Category'] = np.array([str('From') + str(i) for i in pd.factorize(truncated_data.iloc[:, 0])[0].tolist()])
    truncated_data.loc[:, 'B_Partition_Category'] = np.array([str('To') + str(i) for i in pd.factorize(truncated_data.iloc[:, 1])[0].tolist()])

  # A 파티션과 B 파티션 내 노드들의 코드 변수 추출후 해당 컬럼 추가
  a_code = pd.factorize(truncated_data.iloc[:, 0])[0]
  b_code = pd.factorize(truncated_data.iloc[:, 1])[0]
  truncated_data.loc[:, 'A_Partition_Code'] = a_code.tolist()
  truncated_data.loc[:, 'B_Partition_Code'] = b_code.tolist()
  truncated_data.loc[:, 'Connection'] = 1

  # 코드 변수 원핫 임베딩
  a_code_onehot = tf.one_hot(a_code, depth = np.max(a_code) + 1)
  b_code_onehot = tf.one_hot(b_code, depth = np.max(b_code) + 1)

  return truncated_data, a_code_onehot, b_code_onehot

# 시퀀스 토크나이저
'''
데이터 내 시퀀스의 토큰들을 추출, 정수 임베딩, 사전생성을 수행
'''
def data_tokenizer(truncated_data):
  # 텐서플로 텍스트 및 시퀀스 라이브러리 설치
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences

  # (0) <bos>와 <eos> 토큰 추가
  A_sequence = truncated_data.iloc[:, 0].apply(lambda x : ['<bos>'] + list(x) + ['<eos>'])
  B_sequence = truncated_data.iloc[:, 1].apply(lambda x : ['<bos>'] + list(x) + ['<eos>'])

  # (1) 시퀀스 정수 임베딩
  ## A_Partition 내 시퀀스 (노드)들 토크나이징
  A_tokenizer = Tokenizer(filters = ' ', lower = False)
  A_tokenizer.fit_on_texts(A_sequence)
  encoded_A_sequence = A_tokenizer.texts_to_sequences(A_sequence)

  ## B_Partition 내 시퀀스 (노드)들 토크나이징
  B_tokenizer = Tokenizer(filters = ' ', lower = False)
  B_tokenizer.fit_on_texts(B_sequence)
  encoded_B_sequence = B_tokenizer.texts_to_sequences(B_sequence)

  # (2) 시퀀스의 최대길이 (BOS, EOS 추가 후) 추출
  A_maxlen = A_sequence.apply(lambda x : len(x)).max()
  B_maxlen = B_sequence.apply(lambda x : len(x)).max()

  # (3) 시퀀스 패딩하기
  padded_A_sequence = pad_sequences(encoded_A_sequence, maxlen = A_maxlen, padding = 'post')
  padded_B_sequence = pad_sequences(encoded_B_sequence, maxlen = B_maxlen, padding = 'post')

  # (4) 사전 만들기
  A_dict = copy.deepcopy(A_tokenizer.word_index)
  A_dict_reverse = dict(map(reversed, A_dict.items()))
  B_dict = copy.deepcopy(B_tokenizer.word_index)
  B_dict_reverse = dict(map(reversed, B_dict.items()))

  return padded_A_sequence, padded_B_sequence, A_dict, B_dict

'''
-- (2) EDA 용 utils
'''
def check_node_num(a_sequence, b_sequence, truncated_data, task = "generation"):
    _, unique_nodes_A = np.unique(a_sequence, axis = 0, return_counts = True)
    _, unique_nodes_B = np.unique(b_sequence, axis = 0, return_counts = True)
  
    num_unique_nodes_A = len(unique_nodes_A)
    num_unique_nodes_B = len(unique_nodes_B)
    A_max_sequence_len = max(truncated_data.iloc[:, 0].apply(lambda x : len(x)))
    B_max_sequence_len = max(truncated_data.iloc[:, 1].apply(lambda x : len(x)))

    if task == "generation":

        print('단백질 종류 갯수 : {}, 최장 시퀀스 길이 : {}'.format(num_unique_nodes_A, A_max_sequence_len))
        print('화합물 종류 갯수 : {}, 최장 시퀀스 길이 : {}'.format(num_unique_nodes_B, B_max_sequence_len))

    elif task == "recommendation":
        print('인용 문헌 갯수 : {}, 최장 시퀀스 길이 : {}'.format(num_unique_nodes_A, A_max_sequence_len))
        print('피인용 문헌 갯수 : {}, 최장 시퀀스 길이 : {}'.format(num_unique_nodes_B, B_max_sequence_len))


    return num_unique_nodes_A, A_max_sequence_len, num_unique_nodes_B, B_max_sequence_len


'''
-- Modeling 용 utils
'''
# (1) 포지션 인코더
class Position_Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        self.d_model = kwargs['d_model']

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, len_pos, d_model):
        positions = np.arange(len_pos)              # position list array 만들기
        dimension_indices = np.arange(d_model)      # dimension index list array 만들기
        angle_rads =  self.get_angles(positions[:, np.newaxis], 
                                        dimension_indices[np.newaxis, :], 
                                        d_model)

        # 어레이의 짝수 인덱스에 sin을 적용 (2i)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # 어레이의 홀수 인덱스에 cos를 적용; (2i+1)
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # pos_encoding은 (1, len_pos, d_model) 크기의 차원을 가짐
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, embedding_matrix):
        len_pos = embedding_matrix.shape[1]
        pos_encoding = self.positional_encoding(len_pos, self.d_model)

        return pos_encoding

# (2) 마스크 제너레이터
class Mask_Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)

    def padding_mask(self, seq):
        # keras.tokenizer를 사용할경우, padding token 값은 0이되므로, 토큰값이 0인 위치만 찾아서 True (1), 나머지 토큰값들은 False (0)을 만들어준다.
        seq = tf.cast(tf.math.equal(seq, 0), dtype = tf.float32)

        # attention_map = attention_logits 의 차원은 4차원 ((batch_size, num_heads, seq_len, seq_len))이므로 
        # padding_mask도 4차원으로 만들어주어야 함.
        padding_mask = seq[:, np.newaxis, np.newaxis, :] * -10e9

        return padding_mask

    # (대칭행렬인) attention_map에 대하여, 현재 position 이후의 token들에 대해서 mask를 씌워
    # # 참조를 못하게 하는함수
    def subsequent_mask(self, tar_len):
        # 상삼각함수만 1로 만들기
        mask_matrix = tf.linalg.band_part(tf.ones((tar_len, tar_len)), 0, -1)
        subseq_mask = mask_matrix * -10e9
        return subseq_mask
    
    def call(self, inp, tar):
        # 인코더에서 패딩 부분 마스크
        enc_padding_mask = self.padding_mask(inp)

        # 디코더에서 사용되는 인코더의 아웃풋에서 패딩 부분 마스크 (디코더의 두번째 어텐션 블록에서 활용)
        dec_padding_mask = self.padding_mask(inp)

        # (디코더의 첫번째 어텐션 블록에서 활용)
        subseq_mask = self.subsequent_mask(tar.shape[1])
        dec_target_padding_mask = self.padding_mask(tar)
        
        # subsequent mask를 생성하기 위해 target의 length 정의
        tar_len = tar.shape[1]
        dec_subsequent_mask = self.subsequent_mask(tar_len)

        return enc_padding_mask, dec_padding_mask, dec_subsequent_mask

class Compile_Params(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        self.optimizer = tf.keras.optimizers.Adam(lr = 1e-4)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'accuracy')

    # 손실함수 정의
    def loss(self, real, pred):
        # padding인 token들 (i.e., 값이 0인 token들)은 무시하는 mask 정의
        mask = tf.math.logical_not(tf.math.equal(real, 0))  # real 데이터에서 padding이 아닌 부분에만 1값을 마스킹함.
        loss_ = self.loss_object(real, pred) # SparseCategoricalCrossentropy를 활용하여 loss함수 정의
       
        mask = tf.cast(mask, dtype = loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # 평가척도 정의
    def accuracy(self, real, pred):
        # padding인 token들 (i.e., 값이 0인 token들)은 무시하는 mask 정의
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.expand_dims(tf.cast(mask, dtype = pred.dtype), axis = -1) #  SparseCategoricalAccuracy를 활용하여 평가척도 정의
        pred *= mask
        acc = self.train_accuracy(real, pred)

        return tf.reduce_mean(acc)

'''
-- (3) post analysis 용 utils
'''
# 성능 시각화
def plot_graphs(history, string):
    plt.plot(history.history[string])
    # plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    # plt.legend([string, 'val_'+string])
    plt.show()

# %%
# import matplotlib.pyplot as plt
# # 포지션 인코딩 마스크 플롯
# plt.pcolormesh(tf.transpose(tf.squeeze(pos_encoding)), cmap='RdBu')
# plt.ylabel('Depth')
# plt.xlabel('Position')
# plt.colorbar()
# plt.show()
