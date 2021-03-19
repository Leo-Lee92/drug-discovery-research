# %%
import tensorflow as tf 
import numpy as np

# %%
model_name = 'graph_convolution_transformer'
protein_vocab_size = len(protein_dict) + 1
compound_vocab_size = len(compound_dict) + 1


kwargs = {
    'model_name' : model_name,
    'num_layers' : 6,
    'd_model' : 256,
    'num_heads' : 8,
    'dff' : 2048,
    'input_vocab_size' : protein_vocab_size,
    'target_vocab_size' : compound_vocab_size,
    # 'maximum_position_encoding' : MAX_SEQUENCE,
    'rate' : 0.1
}

class Transformer(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        # (1) 인코더 = f(인코더 레이어)
        # (2) 디코더

    def call(self, x):
        return dd

class Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        self.num_layers = kwargs['num_layers']
        self.d_model = kwargs['d_model']

        # (1) 임베딩 레이어
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = kwargs['input_vocab_size'], output_dim = 512)

        # (2) 포지션 인코딩 마스크
        self.pos_mask = positional_encoding(kwargs['maximum_position_encoding'], self.d_model)

    def call(self, x):

        # (1) 토큰 임베딩
        # embedding_matrix 반환
        embedding_matrix = embedding_layer(x)

        # (2) 토큰 포지션 인코딩
        # positional_encoding 하기
        len_position = embedding_matrix.shape[1] # 포지션 총 길이 변수 (len_position)
        d_model = embedding_matrix.shape[-1]    # 임베딩 총 차원크기 변수 (d_model) 정의

        # (3)ㅇㅇ

class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)

        # (1) 임베딩 레이어
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = kwargs['target_vocab_size'], output_dim = 512)

        # (2) 포지션 인코딩


class Encoder_layer(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        ddd

    def call(self, x):
        ddd 


# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # # Currently, memory growth needs to be the same across GPUs
    # for gpu in gpus:
    #   tf.config.experimental.set_virtual_device_configuration(gpu,
    #   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')[1]
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# %%
# 인코더만 짜보기
# for x in tfdata_protein_train):
    # 파라미터 정의
    # d_model = 512                       # 임베딩 총 차원크기 변수 (d_model) 정의 - kwargs
    # num_heads = 8                       # head 갯수
    depth = d_model // num_heads        # 총 임베딩 차원을 head 갯수만큼 분할해줬을 때 각 head별 임베딩 차원 크기
    # len_position = protein_maxlen       # 포지션 인코딩 적용할 최대 길이

    # 레이어 정의 
    embedding_layer = tf.keras.layers.Embedding(input_dim = len(protein_dict) + 1, output_dim = d_model)
    wq = tf.keras.layers.Dense(units = d_model)
    wk = tf.keras.layers.Dense(units = d_model)
    wv = tf.keras.layers.Dense(units = d_model)

    # (1) 토큰 임베딩
    # embedding_matrix 반환
    x = list(tfdata_protein_train)[0]
    embedding_matrix = embedding_layer(x)

    # (2) 토큰 포지션 인코딩
    PE = Position_Encoder(**kwargs)
    pos_encoding = PE(embedding_matrix)
    # pos_encoded_embedding_matrix = embedding_matrix + pos_encoding

    # (3) 멀티헤드 어텐션
    Q = wq(embedding_matrix)    # linear approximation 시켜주기
    K = wk(embedding_matrix)
    V = wv(embedding_matrix)

    def split_heads(x, batch_size):
        
        # d_model을 head 갯수로 나눠 depth크기를 만들기 위해서 나머지 모든 차원들은 특정 값으로 고정되어야 함.
        # 텐서의 축은 (batch_size, sequence_len, d_model) -> (batch_size, num_heads, sequence_len, depth)가 되어야 함
        # 이를 위해서 batch_size라는 파라미터를 사전에 설정하여 고정해줄 필요가 있음.

        x = tf.reshape(x, shape = (batch_size, -1, 8, 64))
        return tf.transpose(x, perm = [0, 2, 1, 3])

    q = split_heads(Q, BATCH_SIZE)
    k = split_heads(K, BATCH_SIZE)
    attention_map = tf.matmul(q, k, transpose_b = True) 
    attention_logits = attention_map / tf.cast(k.shape[1], dtype = tf.float32)


            

    class MultiHeadAttention(tf.keras.Model):        
        def __init__(self, **kwarg):
            super().__init__(self)

            # 하이퍼 파라미터 정의
            self.d_model = kwargs['d_model']
            self.num_heads = kwargs['num_heads']
            self.depth = self.d_model // self.num_heads

            # linear projection 함수 구현
            self.linear_projection = tf.keras.layers.Dense(units = self.d_model)

        # (linearly-projected) embedding_vector를 head갯수로 나누는 함수
        def split_heads(self, x, batch_size):
            
            # d_model을 head 갯수로 나눠 depth크기를 만들기 위해서 나머지 모든 차원들은 특정 값으로 고정되어야 함.
            # 텐서의 축은 (batch_size, sequence_len, d_model) -> (batch_size, num_heads, sequence_len, depth)가 되어야 함
            # 이를 위해서 batch_size라는 파라미터를 사전에 설정하여 고정해줄 필요가 있음.

            x = tf.reshape(x, shape = (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm = [0, 2, 1, 3])        

        def scaled_dot_product_attention(self, q, k, v, mask):
            attention_map = tf.matmul(q, k, transpose_b = True) # (batch_size, sequence_length, sequence_legnth)
            dk = tf.cast(k.shape[1], dtype = tf.float32)
            attention_logits = attention_map / tf.math.sqrt(dk)

            # masking하기
            if mask == True:
                attention_logits += self.subsequent_mask(attention_map)

            attention_weights = tf.nn.softmax(attention_logits, axis = 1)

            output = tf.matmul(attention_weights, v)
            return output

        def call(self, query, key, value):
            # 임베딩 벡터를 linear projection 해주기
            Q = self.linear_projection(query)
            K = self.linear_projection(key)
            V = self.linear_projection(value)

            # linearly projected된 임베딩 벡터를 멀티헤드로 쪼개주기
            q = self.split_heads(Q, Q.shape[0])
            k = self.split_heads(K, K.shape[0])
            v = self.split_heads(V, V.shape[0])

            # scaled_dot_product_attention 적용해주기
            

    q = tf.reshape(Q, shape = (batch_size, -1, num_heads, depth))
    q = tf.transpose(q, perm = [0, 2, 1, 3])
    a_map = tf.matmul(q, q, transpose_b = True)        

# %%
# 포지션 인코딩 마스크 플롯
plt.pcolormesh(tf.transpose(tf.squeeze(pos_encoding)), cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()
