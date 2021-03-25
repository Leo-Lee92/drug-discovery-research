# %%
from utils import *
# %%
# %%
# 멀티 헤드 어텐션
class MultiHeadAttention(tf.keras.Model):        
    def __init__(self, **kwarg):
        super().__init__(self)

        # 하이퍼 파라미터 정의
        self.d_model = kwargs['d_model']
        self.num_heads = kwargs['num_heads']
        self.depth = self.d_model // self.num_heads

        # linear projection 함수 : embedding to attnetion 
        self.wq = tf.keras.layers.Dense(units = self.d_model)
        self.wk = tf.keras.layers.Dense(units = self.d_model)
        self.wv = tf.keras.layers.Dense(units = self.d_model)

        # linear proejection 함수 : scaled attention to output
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

        # scale 적용
        attention_logits = attention_map / tf.math.sqrt(dk)

        # masking 적용
        if mask is not None:
            attention_logits += mask

        attention_weights = tf.nn.softmax(attention_logits, axis = 3)
        attention_scores = tf.matmul(attention_weights, v)
        return attention_scores, attention_weights

    def call(self, query, key, value, mask):
        # query, key, value는 embedded representation된 문장 데이터들이다.
        # mask는 미리 뽑아서 여기까지 계속 전달해주어야 함.

        # 임베딩 벡터를 linear projection 해주기
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)

        # linearly projected된 임베딩 벡터를 멀티헤드로 쪼개주기
        # q, k, v는 (batch, num_heads, seq_len, depth) 4차원임.
        q = self.split_heads(Q, tf.shape(Q)[0])
        k = self.split_heads(K, tf.shape(K)[0])
        v = self.split_heads(V, tf.shape(V)[0])

        # scaled_dot_product_attention 적용해주기
        scaled_attention_scores, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # self-attention output을 concat해주기
        scaled_attention_scores = tf.transpose(scaled_attention_scores, perm = [0, 2, 1, 3])
        scaled_attention_scores = tf.reshape(scaled_attention_scores, shape = (tf.shape(Q)[0], -1, self.d_model))

        # concat된 output에 linear-proejction 적용
        output = self.linear_projection(scaled_attention_scores)

        return output, attention_weights

# 포지션-와이즈 피드포워드 네트워크
class FeedForwardNetwork(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']

        # linear proejection 함수 : normalized scaled attention to output
        self.linear_projection1 = tf.keras.layers.Dense(units = self.d_ff, activation = 'relu')
        self.linear_projection2 = tf.keras.layers.Dense(units = self.d_model)

    def call(self, x):
        # 여기서 x는 add & norm 레이어를 통과한 값
        output = self.linear_projection1(x)
        output = self.linear_projection2(output)

        return output

# 인코더 레이어
class EncoderLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)

        self.mha = MultiHeadAttention(**kwargs)
        self.ffn = FeedForwardNetwork(**kwargs)
        
        self.dropout1 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])
        self.normalization1 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)

        self.dropout2 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])
        self.normalization2 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)

    def call(self, x, mask):
        mha_outputs, _ = self.mha(x, x, x, mask)
        mha_outputs = self.dropout1(mha_outputs) + x
        out1 = self.normalization1(mha_outputs)

        ffn_outputs = self.ffn(out1)
        ffn_outputs = self.dropout2(ffn_outputs) + out1
        out2 = self.normalization2(ffn_outputs)

        return out2

# 디코더 레이어
class DecoderLayer(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        
        self.mha1 = MultiHeadAttention(**kwargs)
        self.mha2 = MultiHeadAttention(**kwargs)
        self.ffn = FeedForwardNetwork(**kwargs)

        self.dropout1 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])
        self.dropout2 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])
        self.dropout3 = tf.keras.layers.Dropout(rate = kwargs['dropout_rate'])

        self.normalization1 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        self.normalization2 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        self.normalization3 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)

    def call(self, enc_x, x, dec_pad_mask, dec_subseq_mask):
        mha_outputs1, attn_weights1 = self.mha1(x, x, x, dec_subseq_mask)
        mha_outputs1 = self.dropout1(mha_outputs1) + x
        out1 = self.normalization1(mha_outputs1)

        mha_outputs2, attn_weights2 = self.mha2(out1, enc_x, enc_x, dec_pad_mask)
        mha_outputs2 = self.dropout2(mha_outputs2) + out1
        out2 =  self.normalization2(mha_outputs2)

        ffn_outputs = self.ffn(out2)
        ffn_outputs = self.dropout3(ffn_outputs) + out2
        out3 = self.normalization3(ffn_outputs)

        return out3, attn_weights1, attn_weights2


# 인코더 모듈
class Encoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        
        self.d_model = kwargs['d_model']
        self.enc_dict_len = kwargs['input_vocab_size']
        self.num_layers = kwargs['num_layers']
        self.position_encoder = Position_Encoder(**kwargs)

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.enc_dict_len, output_dim = self.d_model, mask_zero = True)        
        self.encoder_layer = EncoderLayer(**kwargs)

        self.stacked_enc_layers = [self.encoder_layer for i in range(self.num_layers)]

    def call(self, inputs, enc_pad_mask):
        embeddings = self.embedding_layer(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, dtype = tf.float32))

        pos = self.position_encoder(embeddings)
        x = embeddings + pos

        for enc_layer in self.stacked_enc_layers:
            x = enc_layer(x, enc_pad_mask)

        return x

# 디코더 모듈
class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        self.d_model = kwargs['d_model']
        self.dec_dict_len = kwargs['target_vocab_size']
        self.num_layers = kwargs['num_layers']

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.dec_dict_len, output_dim = self.d_model, mask_zero = True)
        self.position_encoder = Position_Encoder(**kwargs)
        self.decoder_layer = DecoderLayer(**kwargs)
        
        self.stacked_dec_layers = [self.decoder_layer for i in range(self.num_layers)]


    def call(self, enc_x, outputs, dec_pad_mask, dec_seq_mask):
        attn_weights_dict = {}
        
        embeddings = self.embedding_layer(outputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, dtype = tf.float32))

        pos = self.position_encoder(embeddings)
        x = embeddings + pos

        for i, dec_layer in enumerate(self.stacked_dec_layers):
            x, attn_w1, attn_w2 = dec_layer(enc_x, x, dec_pad_mask, dec_seq_mask)
            attn_weights_dict['decoder_layer{}_attn_block1'.format(i+1)] = attn_w1
            attn_weights_dict['decoder_layer{}_attn_block2'.format(i+1)] = attn_w2

        return x, attn_weights_dict

# 트랜스포머 모델
class Transformer(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, name = kwargs['model_name'])
        
        self.mask_generator = Mask_Generator(**kwargs)
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)
        # self.linear_layer = tf.keras.layers.Dense(units = kwargs['target_vocab_size'], activation = 'softmax')
        # 굳이 여기서 activation = 'softmax' 하지 말고 loss 함수에서 from_logits = True 해주는게 나음.
        # 왜냐하면 from_logtis = True가 training 과정에서 numerical stability를 더 보장하기 떄문이다.
        # 참고 https://stackoverflow.com/questions/57253841/from-logits-true-and-from-logits-false-get-different-training-result-for-tf-loss

        self.linear_layer = tf.keras.layers.Dense(units = kwargs['target_vocab_size'])

    def call(self, x):
        inputs, outputs = x
        enc_pad_mask, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, outputs)
        enc_outputs = self.encoder(inputs, enc_pad_mask)
        dec_outputs, attention_weights = self.decoder(enc_outputs, outputs, dec_pad_mask, dec_subseq_mask)
        final_outputs = self.linear_layer(dec_outputs)
        
        return final_outputs

    # def inference(self, x):
    #     dd
# %%
model = Transformer(**kwargs)
cp = Compile_Params(**kwargs)
model.compile(optimizer = cp.optimizer, loss = cp.loss, metrics = [cp.accuracy])
model.fit([padded_FASTA[:128, ], padded_SMILES[:128, 0:padded_SMILES.shape[1]-1]], padded_SMILES[:128, 1:], batch_size = 32, epochs = 100)
# %%
tf.nn.softmax(model.predict([padded_FASTA[:32, :], padded_SMILES[:32, :padded_SMILES.shape[1] - 1]]), axis = 2)
tf.math.argmax(tf.nn.softmax(model.predict([padded_FASTA[:1, :], padded_SMILES[:1, :padded_SMILES.shape[1] - 1]]), axis = 2)[0, :, :], axis = 1)
# %%
no_train_model = Transformer(**kwargs)
tf.math.argmax(tf.nn.softmax(no_train_model.predict([padded_FASTA[:32, :], padded_SMILES[:32, :padded_SMILES.shape[1] - 1]]), axis = 2)[31, :, :], axis = 1)