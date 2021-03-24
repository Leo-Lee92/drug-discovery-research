# %%
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
# %%
import matplotlib.pyplot as plt
# 포지션 인코딩 마스크 플롯
plt.pcolormesh(tf.transpose(tf.squeeze(pos_encoding)), cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()
