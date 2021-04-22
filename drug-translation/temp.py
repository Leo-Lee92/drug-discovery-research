# %%
import tensorflow as tf 
import numpy as np
import copy
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

model_name = 'graph_convolution_transformer'
a_partition_vocab_size = len(a_dict) + 1
b_partition_vocab_size = len(b_dict) + 1
tar_maxlen = copy.deepcopy(maxlen_b)
kwargs = {
    'model_name' : model_name,
    'a_dict' : a_dict,
    'b_dict' : b_dict, 
    'batch_size' : 64,
    'num_layers' : 3,
    'd_model' : 64,
    'num_heads' : 4,
    'd_ff' : 32,
    'input_vocab_size' : a_partition_vocab_size,
    'target_vocab_size' : b_partition_vocab_size,
    'maximum_position_encoding' : tar_maxlen,
    'dropout_rate' : 0.1,
    'end_token_index' : b_dict['<eos>'],
    'use_bgslp' : True,
    'd_bgc' : 64,
    'pooling' : 'mean',
    'degree_factor' : 1e+8,
    'num_hop' : 3,
    'normalized_laplacian' : True
}

# %%
mha = MultiHeadAttention(**kwargs)
mask_generator = Mask_Generator(**kwargs)
_, dec_padding_mask, _ = mask_generator(a_sequence[:30, :], b_sequence[:30, 1:])

enc_emb = tf.keras.layers.Embedding(input_dim = a_partition_vocab_size, output_dim = 256)
dec_emb = tf.keras.layers.Embedding(input_dim = b_partition_vocab_size, output_dim = 256)
query = dec_emb(b_sequence[:30, 1:])
key = enc_emb(a_sequence[:30, :])
value = enc_emb(a_sequence[:30, :])

mha(query, key, value, a_code[:30, :], b_code[:30, :], dec_padding_mask)

# %%
att_map = tf.Variable(np.random.randn(30, 54, 44), dtype = tf.float32)
slp = SignlessLaplacianPropagation(**kwargs)
slp(a_code[:30], b_code[:30], att_map)



# %%
# tt = list(map(lambda x: np.where(aa == x)[0], aa_unq))

node_batch = truncated_data['SMILES Category'][320:420]
tf_var = tf.Variable(node_batch)
tf_unq = tf.unique(tf_var)[0]
domain_indicator = tf.strings.substr(tf_unq, pos = 0, len = 1)[0].numpy()
code_indicator = tf.strings.regex_replace(tf_unq.numpy(), domain_indicator, '')
code_num = tf.strings.to_number(code_indicator, out_type = tf.int32)

# %%
a = np.random.randint(0, 2, (7, 5))
b = np.random.randn(5, 2, 2)
c = tf.reshape(tf.reduce_sum(a, axis = 1), shape = (-1, 1))

print('a :', a)
print('c :', c)

d = a * 1 / tf.reshape(tf.reduce_sum(a, axis = 1), shape = (-1, 1))
print('d :', d)
tf.tensordot(d, b, axes = [[1], [0]])