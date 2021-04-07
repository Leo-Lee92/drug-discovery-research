# %%
import tensorflow as tf 
import numpy as np
import copy
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
# %%
model_name = 'graph_convolution_transformer'
protein_vocab_size = len(protein_dict) + 1
compound_vocab_size = len(compound_dict) + 1
tar_maxlen = copy.deepcopy(compound_maxlen)
apart_code_len = len(np.unique(trunc_sample_data['BindingDB Target Chain Sequence'].factorize()[0]))
bpart_code_len = len(np.unique(trunc_sample_data['Ligand SMILES'].factorize()[0]))
kwargs = {
    'model_name' : model_name,
    'batch_size' : 32,
    'num_layers' : 3,
    'd_model' : 128,
    'num_heads' : 4,
    'd_ff' : 64,
    'input_vocab_size' : protein_vocab_size,
    'target_vocab_size' : compound_vocab_size,
    'maximum_position_encoding' : tar_maxlen,
    'dropout_rate' : 0.1,
    'end_token_index' : compound_dict['<eos>'],
    'use_bgslp' : True,
    'd_bgc' : 128,
    'pooling' : 'mean',
    'num_hop' : 1,
    'normalized_laplacian' : True
}

# %%
bpg = BipartiteGraphConvolution(**kwargs)

att_score = tf.Variable(np.random.randn(100, 54, 44), dtype = tf.float32)
a, b = bpg(trunc_sample_data['FASTA Category'][320:420], trunc_sample_data['SMILES Category'][320:420], att_score)

slp = SignlessLaplacianPropagation(**kwargs)
slp(trunc_sample_data['FASTA Category'][320:420], trunc_sample_data['SMILES Category'][320:420], att_score)

# tt = list(map(lambda x: np.where(aa == x)[0], aa_unq))

node_batch = trunc_sample_data['SMILES Category'][320:420]
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