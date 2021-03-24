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
    'd_ff' : 2048,
    'input_vocab_size' : protein_vocab_size,
    'output_vocab_size' : compound_vocab_size,
    # 'maximum_position_encoding' : MAX_SEQUENCE,
    'rate' : 0.1
}