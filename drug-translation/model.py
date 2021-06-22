# %%
from utils import *
# %%
# %%
# bipartite한 graph 구조에서 convolution 연산을 수행하는 operator
# second multi-head attention block (2nd mha)에 맞물려 작동함
## 2nd mha의 차원은 (batch_size, query_length, key_length)이며, 여기서 query는 docoder input (outputs), key는 encoder input (inputs)임.
# bipartite_graph_convolution (bgc) 연산은 batch_size 축에 대해서 수행되어야 함
class BipartiteGraphConvolution(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        # self.mlb = MultiLabelBinarizer()
        self.pooling = kwargs['pooling']
        self.d_bgc = kwargs['d_bgc']

    # # 근접행렬 생성함수
    # def incidence_matrix_generator(self, node_batch):
    #     # 노드 종류
    #     print('node_batch :', node_batch)
    #     unique_nodes = tf.unique(node_batch)[0]
    #     print('unique_nodes :', unique_nodes)

    #     # 노드 리스트 - 엣지 어레이 (list of arrays 구조)
    #     index_per_unique_nodes = list(map(lambda x : np.concatenate(tf.where(node_batch == x).numpy()), unique_nodes))

    #     # Note. 아래의 코드는 sklearn.preprocessing 라이브러리의 MultilabelBinarizer를 활용 (참조: https://stackoverflow.com/questions/52189126/how-to-elegantly-one-hot-encode-a-series-of-lists-in-pandas)
    #     ## incidence_matrix의 차원 : (num_node, batch_size)
    #     index_per_unique_nodes = pd.Series(index_per_unique_nodes)  
    #     incidence_matrix = pd.DataFrame(self.mlb.fit_transform(index_per_unique_nodes), columns=self.mlb.classes_, index=index_per_unique_nodes.index) 
    #     incidence_matrix = np.array(incidence_matrix)
    #     print('incidence_matrix :', incidence_matrix)

    #     return unique_nodes, incidence_matrix

    # numpy array에서 indexing 하듯이 tensor-graph에도 적용해주는 함수
    def numpy_like_indexing(self, data, target_index, axis = 0):
        # axis = 0이면 row에 대한 indexing, axis = 1이면 col에 대한 indexing

        if axis == 0:
            data = data[target_index, :]
        elif axis == 1:
            data = data[:, target_index]

        return data

    # 근접행렬 생성함수
    def incidence_matrix_generator(self, a_node, b_node):

        ## a_incidence의 차원 : (num_node_in_a, batch_size)
        ## a_target_indices 차원 : (num_node_in_a (activated), ) - 주어진 배치에서 활성화된 노드의 갯수를 row의 갯수로 가짐
        a_incidence = tf.transpose(a_node)
        a_target_indices = tf.squeeze(tf.where(tf.reduce_sum(a_incidence, axis = 1) > 0))
        num_a_nodes_per_batch = tf.shape(a_target_indices)[0]

        ## b_incidence의 차원 : (num_node_in_b, batch_size)
        ## b_target_indices 차원 : (num_node_in_b (activated), )
        b_incidence = tf.transpose(b_node)
        b_target_indices = tf.squeeze(tf.where(tf.reduce_sum(b_incidence, axis = 1) > 0))
        num_b_nodes_per_batch = tf.shape(b_target_indices)[0]

        ## a_incidence_active의 차원 : (num_node_in_a (activated), batch_size)
        ## b_incidence_active의 차원 : (num_node_in_a (activated), batch_size)
        a_incidence_active = tf.numpy_function(self.numpy_like_indexing, (a_incidence, a_target_indices, 0), Tout = tf.float32)
        b_incidence_active = tf.numpy_function(self.numpy_like_indexing, (b_incidence, b_target_indices, 0), Tout = tf.float32)

        ## bipartite_incidence의 차원 : (num_node_in_a (activated) + num_node_in_b (activated), batch_size)
        bipartite_incidence = tf.concat([a_incidence_active, b_incidence_active], axis = 0)

        return bipartite_incidence, num_a_nodes_per_batch, num_b_nodes_per_batch

    def convolution(self, bipartite_incidence, attention_map, pooling):
        # bipartite_incidence = tf.cast(bipartite_incidence, dtype = tf.float32).numpy()
        bipartite_incidence = tf.cast(bipartite_incidence, dtype = tf.float32)

        # Bi-incidence 행렬 생성
        ## average_weighted incidence matrix
        if pooling == "mean":       # average pooling
            num_edges_per_node = tf.reduce_sum(bipartite_incidence, axis = 1)

            bipartite_incidence *= 1 / tf.reshape(num_edges_per_node, shape = (-1, 1))

            # # 만약 bipartite_incidence 행렬에 nan 값이 하나라도 존재한다면
            # if tf.not_equal(tf.size(tf.where(tf.math.is_nan(bipartite_incidence) == True)), 0):
            #     # 1/0 = inf -> nan이 문제가 될 경우 아래와 같이 처리해야 함.
            #     # nan값을 0으로 replace해주기 ㅠㅠ self.degree_factor 활용해도 될 듯. 
            #     # 아니면 그냥 tf.math.nan 해서 True에 해당하는 index는 0, 나머지는 1인 벡터 만들어서 곱해주던가.
            #     nan_indices = tf.where(tf.math.is_nan(bipartite_incidence) == True)
            #     update_values = tf.zeros(tf.shape(tf.squeeze(nan_indices))[0])
            #     bipartite_incidence = tf.tensor_scatter_nd_update(bipartite_incidence, nan_indices, update_values)
            #     print('bipartite_incidence 2:', bipartite_incidence)


        ## max_cut incidence matrix
        ## frobenius norm이 각 attention score 행렬에 담김 총 정보의 양을 대표한다고 간주
        ## frobenuus norm은 n차원 유클리드 평변에 존재하는 벡터의 0점으로부터의 길이 (크기)를 의미
        elif pooling == "max":      # frobenius max pooling
            
            # attention_map 정규화 해주기
            ## attention_map의 차원 : (batch_size, sequence_len1, sequence_len2)
            ## denom의 차원 : (batch_size, sequence_len1)  -- tf.expand_dims() --> (batch_size, sequence_len1, 1)
            ## normalized_attention_map 차원 : (batch_size, sequence_len1, sequence_len2)
            # (1) linear 정규화의 경우
            denom = tf.reduce_sum(attention_map, axis = 2)
            print('denom :', denom)

            normalized_attention_map = attention_map / tf.expand_dims(denom, axis = 2)
            print('normalized_attention_map :', normalized_attention_map)

            # (2) softmax 정규화의 경우 
            # normalized_attention_map = tf.nn.softmax(attention_map)

            # 가장 frobenius norm이 큰 attention score 행렬의 인덱스 확인
            ## frobenius_norm_vec의 차원 : (batch_size, ) -- tf.expand_dims() --> (1, batch_size)
            ## bipartite_incidence의 차원 : (num_node_in_a + num_node_in_b, batch_size)
            ## frobenius_norm_mat의 차원 : (num_node_in_a + num_node_in_b, batch_size)
            frobenius_norm_vec = tf.norm(normalized_attention_map, ord = 'fro', axis = (1, 2))
            print('frobenius_norm_vec :', frobenius_norm_vec)

            frobenius_norm_mat = bipartite_incidence * tf.expand_dims(frobenius_norm_vec, axis = 0)
            print('frobenius_norm_mat :', frobenius_norm_mat)

            ## maxval_edge_idx_vec의 차원 : (num_node_in_a + num_node_in_b, )
            ## maxval_edge_idx_mat 차원 : (num_node_in_a + num_node_in_b, batch_size)
            ## bipartite_incidence 차원 : (num_node_in_a + num_node_in_b, batch_size)
            maxval_edge_idx_vec = tf.argmax(frobenius_norm_mat, axis = 1)
            batch_size = attention_map.shape[0]
            maxval_edge_idx_mat = tf.one_hot(maxval_edge_idx_vec, depth = batch_size)

            bipartite_incidence = maxval_edge_idx_mat

        ## bipartite_incidence 차원 : (num_node_in_a + num_node_in_b, batch_size)
        ## attention_map의 차원 : (batch_size, sequence_len1, sequence_len2)
        ## conv_incidence의 차원 : (num_node_in_a + num_node_in_b, sequence_len1, sequence_len2)
        conv_incidence = tf.tensordot(bipartite_incidence, attention_map, axes = [[1], [0]])
        # print('conv_incidence :', conv_incidence)
        return conv_incidence

    def call(self, a_node, b_node, attention_map):
        # a_node : source domain을 구성하는 nodes set
        # b_node : target domain을 구성하는 nodes set (예. trunc_sample['Ligand SMILES'])
        ## a_node의 차원 : (batch_size, num_node_in_a)
        ## b_node의 차원 : (batch_size, num_node_in_b)

        # # incidence_matrix 생성 ver1
        # a_node = tf.reshape(a_node, shape = [-1])
        # b_node = tf.reshape(b_node, shape = [-1])
        # a_nodeVec, a_incidence = self.incidence_matrix_generator(a_node)
        # b_nodeVec, b_incidence = self.incidence_matrix_generator(b_node)

        # incidence_matrix 이어 붙여서 bipartite_incidence_matrix 생성
        ## bipartite_incidence의 차원 : (num_node_in_a + num_node_in_b, batch_size)
        # bipartite_incidence = tf.concat([a_incidence, b_incidence], axis = 0)

        # incidence_matrix 생성 ver2
        # a_node의 차원 : (num_node_in_a + num_node_in_b, batch_size)
        # b_node의 차원 : (num_node_in_a + num_node_in_b, batch_size)
        bipartite_incidence, num_a_nodes_per_batch, num_b_nodes_per_batch = self.incidence_matrix_generator(a_node, b_node)

        # bipartite_graph_convolution 수행
        out = self.convolution(bipartite_incidence, attention_map, self.pooling)
        # print('out :', out)

        return out, bipartite_incidence, num_a_nodes_per_batch, num_b_nodes_per_batch

class SignlessLaplacianPropagation(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)

        # 이분그래프 컨볼루션 연산자 정의
        self.bgc_operator = BipartiteGraphConvolution(**kwargs)
        self.hop = kwargs['num_hop']
        self.normalized = kwargs['normalized_laplacian']
        self.degree_factor = kwargs['degree_factor']

    def node_square_zero_indexing(self, node_index_array, adjacency_mat):
        node_index_list = list(node_index_array)
        node_square_index_list = [[node_i, node_j] for node_i in node_index_array for node_j in node_index_array]
        zero_tensor = tf.zeros_like(node_square_index_list, dtype = tf.float32)[:, 0]
        zero_indexed_adjacency_mat = tf.tensor_scatter_nd_update(tensor = adjacency_mat, indices = node_square_index_list, updates = zero_tensor)

        return zero_indexed_adjacency_mat

    def make_zero_entry_block(self, node_index_list, adjacency_matrix):
        adjacency_matrix[node_index_list, node_index_list] = 0 
        
        return adjacency_matrix

    def multihop_operator(self, adjacency_mat, hop):
        # (참고) bipartite graph 특성상 hop을 홀수 (odd)로 해야 함. 왜냐하면 짝수 (even) hop일 경우 최종적으로 같은 partition에 속한 node와의 연결을 반영하게 됨.

        # hop 갯수만큼 인접행렬의 내적을 누적시키는 adjacency_mat_op 변수 정의
        adjacency_mat_op = adjacency_mat

        # for문이 (hop - 1) 번 iter 돌면 hop갯수만큼 adjacency를 내적한 셈이됨.
        ## iter = 0에서 adjacency_mat_op과 adjacency_mat가 내적되어 hop = 1를 의미함.
        ## 고로 iter + 1 = hop
        if hop > 1:
            for i in range(hop):
                adjacency_mat_op = tf.tensordot(adjacency_mat_op, adjacency_mat, axes = [[1], [0]])

        return adjacency_mat_op

    # biadjacency 행렬 생성
    def bipartite_adjacency_matrix_generator(self, bi_incidence, num_a_nodes, num_b_nodes, hop):
        # biadjacency 행렬을 만들기 위해선 bi_incidence 행렬을 자기 자신의 전치행렬과 내적한 뒤 특정 block을 0값으로 할당하면 됨.

        # (1) 내적행렬 계산
        ## 내적행렬의 차원 : (num_node_in_a + num_node_in_b, num_node_in_a + num_node_in_b)
        bi_incidence_dot_product = tf.matmul(bi_incidence, bi_incidence, transpose_b = True)

        # (2) batch별로 각 partition에 존재하는 노드의 인덱스 정의
        node1_idx = tf.range(num_a_nodes)
        node2_idx = tf.range(num_a_nodes, num_a_nodes + num_b_nodes)

        # (3) 내적행렬에서 같은 partition에 속한 노드간 관계를 의미하는 블록에 0값 할당
        ## 블록 1의 차원 : (num_node_in_a, num_node_in_a)
        ## 블록 2의 차원 : (num_node_in_b, num_node_in_b)
        adjacency_mat = tf.numpy_function(self.make_zero_entry_block, (node1_idx, bi_incidence_dot_product), Tout = tf.float32)
        adjacency_mat = tf.numpy_function(self.make_zero_entry_block, (node2_idx, adjacency_mat), Tout = tf.float32)

        # adjacency_mat = bi_incidence_dot_product.numpy()
        # adjacency_mat[node1_idx, node1_idx] = 0             # 블록 1
        # adjacency_mat[node2_idx, node2_idx] = 0             # 블록 2

        # (Optional) Multi-hop adjacency matrix 연산
        # if hop > 1:
        #     adjacency_mat = self.multihop_operator(adjacency_mat, hop)
        adjacency_mat = self.multihop_operator(adjacency_mat, hop)

        return tf.cast(adjacency_mat, dtype = tf.float32)

    # bidegree 행렬 생성
    def bipartite_degree_matrix_generator(self, bipartite_adjacency_mat, normalized):

        # degree_vec은 각 요소들이 해당 node의 degree인 벡터
        ## degree_vec의 차원 (num_node_in_a + num_node_in_b, 1)
        degree_vec = tf.reshape(tf.reduce_sum(bipartite_adjacency_mat, axis = 1), shape = (-1, 1))
        degree_vec = tf.cast(degree_vec, dtype = tf.float32)

        # # degree_factor를 활용하여 최소 degree 정의 
        # ## degree = 0 일경우 Normalized에서 inf값으로 발산하는 것을 막기 위함.            
        # ## 만약 degree 값이 0인 노드가 하나라도 존재한다면
        # if tf.not_equal(tf.size(tf.where(tf.squeeze(degree_vec) == 0)), 0):
        #     target_indices = tf.cast(tf.where(tf.squeeze(degree_vec) == 0), dtype = tf.int32)   # degree = 0인 target_indices
        #     update_values = tf.expand_dims(tf.repeat(self.degree_factor, tf.shape(target_indices)[0]), axis = 1)
        #     target_shape = tf.shape(degree_vec)
        #     update_vec = tf.cast(tf.scatter_nd(target_indices, update_values, target_shape), dtype = tf.float32)
        #     degree_vec = degree_vec + update_vec

        # diagonal_band_mat은 bipartite_adjacency_mat과 같은 형상으로 대각축 요소에는 1값, 그 외 요소에는 0값이 입력되어 있는 대칭행렬
        diagonal_band_mat = tf.linalg.band_part(tf.ones_like(bipartite_adjacency_mat), 0, 0)
        diagonal_band_mat = tf.cast(diagonal_band_mat, dtype = tf.float32)

        # 만약 normalized_laplacian이라면
        if normalized == True:
            
            # degree_vec에 1 / sqrt(node_degree) = 1 / sqrt(|each_node_i|)
            degree_vec = tf.math.rsqrt(degree_vec)

        # bipartite_degree_mat은 diagonal_band_mat의 대각요소에 degree_vec의 각 요소값들이 들어간 행렬
        bipartite_degree_mat = diagonal_band_mat * degree_vec
        return bipartite_degree_mat

    def laplacian_propagation(self, bipartite_degree_mat, bipartite_adjacency_mat, bi_incidence, conv_att_map):

        # (1) 이분그래프 라플라시안 (aka Signless Laplacian) 행렬 계산
        ## signless_laplacian_mat의 차원 : (num_node_in_a + num_node_in_b, num_node_in_a + num_node_in_b)

        # (1-1) Normalized Laplacian
        if self.normalized == True:
            signless_laplacian_mat = tf.tensordot(tf.tensordot(bipartite_degree_mat, bipartite_adjacency_mat, axes = [[1], [0]]), bipartite_degree_mat, axes = [[1], [0]])

        # (1-2) Vanilla Laplacian
        else:
            signless_laplacian_mat = bipartite_degree_mat - bipartite_adjacency_mat

        # (2) 무부호 라플라시안 전파 (Signless Laplacian Propagation) 수행
        # (2-1) 라플라시안 전파1
        ## lap_propagated_out1의 차원 : (num_node_in_a + num_node_in_b, sequence_len1, sequence_len2)
        lap_propagated_out1 = tf.tensordot(signless_laplacian_mat, conv_att_map, axes = [[1], [0]])

        # (2-2) 라플라시안 전파2
        ## bi_incidence의 차원 : (num_node_in_a + num_node_in_b, batch_size)
        ## bi_incidence_T의 차원 : (batch_size, num_node_in_a + num_node_in_b)
        ## lap_propagated_out2의 차원 : (batch_size, sequence_len1, sequence_len2)
        bi_incidence_T = tf.cast(tf.transpose(bi_incidence), dtype = tf.float32)
        lap_propagated_out2 = tf.tensordot(bi_incidence_T, lap_propagated_out1, axes = [[1], [0]])

        return lap_propagated_out2, signless_laplacian_mat

    def call(self, a_node, b_node, attention_map):
        # a_node : source domain을 구성하는 nodes set
        # b_node : target domain을 구성하는 nodes set (예. trunc_sample['Ligand SMILES'])
        ## a_node의 차원 : (batch_size, num_node_in_a)
        ## b_node의 차원 : (batch_size, num_node_in_b)

        # (1) 이분그래프 컨볼루션을 통해 node의 임베딩 텐서 및 bi_incidence 행렬 계산
        ## conv_att_map의 차원 : (num_node_in_a + num_node_in_b, sequence_len1, sequence_len2)
        ## bi_incidence의 차원 : (num_node_in_a + num_node_in_b, batch_size)
        ## num_a_nodes_per_batch의 차원 : (batch_size, num_node_in_a)
        ## num_b_nodes_per_batch의 차원 : (batch_size, num_node_in_b)
        conv_att_weights, bi_incidence, num_a_nodes_per_batch, num_b_nodes_per_batch = self.bgc_operator(a_node, b_node, attention_map)

        # (2) 이분그래프 인접행렬 (bipartite_adjacency_mat) 및 이분그래프 차수행렬 (bipartite_degree_mat) 계산
        ## bipartite_adjacency_mat의 차원 : (num_node_in_a + num_node_in_b, num_node_in_a + num_node_in_b)
        ## bipartite_degree_mat의 차원 : (num_node_in_a + num_node_in_b, num_node_in_a + num_node_in_b)
        bipartite_adjacency_mat = self.bipartite_adjacency_matrix_generator(bi_incidence, num_a_nodes_per_batch, num_b_nodes_per_batch, hop = self.hop)    # hop 조절 가능
        bipartite_degree_mat = self.bipartite_degree_matrix_generator(bipartite_adjacency_mat, normalized = self.normalized)     # normalize 여부 선택 가능

        # (3) 라플라시안 행렬 계산 및 전파 수행
        ## singless_laplacian_mat : 라플라시안 행렬 
        # 의 차원 : (num_node_in_a + num_node_in_b, num_node_in_a + num_node_in_b)
        ## laplace_propagated_map : 라플라시안 전파가 수행된 attention_map
        # 의 차원 : (num_node_in_a + num_node_in_b, num_node_in_a + num_node_in_b)
        laplace_propagated_map, signless_laplacian_mat = self.laplacian_propagation(bipartite_degree_mat, bipartite_adjacency_mat, bi_incidence, conv_att_weights)

        return laplace_propagated_map

# 멀티 헤드 어텐션
class MultiHeadAttention(tf.keras.Model):        
    def __init__(self, **kwargs):
        super().__init__(self)

        # 하이퍼 파라미터 정의
        self.d_model = kwargs['d_model']
        self.num_heads = kwargs['num_heads']
        self.depth = self.d_model // self.num_heads
        self.bgslp = kwargs['use_bgslp']
        self.slp = SignlessLaplacianPropagation(**kwargs)

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

    def scaled_dot_product_attention(self, q, k, v, a_node, b_node, mask):
        
        # attention_map의 차원 : (batch_size, num_heads, sequence_len1, sequence_len2)
        attention_map = tf.matmul(q, k, transpose_b = True)

        '''
        여기에 SignlessLaplacianPropagation
        '''
        # 만약 SignlessLaplacianPropagation을 사용한다면 if문 안의 연산을 수행
        if self.bgslp == True:
            # Graph Convolution & Signeless Laplacian Operation 수행
            # a_node과 b_node에 해당하는 값이 여기까지 넘어올 수 있어야 함
            laplace_propagated_map = self.slp(a_node, b_node, attention_map)
            attention_map = laplace_propagated_map

        # scale 적용
        dk = tf.cast(k.shape[1], dtype = tf.float32)
        attention_logits = attention_map / tf.math.sqrt(dk)

        # masking 적용
        if mask is not None:
            attention_logits += mask

        attention_weights = tf.nn.softmax(attention_logits, axis = -1)

        # attention_scores의 차원 : (batch_size, num_heads, sequnece_len1, depth)
        attention_scores = tf.matmul(attention_weights, v)

        return attention_scores, attention_weights

    def call(self, query, key, value, a_node, b_node, mask):
        # query, key, value는 embedded representation된 문장 데이터들이다.
        # mask는 미리 뽑아서 여기까지 계속 전달해주어야 함.

        # 임베딩 벡터를 linear projection 해주기
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)

        # linearly projected된 임베딩 벡터를 멀티헤드로 쪼개주기
        # q, k, v는 (batch_size, num_heads, sequence_len, depth) 4차원임.
        q = self.split_heads(Q, tf.shape(Q)[0])
        k = self.split_heads(K, tf.shape(K)[0])
        v = self.split_heads(V, tf.shape(V)[0])

        # scaled_dot_product_attention 적용해주기
        # scaled_attention_scores의 차원: (batch_size, num_heads, sequence_len1, depth)
        scaled_attention_scores, attention_weights = self.scaled_dot_product_attention(q, k, v, a_node, b_node, mask)

        # self-attention output의 multi head들을 concat해주기
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

    def call(self, x, a_node, mask):
        mha_outputs, _ = self.mha(x, x, x, a_node, a_node, mask)
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

    def call(self, enc_x, x, a_node, b_node, dec_pad_mask, dec_subseq_mask):
        # within-domain self-attention 파트
        mha_outputs1, attn_weights1 = self.mha1(x, x, x, b_node, b_node, dec_subseq_mask)
        mha_outputs1 = self.dropout1(mha_outputs1) + x
        out1 = self.normalization1(mha_outputs1)

        # cross-domain attention 파트
        mha_outputs2, attn_weights2 = self.mha2(out1, enc_x, enc_x, a_node, b_node, dec_pad_mask)
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

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.enc_dict_len, output_dim = self.d_model, mask_zero = False)        
        self.encoder_layer = EncoderLayer(**kwargs)

        self.stacked_enc_layers = [self.encoder_layer for i in range(self.num_layers)]

    def call(self, inputs, a_node, enc_pad_mask):
        embeddings = self.embedding_layer(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, dtype = tf.float32))

        pos = self.position_encoder(embeddings)
        x = embeddings + pos

        for enc_layer in self.stacked_enc_layers:
            x = enc_layer(x, a_node, enc_pad_mask)

        return x

# 디코더 모듈
class Decoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self)
        self.d_model = kwargs['d_model']
        self.dec_dict_len = kwargs['target_vocab_size']
        self.num_layers = kwargs['num_layers']

        self.embedding_layer = tf.keras.layers.Embedding(input_dim = self.dec_dict_len, output_dim = self.d_model, mask_zero = False)
        self.position_encoder = Position_Encoder(**kwargs)
        self.decoder_layer = DecoderLayer(**kwargs)
        
        self.stacked_dec_layers = [self.decoder_layer for i in range(self.num_layers)]


    def call(self, enc_x, outputs, a_node, b_node, dec_pad_mask, dec_seq_mask):
        attn_weights_dict = {}
        
        embeddings = self.embedding_layer(outputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, dtype = tf.float32))

        pos = self.position_encoder(embeddings)
        x = embeddings + pos

        for i, dec_layer in enumerate(self.stacked_dec_layers):
            x, attn_w1, attn_w2 = dec_layer(enc_x, x, a_node, b_node, dec_pad_mask, dec_seq_mask)
            attn_weights_dict['decoder_layer{}_attn_block1'.format(i+1)] = attn_w1
            attn_weights_dict['decoder_layer{}_attn_block2'.format(i+1)] = attn_w2

        return x, attn_weights_dict

# 트랜스포머 모델
class Transformer(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, name = kwargs['model_name'])
        
        self.maxlen = kwargs['maximum_position_encoding']
        self.end_token_idx = kwargs['end_token_index']
        self.a_dict = kwargs['a_dict']
        self.b_dict = kwargs['b_dict']

        self.mask_generator = Mask_Generator(**kwargs)

        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)
        # self.linear_layer = tf.keras.layers.Dense(units = kwargs['target_vocab_size'], activation = 'softmax')
        # 굳이 여기서 activation = 'softmax' 하지 말고 loss 함수에서 from_logits = True 해주는게 나음.
        # 왜냐하면 from_logtis = True가 training 과정에서 numerical stability를 더 보장하기 떄문이다.
        # 참고 https://stackoverflow.com/questions/57253841/from-logits-true-and-from-logits-false-get-different-training-result-for-tf-loss

        self.linear_layer = tf.keras.layers.Dense(units = kwargs['target_vocab_size'])

    def call(self, data):
        '''
        inputs : citing articles / protein sequences (FASTA)
        outputs : cited articles / compound sequences (SMILES)
        a_code : nodes in a partition (node of input domain; a_node)
        b_code : nodes in b partition (node of output domain; b_node)
        '''

        # a_code & b_code의 차원 : (batch_size, num_nodes)
        inputs, a_node, b_node, outputs = data
        
        enc_pad_mask, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, outputs)
        enc_outputs = self.encoder(inputs, a_node, enc_pad_mask)
        dec_outputs, attention_weights = self.decoder(enc_outputs, outputs, a_node, b_node, dec_pad_mask, dec_subseq_mask)
        final_outputs = self.linear_layer(dec_outputs)
        
        return final_outputs

    def inference(self, data):

        inputs, a_node, b_node, outputs = data

        # (1) 시작토큰 ('<bos>')으로 decoder에 활용될 outputs 벡터 생성. 
        ## outputs의 크기는 (batch_size, 1)
        batch_size = inputs.shape[0]
        outputs = tf.expand_dims([self.b_dict['<bos>']] * batch_size, 1)

        # (2) 초기 mask 생성
        ## enc_pad_mask, dec_subseq_mask : 각각 encoder, decoder에 대한 self-attention용 mask
        ## dec_pad_mask : encoder와 decoder의 토큰들 간 attention용 mask
        enc_pad_mask, dec_pad_mask, dec_subseq_mask = self.mask_generator(inputs, outputs)

        # (3) inputs값을 encoder에 통과시킨 임베딩 행렬인 enc_outputs 생성
        enc_outputs = self.encoder(inputs, a_node, enc_pad_mask)
        
        # (4) 예측토큰을 담을 리스트 정의
        ## 시작 토큰 ('<bos>')으로 구성됨
        predict_tokens = outputs

        # ㅇㅇ
        fin_seq_idx = np.array([])


        # (5) 앞서 예측한 토큰을 활용하여 다음 토큰을 예측하는 과정
        ## for문 최대길이는 target 시퀀스의 최대길이
        for t in range(0, self.maxlen):

            # enc_outputs, 이전 iter에서 예측된 outputs, node 행렬, mask 행렬을 decoder의 입력으로 활용하여 현재 iter에서의 임베딩 행렬 dec_outputs 예측
            dec_outputs, _ = self.decoder(enc_outputs, outputs, a_node, b_node, dec_pad_mask, dec_subseq_mask)
            final_outputs = self.linear_layer(dec_outputs)

            # logit이 크면 확률도 크니까 그냥 final_outputs에서 argmax해주기
            ## 여기서 다양한 sequence_generation algorithm을 구현하기
            maxlogit_tokens = tf.argmax(final_outputs, -1).numpy()
            pred_token = tf.cast(tf.expand_dims(maxlogit_tokens[:, -1], 1), dtype = tf.int32)

            # 이미 end_token이 나온 시퀀스의 경우 예측결과와 상관없이 계속 end_token 입력
            if len(fin_seq_idx) != 0:
                pred_token = np.array(pred_token)
                pred_token[fin_seq_idx, :] = self.end_token_idx
                # print('fin_seq_idx : {}'.format(fin_seq_idx))

            # if t % 10 == 0:
            #     print('pred_token : {}'.format(pred_token))
            #     print('tf.math.count_nonzero(tf.math.equal(pred_token, self.end_token_idx)) : {}'.format(tf.math.count_nonzero(tf.math.equal(pred_token, self.end_token_idx))))

            # 만약 예측토큰들이 모두 end_token일 경우 생성을 중단.
            fin_seq_idx = tf.where(tf.math.equal(pred_token, self.end_token_idx))
            if tf.math.count_nonzero(tf.math.equal(pred_token, self.end_token_idx)) >= batch_size:
                break

            # predict_tokens.append(pred_token)
            predict_tokens = tf.concat([predict_tokens, pred_token], axis = 1)
            _, dec_pad_mask, dec_subseq_mask= self.mask_generator(inputs, predict_tokens)
            
            # embedding layer 초기에 한번 통과하면 끝나는 건가?
            outputs = copy.deepcopy(predict_tokens)

        return predict_tokens
        
# %%
model = Transformer(**kwargs)
cp = Compile_Params(**kwargs)
model.compile(optimizer = cp.optimizer, loss = cp.loss, metrics = [cp.accuracy])
# histories = model.fit([padded_FASTA[:30, :], padded_SMILES[:30, :padded_SMILES.shape[1]-1]], padded_SMILES[:30, 1:], batch_size = 1, epochs = 1000)
# histories = model.fit([padded_FASTA[:256 ,:], padded_SMILES[:256, :padded_SMILES.shape[1]-1]], padded_SMILES[:256, 1:], batch_size = 32, epochs = 1000)
histories = model.fit([a_sequence[:640, :], a_code[:640, :], b_code[:640, :], b_sequence[:640, :b_sequence.shape[1]-1]], b_sequence[:640, 1:], batch_size = kwargs['batch_size'], epochs = 1000)

# %%
# inference가 상식적으로 동작하도록 만들어야 함.
# 왜 model.predict는 잘되는데 model.inference는 잘 안될까?
# ttt = model.inference(padded_FASTA)
ttt = model.inference([a_sequence[:64, :], a_code[:64, :], b_code[:64, :], b_sequence[:64, :b_sequence.shape[1]-1]])
np.vectorize(dict(map(reversed, b_dict.items())).get)(ttt.numpy())

# %%
tf.nn.softmax(model.predict([padded_FASTA[:32, :], padded_SMILES[:32, :padded_SMILES.shape[1] - 1]]), axis = 2)
tf.math.argmax(tf.nn.softmax(model.predict([padded_FASTA[:32, :], padded_SMILES[:32, :padded_SMILES.shape[1] - 1]]), axis = 2)[:, :, :], axis = 2)


tf.math.argmax(tf.nn.softmax(model.predict([a_sequence[:100, :], a_code[:100, :], b_code[:100, :], b_sequence[:100, :b_sequence.shape[1]-1]]), axis = 2)), axis = 2)[:, :, :], axis = 2)

# %%
no_train_model = Transformer(**kwargs)
tf.math.argmax(tf.nn.softmax(no_train_model.predict([padded_FASTA[:32, :], padded_SMILES[:32, :padded_SMILES.shape[1] - 1]]), axis = 2)[31, :, :], axis = 1)

tf.expand_dims([compound_dict['<bos>']] + [compound_dict['pad']] * 43, 0)

# %%
# 변환하는 것
tmp = tf.math.argmax(tf.nn.softmax(model.predict([a_sequence[:32, :], b_sequence[:32, :b_sequence.shape[1] - 1]]), axis = 2)[:, :, :], axis = 2)
np.vectorize(a_dict.get)(tmp)



# %%
# 플롯
import matplotlib.pyplot as plt
plot_graphs(histories, 'accuracy')