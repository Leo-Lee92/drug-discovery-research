# %%
# Run this line only when you want to run whole file at once.
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import copy

from preprocess import sample_data, A_Partition_len_distribution, truncated_A_Partition_len_distribution, B_Partition_len_distribution, truncated_data

def describe_stats(sample_data, A_Partition_len_distribution, B_Partition_len_distribution, task = 'generation'):

    if task == 'generation':
        # sample_data에 존재하는 모든 단백질, 약물 길이 분포 통계량 요약
        A_len_sample_data = pd.concat([pd.Series(np.repeat("protein", len(A_Partition_len_distribution))), A_Partition_len_distribution], axis = 1)
        B_len_sample_data = pd.concat([pd.Series(np.repeat("compound", len(B_Partition_len_distribution))), B_Partition_len_distribution], axis = 1)

        sequence_len_dist_sample_data = pd.concat([A_len_sample_data, B_len_sample_data], axis = 0)
        sequence_len_dist_sample_data.columns = ['sequence', 'length']
        len_stats_df = sequence_len_dist_sample_data.groupby('sequence').describe().astype('int')

        # sample_data 내 A_Partition (e.g., 단백질)과 B_Partition (e.g., 약물)의 unique한 sequence 갯수 요약
        A_Partition_stats = pd.DataFrame(sample_data.iloc[:, 0].describe(include = 'all'))
        A_Partition_stats.columns = ['protein']
        B_Partition_stats = pd.DataFrame(sample_data.iloc[:, 1].describe(include = 'all'))
        B_Partition_stats.columns = ['compound']

        freq_stats_df = pd.concat([A_Partition_stats.T, B_Partition_stats.T], axis = 0)
        total_stats_df = pd.concat([freq_stats_df, len_stats_df], axis = 1)

        print(total_stats_df)


    elif task == 'recommendation':
        # sample_data에 존재하는 모든 단백질, 약물 길이 분포 통계량 요약
        A_len_sample_data = pd.concat([pd.Series(np.repeat("citing_article", len(A_Partition_len_distribution))), A_Partition_len_distribution], axis = 1)
        B_len_sample_data = pd.concat([pd.Series(np.repeat("cited_article", len(B_Partition_len_distribution))), B_Partition_len_distribution], axis = 1)

        sequence_len_dist_sample_data = pd.concat([A_len_sample_data, B_len_sample_data], axis = 0)
        sequence_len_dist_sample_data.columns = ['sequence', 'length']
        stats_df = sequence_len_dist_sample_data.groupby('sequence').describe().astype('int')

        # sample_data 내 A_Partition (e.g., 단백질)과 B_Partition (e.g., 약물)의 unique한 sequence 갯수 요약
        A_Partition_stats = pd.DataFrame(sample_data.iloc[:, 0].describe(include = 'all'))
        A_Partition_stats.columns = ['citing_article']
        B_Partition_stats = pd.DataFrame(sample_data.iloc[:, 1].describe(include = 'all'))
        B_Partition_stats.columns = ['cited_article']

        freq_stats_df = pd.concat([A_Partition_stats.T, B_Partition_stats.T], axis = 0)
        total_stats_df = pd.concat([freq_stats_df, len_stats_df], axis = 1)

        print(total_stats_df)

    return total_stats_df

def plot_seq_len(A_Partition_len_distribution, B_Partition_len_distribution, task = 'generation'):

    # 길이 데이터 만들기
    ## protein 길이 데이터 만들기 (시퀀스 대상과 시퀀스 길이를 의미하는 변수로 구성된 2차원 데이터 프레임 만들기)
    A_Partition_len_data = pd.concat([pd.Series(np.repeat("A_Partition", len(A_Partition_len_distribution))), A_Partition_len_distribution], axis = 1)
    ## compound 길이 데이터 만들기 (시퀀스 대상과 시퀀스 길이를 의미하는 변수로 구성된 2차원 데이터 프레임 만들기)
    B_Partition_len_data = pd.concat([pd.Series(np.repeat("B_Partition", len(B_Partition_len_distribution))), B_Partition_len_distribution], axis = 1)

    # 위에서 만든 두 개의 데이터 프레임을 행방향으로 합치기
    sequence_len_distribution = pd.concat([A_Partition_len_data, B_Partition_len_data], axis = 0)
    sequence_len_distribution.columns = ['sequence', 'length']
    sequence_len_distribution.groupby('sequence').describe()

    # 풀롯팅 하기
    ## In case your task is 'generation'
    if task == "generation":
        sns.distplot(sequence_len_distribution[sequence_len_distribution.sequence == "A_Partition"]['length'], label = "protein", kde = False, color = 'tab:red')
        sns.distplot(sequence_len_distribution[sequence_len_distribution.sequence == "B_Partition"]['length'], label = "compound", kde = False, color = 'tab:blue')
        plt.style.use('ggplot')
        plt.ylabel('Frequency')
        plt.ylabel('Sequence length')
        plt.legend(title = 'Sequence', facecolor = 'white')
        # plt.title('Sequence Length Distribution')

        if os.path.isfile('../images/' + str(task) + '/sequence_length_distribution.png') == False:
            plt.savefig('../images/' + str(task) + '/sequence_length_distribution.png')
            plt.show()
        else:
            print('')
            print('NOTE: file {} has already saved !'.format('../images/' + str(task) + '/sequence_length_distribution.png'))
            plt.show()
        plt.clf()

    ## In case your task is 'recommendation'
    elif task == 'recommendation':
        sns.distplot(sequence_len_distribution[sequence_len_distribution.sequence == "A_Partition"]['length'], label = "citing_articles", kde = False, color = 'tab:red')
        sns.distplot(sequence_len_distribution[sequence_len_distribution.sequence == "B_Partition"]['length'], label = "cited_articles", kde = False, color = 'tab:blue')
        plt.style.use('ggplot')
        plt.ylabel('Frequency')
        plt.ylabel('Sequence length')
        plt.legend(title = 'Sequence', facecolor = 'white')
        # plt.title('Sequence Length Distribution')
        
        if os.path.isfile('../images/' + str(task) + '/sequence_length_distribution.png') == False:
            plt.savefig('../images/' + str(task) + '/sequence_length_distribution.png')
            plt.show()
        else:
            print('')
            print('NOTE: file {} has already saved !'.format('../images/' + str(task) + '/sequence_length_distribution.png'))
            plt.show()
        plt.clf()

    return None

# This function plots the frequency of connections.
def plot_connective_frequency(truncated_data, root_partition, leaf_partition, top_n = 10, task = 'generation'):
    '''
    'root_partition' or 'leaf_partition' option must be exclusively either 'A_Partition_Category' or 'B_Partition_Category' argument repsectively, and vice versa.

    For example, let assume that you're handling 'generation' task now.
    Then, if you set 'A_Partition_Category' argument in 'root_partition' option and 'B_Partition_Category' in 'leaf_partition',
    this function plots the number of ligands (i.e., compounds) bounded to each protein.
    '''

    # (1) network matrix (transaction matrix) 만들어주기
    bipartite_connection_matrix = pd.pivot_table(truncated_data, values = 'Connection', index = leaf_partition, columns = root_partition, fill_value = 0)
    # print('bipartite_connection_matrix :', bipartite_connection_matrix)
    print(bipartite_connection_matrix.sum(axis=0).sort_values(ascending = False))

    # (2) 갯수 플로팅 (The number of bound ligands)
    roots_code = bipartite_connection_matrix.sum(axis=0).sort_values(ascending = True).index.astype('str')[-top_n:]
    connective_freqs = bipartite_connection_matrix.sum(axis=0).sort_values(ascending = True)[-top_n:]

    # In case you address 'generation' task,
    if task == 'generation':

        # This setting plots the number of ligands (compounds) bounded to each protein.
        if root_partition == 'A_Partition_Category' and leaf_partition == 'B_Partition_Category':
            plt.style.use('ggplot')
            plt.yticks(fontsize = 300/top_n - 2.5)
            plt.barh(roots_code, connective_freqs, color = 'tab:red', alpha = 0.5)
            plt.ylabel('Protein code (top ' + str(top_n) + ')')
            plt.xlabel('The number of bound ligands')

            if os.path.isfile('../images/' + str(task) + '/num_bound_ligands.png') == False:
                plt.savefig('../images/' + str(task) + '/num_bound_ligands.png')
                plt.show()
            else:
                print('')
                print('NOTE: file {} has already saved !'.format('../images/' + str(task) + '/num_bound_ligands.png'))
                plt.show()
            plt.clf()

        # This setting plots the number of targets (proteins) activated by each compound.
        elif root_partition == 'B_Partition_Category' and leaf_partition == 'A_Partition_Category':
            plt.style.use('ggplot')
            plt.yticks(fontsize = 300/top_n - 2.5)
            plt.barh(roots_code, connective_freqs, color = 'tab:blue', alpha = 0.5)
            plt.ylabel('Compound code (top ' + str(top_n) + ')')
            plt.xlabel('The number of activated targets')

            if os.path.isfile('../images/' + str(task) + '/num_activated_targets.png') == False:
                plt.savefig('../images/' + str(task) + '/num_activated_targets.png')
                plt.show()
            else:
                print('')
                print('NOTE: file {} has already saved !'.format('../images/' + str(task) + '/num_activated_targets.png'))
                plt.show()
            plt.clf()


    # In case you address 'recommendation' task,
    elif task == 'recommendation':

        # This setting plots the number of articles to which each article cite.
        if root_partition == 'A_Partition_Category' and leaf_partition == 'B_Partition_Category':
            plt.style.use('ggplot')
            plt.yticks(fontsize = 300/top_n - 2.5)
            plt.barh(roots_code, connective_freqs, color = 'tab:red', alpha = 0.5)
            plt.ylabel('Citing article code (top ' + str(top_n) + ')')
            plt.xlabel('The number of articles to which each article cites')

            if os.path.isfile('../images/' + str(task) + '/num_articles (to be citing).png') == False:
                plt.savefig('../images/' + str(task) + '/num_articles (to be citing).png')
                plt.show()
            else:
                print('')
                print('NOTE: file {} has already saved !'.format('../images/' + str(task) + '/num_articles (to be citing).png'))
                plt.show()
            plt.clf()

        # This setting plots the number of articles by which each article is cited.
        elif root_partition == 'B_Partition_Category' and leaf_partition == 'A_Partition_Category':
            plt.style.use('ggplot')
            plt.yticks(fontsize = 300/top_n - 2.5)
            plt.barh(roots_code, connective_freqs, color = 'tab:blue', alpha = 0.5)
            plt.ylabel('Cited article code (top ' + str(top_n) + ')')
            plt.xlabel('The number of articles by which each article is cited')

            if os.path.isfile('../images/' + str(task) + '/num_articles (to be cited).png') == False:
                plt.savefig('../images/' + str(task) + '/num_articles (to be cited).png')        
                plt.show()
            else:
                print('')
                print('NOTE: file {} has already saved !'.format('../images/' + str(task) + '/num_articles (to be cited).png'))
                plt.show()
            plt.clf()

    return roots_code, connective_freqs

# This function plots the bipartite network.
def plot_bipartite_network(top_connected_roots, top_n, truncated_data, task = 'generation'):

    # Rearrange in reverse order (i.e., change the list order from ascending to descending)
    top_connected_roots = list(top_connected_roots)
    top_connected_roots.reverse()
    print(top_connected_roots)

    # Take top_n elements from the rearranged list and save it to variable 'target_codes'.
    # That is, 'target_codes' is the reversed list of 'roots_code'
    target_codes = top_connected_roots[:top_n]

    # Initialize arrays of all indices and codes.
    all_idx = np.array([])
    all_code = np.array([])

    # dddddd
    # In case your task is 'generation'
    if task == 'generation':

        # Get the indices that match with elements of 'target_codes' list. 
        for i, val in enumerate(target_codes):
            
            # In case the node is 'Protein' and the leaf is 'Compound'
            if list(top_connected_roots[0])[0] == 'P':
                idx = np.where(np.array(truncated_data['A_Partition_Category']) == val)[0]

            # In case the node is 'Compound' and the leaf is 'Protein'
            elif list(top_connected_roots[0])[0] == 'C':
                idx = np.where(np.array(truncated_data['B_Partition_Category']) == val)[0]

            # Repeat each code of 'target_codes' list as many times as the code appears in the 'truncated_data', and save it into variable 'code'.
            code = np.repeat(val, len(idx))         # 코드 생성

            # Cumulatively append the variable 'code' and 'idx' to the variable 'all_code' and 'all_idx' respectively.
            all_code = np.append(all_code, code)    # 코드의 str값을 담은 array
            all_idx = np.append(all_idx, idx)       # 각 코드에 해당하는 index값을 담은 array

        all_idx = all_idx.astype('int')
        all_idx.sort()
        print(' ')
        print('all unique codes :', np.unique(all_code))
        print('num of all leafs :', len(all_idx))

    # In case your task is 'recommendation'
    elif task == 'recommendation':
        '''
        We will write code for here as soon as possible.
        '''

    # Build a graph dataset using variable 'all_idx' which is the index array of every 'code' listed in 'target_codes'.
    ## That is, 'graph_dataset' is the smaller version of 'truncated_data' sliced with the indices of 'target_codes (roots_codes)'. 
    graph_dataset = truncated_data[['A_Partition_Category', 'B_Partition_Category']].iloc[all_idx]

    # (1) node의 색깔을 결정하는 property dataframe 만들기
    ## (1-1) Get the 'A_Partition_Category' and 'B_Partition_Category' column from 'graph_dataset' respectely, and save each to the variable 'all_A_code' and 'all_B_code'.
    all_A_code = np.array(graph_dataset['A_Partition_Category'])
    all_B_code = np.array(graph_dataset['B_Partition_Category'])

    ## (1-2) Save the length of 'A_Partition_Category' and 'B_Partition_Category' column respectely from 'graph_dataset'.
    len_A_codes = len(np.unique(all_A_code))
    len_B_codes = len(np.unique(all_B_code))

    ## (1-3) Get all unique codes of both A and B Partition and append the codes.
    all_code_ids = np.append(np.unique(all_A_code), np.unique(all_B_code))

    ## (1-4) Define the variable that contains the color value of each code.
    ### Allocate red ('tab:red') to 'A_codes' and blue ('tab:blue') to 'B_codes'.
    all_code_color_values = copy.deepcopy(all_code_ids)
    all_code_color_values[np.arange(len_A_codes)] = 'tab:red'
    all_code_color_values[np.arange(len_A_codes, len(all_code_ids))] = 'tab:blue'

    ## (1-5) Define the variable 'property_dat' which is dataframe having two columns each of which is unique 'code' and corresponding 'color'.
    property_dat = pd.DataFrame([all_code_ids, all_code_color_values]).T
    property_dat.columns = ['code', 'color']

    # (2) 네트워크 그리기
    ## 참고 1: https://www.python-graph-gallery.com/network-chart/
    ## 참고 2: https://networkx.org/documentation/latest/_downloads/networkx_reference.pdf

    ## (2-1) Define graph object 'G' using 'graph_dataset' where the set of bipartite nodes (from_node and to_node) comes from 'A_Partition_Category' and 'B_Partition_Category' respectively.
    G = nx.from_pandas_edgelist(graph_dataset, 'A_Partition_Category', 'B_Partition_Category')

    ## (2-2) Set variable 'property_dat', which is dataframe, to have 'code' as row index and sort the row index by order the nodes of object 'G' is arranged.
    property_dat = property_dat.set_index('code')
    property_dat = property_dat.reindex(np.array(G.nodes()))

    ## (2-3) Draw the network.
    nx.draw(G, with_labels=True, node_size=75, node_color = property_dat['color'], edge_color='black', linewidths=1, font_size=8.5, font_weight = 'bold', pos = graphviz_layout(G, prog = 'neato'), alpha = 0.5)
    red_patch = mpatches.Patch(color='tab:red', label='protein', alpha = 0.5)
    blue_patch = mpatches.Patch(color='tab:blue', label='compound', alpha = 0.5)
    plt.legend(handles=[red_patch, blue_patch], facecolor = 'white', title = "Sequence")
    plt.axis('on')
    plt.title('Bipartite Network (top ' + str(top_n) + ')')
    plt.savefig('../images/' + task + '/protein-compound_network.png')            

    if task == 'generation':
        if os.path.isfile('../images/' + task + '/protein-compound_network.png') == False:
            plt.savefig('../images/' + task + '/protein-compound_network.png')            
            plt.show()
        else:
            print('')
            print('NOTE: file {} has already saved !'.format('../images/' + task + '/protein-compound_network.png'))
            plt.show()
        plt.clf()

    elif task == 'recommendation':
        if os.path.isfile('../images/' + task + '/cite-reference_network.png') == False:
            plt.savefig('../images/' + task + '/cite-reference_network.png')
            plt.show()
        else:
            print('')
            print('NOTE: file {} has already saved !'.format('../images/' + task + '/cite-reference_network.png'))        
            plt.show()
        plt.clf()

    return None

# 데이터 통계량 확인
data_stats = describe_stats(sample_data, A_Partition_len_distribution, B_Partition_len_distribution, task = 'generation')

# 각 파티션 (partition)의 시퀀스 길이 분포 플롯팅하기
plot_seq_len(truncated_A_Partition_len_distribution, B_Partition_len_distribution, task = 'generation')

# 파티션 간 연결 빈도의 분포 플롯
top_connected_A_nodes, num_B_leafs_to_A_node = plot_connective_frequency(truncated_data, 'A_Partition_Category', 'B_Partition_Category', top_n = 30, task = 'generation')   ## A_Partition as root and B_Partition as leaf
top_connected_B_nodes, num_A_leafs_to_B_node = plot_connective_frequency(truncated_data, 'B_Partition_Category', 'A_Partition_Category', top_n = 30, task = 'generation')   ## B_Partition as root and A_Partition as leaf

# 파티션 네트워크 플롯팅
plot_bipartite_network(top_connected_A_nodes, 10, truncated_data)   ## A_Partition as root and B_Partition as leaf
plot_bipartite_network(top_connected_B_nodes, 10, truncated_data)   ## B_Partition as root and A_Partition as leaf