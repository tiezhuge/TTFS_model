import numpy as np
import pandas as pd
import os
import csv
import scipy
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def get_genes_graph(genes_path, save_path, method='pearson', thresh=0.95, p_value=False):
    """
    determining adjaceny matrix based on correlation
    :param genes_exp_path:
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    genes_exp_df = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)

    # calculate correlation matrix
    genes_exp_corr = genes_exp_df.corr(method=method)
    genes_exp_corr = genes_exp_corr.apply(lambda x: abs(x))
    n = genes_exp_df.shape[0]

    # binarize
    if p_value == True:
        dist = scipy.stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        thresh = dist.isf(0.05)

    adj = np.where(genes_exp_corr > thresh, 1, 0)
    adj = adj - np.eye(genes_exp_corr.shape[0], dtype=np.int)
    edge_index = np.nonzero(adj)
    np.save(os.path.join(save_path, 'edge_index_{}_{}.npy').format(method, thresh), edge_index)

    return n, edge_index


def ensp_to_hugo_map():
    with open('./data/9606.protein.info.v11.0.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        # print("csv_reader: ",csv_reader)
        ensp_map = {row[0]: row[1] for row in csv_reader if row[0] != ""}
        # print("ensp_map: ",ensp_map)   #                              {'9606.ENSP00000000233':  'ARF5',             '9606.ENSP00000000412':      'M6PR',    '9606.ENSP00000001008': 'FKBP4',
        #                                                         #       protein_external_id	  preferred_name(常用名)    protein_external_id	 preferred_name
        # print("type(ensp_map): ", type(ensp_map))
        # print("len(ensp_map): ", len(ensp_map))     # 19566
    return ensp_map


def hugo_to_ncbi_map():
    with open('./data/enterez_NCBI_to_hugo_gene_symbol_march_2019.txt') as csv_file:
        next(csv_file)  # Skip first line
        csv_reader = csv.reader(csv_file, delimiter='\t')
        hugo_map = {row[0]: int(row[1]) for row in csv_reader if row[1] != ""}

        # print("hugo_map: ", hugo_map)   #             {'A1BG':                  1,                       'A1BG-AS1': 503538, 'A1CF': 29974, 'A2M': 2,
                                                    # Approved symbol	NCBI Gene ID(supplied by NCBI)
        # print("len(ensp_map): ", len(hugo_map))   # 41550

    return hugo_map


def save_cell_graph(genes_path, save_path):
    if not os.path.exists(save_path):     #  创建存储文件夹
        os.makedirs(save_path)
    exp = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)      # gene expression (EXP)
    cn = pd.read_csv(os.path.join(genes_path, 'cn.csv'), index_col=0)        #  拷贝数变异                                                                                                                                                                                       
    mu = pd.read_csv(os.path.join(genes_path, 'mu.csv'), index_col=0)       #  体细胞突变
    # me = pd.read_csv(os.path.join(genes_path, 'me.csv'), index_col=0)
    # print('Miss values：{}，{}，{}, {}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum(),
    #                                         me.isna().sum().sum()))

    index = exp.index
    # print("index(exp.index): ",index)         # Index(['ACH-000001', 'ACH-000002', 'ACH-000006', 'ACH-000007', 'ACH-000008',  第一列
    columns = exp.columns
    # print("columns(exp.columns): ", columns)  # Index(['(10000)', '(10006)', '(10019)', '(100533467)', '(1008)', '(1009)',    第一行
    """
    数据预处理中方法
    fit(): Method calculates the parameters μ and σ and saves them as internal objects.
    解释：简单来说，就是求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性。
    transform(): Method using these calculated parameters apply the transformation to a particular dataset.
    解释：在fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）。
    fit_transform(): joins the fit() and transform() method for transformation of dataset.
    解释：fit_transform是fit和transform的组合，既包括了训练又包含了转换。
    transform()和fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
    fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
    """

    scaler = StandardScaler()         # sklearn.preprocessing 中的  StandardScaler()   # StandardScaler类是处理数据归一化和标准化
    # print("1exp: ",exp)   # [580 rows x 706 columns]
    exp = scaler.fit_transform(exp)   # fit_transform 对数据先进行拟合，然后标准化。   对exp进行scaler.fit_transform
    # print("2exp: ", exp.shape)   # (580, 706)

    cn = scaler.fit_transform(cn)    #                                            对cn进行scaler.fit_transform
    print("cn: ", cn.shape)     # (580, 706)
    # me = scaler.fit_transform(me)

    imp_mean = SimpleImputer()
    exp = imp_mean.fit_transform(exp)    #  缺失值处理
    print("3exp: ", exp.shape)  #  # (580, 706)
    print("mu: ", mu.shape)  # (580, 706)
    exp = pd.DataFrame(exp, index=index, columns=columns)
    cn = pd.DataFrame(cn, index=index, columns=columns)
    mu = pd.DataFrame(mu, index=index, columns=columns)
    # me = pd.DataFrame(me, index=index, columns=columns)
    cell_names = exp.index   # 获得细胞系名称
    # print('Miss values：{}，{}，{}, {}'.format(exp.isna().sum().sum(), cn.isna().sum().sum(), mu.isna().sum().sum(),
    #                                         me.isna().sum().sum()))

    cell_dict = {}     #
    for i in cell_names:
        # cell_dict[i] = Data(x=torch.tensor([exp.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([cn.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = Data(x=torch.tensor([mu.loc[i]], dtype=torch.float).T)
        # print("i: ",i)
        # print("exp.loc[i]: ", exp.loc[i])
        """
        exp.loc[i]:  
        (10000)       -0.066005
        (10006)       -1.341435
        (10019)       -0.972647
        (100533467)   -0.621393
        (1008)        -0.261276
        ...
        Name: ACH-000001, Length: 706, dtype: float64
        """
        # print("cn.loc[i]: ", cn.loc[i])
        # print("mu.loc[i]: ", mu.loc[i])
        cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i]], dtype=torch.float).T)   # Data(x=[706, 3]
        # cell_dict[i] = Data(x=torch.tensor([exp.loc[i], cn.loc[i], mu.loc[i], me.loc[i]], dtype=torch.float).T)
        # cell_dict[i] = [np.array(exp.loc[i], dtype=np.float32), np.array(cn.loc[i], dtype=np.float32),
        #                 np.array(mu.loc[i], dtype=np.float32)]

    np.save(os.path.join(save_path, 'cell_feature_cn_std.npy'), cell_dict)
    print("finish saving cell mut data!")


def get_STRING_graph(genes_path, thresh=0.95):
    save_path = os.path.join(genes_path, 'edge_index_PPI_{}.npy'.format(thresh))
    print(" save_path:",save_path)
    if not os.path.exists(save_path):
        # gene_list
        exp = pd.read_csv(os.path.join(genes_path, 'exp.csv'), index_col=0)   # # gene expression (EXP)
        # print("exp.shape: ",exp.shape)    # (580, 706)
        # print("exp.columns.shape: ", exp.columns.shape)   #  (706,)
        gene_list = exp.columns.to_list()
        # print("gene_list_1:", gene_list)                    # ['(10000)', '(10006)', '(10019)', '(100533467)', '(1008)', '(1009)', '(10142)', '(1015)',
        # print("len(gene_list): ", len(gene_list))  # 706
        gene_list = [int(gene[1:-1]) for gene in gene_list]    # 这一步骤去点了单引号和括号
        # print("len(gene_list): ", len(gene_list))  #  706
        # print("gene_list_2:",gene_list)                    # [10000, 10006, 10019, 100533467, 1008, 1009, 10142, 1015,

        # load STRING
        ensp_map = ensp_to_hugo_map()
        hugo_map = hugo_to_ncbi_map()
        edges = pd.read_csv('./data/9606.protein.links.detailed.v11.0.txt', sep=' ')

        # edge_index
        selected_edges = edges['combined_score'] > (thresh * 1000)
        print("selected_edges:",selected_edges)  #    0           False   1           False   11759453    False
        edge_list = edges[selected_edges][["protein1", "protein2"]].values.tolist()
        # print("edge_list:", edge_list)   #   ['9606.ENSP00000485663', '9606.ENSP00000472985'], ['9606.ENSP00000485663', '9606.ENSP00000270625']] 数值大于950的组合
        # print("len(edge_list): ", len(edge_list))  #  数值大于950的组合数量为  131566
        edge_list = [[ensp_map[edge[0]], ensp_map[edge[1]]] for edge in edge_list if
                     edge[0] in ensp_map.keys() and edge[1] in ensp_map.keys()]
        # print("edge_list1:", edge_list)   #  ['EIF3L', 'RPS19'], ['EIF3L', 'RPS15'], ['EIF3L', 'RPS5'], ['EIF3L', 'RPS11']]

        edge_list = [[hugo_map[edge[0]], hugo_map[edge[1]]] for edge in edge_list if
                     edge[0] in hugo_map.keys() and edge[1] in hugo_map.keys()]
        # print("edge_list2:", edge_list)     # [51386, 6209], [51386, 6193], [51386, 6205]]
        edge_index = []
        import sys
        for i in edge_list:
            if (i[0] in gene_list) & (i[1] in gene_list):   # gene_list 是cell line (580, 706) 中由于706个列所组成的列表
                edge_index.append((gene_list.index(i[0]), gene_list.index(i[1])))   # gene_list.index(i[0])可以得到i[0]所代表的基因比如1015在 gene_list中的第几行（7）
                # print("i:",i)                                           #  [1015, 1500]  [1015, 1499]    # [2767, 5290]
                # print("gene_list.index(i[0]:",gene_list.index(i[0]))    #  7                7            #  202
                # print("gene_list.index(i[1]:", gene_list.index(i[1]))   #  72               71           # 381

                edge_index.append((gene_list.index(i[1]), gene_list.index(i[0])))
        # print("edge_index1:", edge_index)       #[(7, 72), (72, 7), (7, 71), (71, 7), (202, 381), (381, 202),.......(72, 7), (7, 72),(71, 7), (7, 71)
        # print("edge_index1:",len(edge_index))    # 6300
        edge_index = list(set(edge_index))     # set这里可消除重复元素
        # print("edge_index1:", edge_index)
        # print("edge_index1:",len(edge_index))  # 3150
        edge_index = np.array(edge_index, dtype=np.int64).T

        # print("edge_index3:", edge_index)
        # 保存edge_index
        # print(len(gene_list))
        # print(thresh, len(edge_index[0]) / len(gene_list))

        # sys.exit(0)
        np.save(
            os.path.join('./data/CellLines_DepMap/CCLE_580_18281/census_706/', 'edge_index_PPI_{}.npy'.format(thresh)),
            edge_index)
    else:
        print("进入else")
        edge_index = np.load(save_path)
        # print("edge_index: ",edge_index)
        """
        [[ 17 525 298 ... 524 557 265]
        [467 379  71 ... 298 295 532]]
        """
        # print(" edge_index.shape: ", edge_index.shape)  # (2, 3150)
    return edge_index


def get_predefine_cluster(edge_index, save_path, thresh, device):
    save_path = os.path.join(save_path, 'cluster_predefine_PPI_{}.npy'.format(thresh))
    if not os.path.exists(save_path):
        print("进入if not os.path.exists(save_path):")
        g = Data(edge_index=torch.tensor(edge_index, dtype=torch.long), x=torch.zeros(706, 1))  # 使用pyG的Data来建图 https://www.pudn.com/news/62f27d0ff97302478e27a0f5.html
        # edge_index为边特征 (2, 3150)  (17->467,467->17,528->379,379->528)   x为节点特征
        print("g",g)   # Data(edge_index=[2, 3150], x=[706, 1])
        print("[g]", [g])  # [Data(edge_index=[2, 3150], x=[706, 1])]
        g = Batch.from_data_list([g])
        print("g", g)  # Batch(batch=[706], edge_index=[2, 3150], x=[706, 1])
        print("type(g): ",type(g))  # <class 'torch_geometric.data.batch.Batch'>
        cluster_predefine = {}
        for i in range(5):
            cluster = graclus(g.edge_index, None, g.x.size(0))
            print(len(cluster.unique()))
            g = max_pool(cluster, g, transform=None)
            cluster_predefine[i] = cluster
        np.save(save_path, cluster_predefine)
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}
    else:
        cluster_predefine = np.load(save_path, allow_pickle=True).item()
        cluster_predefine = {i: j.to(device) for i, j in cluster_predefine.items()}
        # print("cluster_predefine: ",cluster _predefine)  # {0: tensor([  0,   1,  ....   1: tensor([  0,   1,   2,   3, ....  2: tensor([  0,   1,   2,   3,     3: tensor([  0,   1,   2,   3,   4,   4: tensor([  0,   1,   2,   3,   4,   5
        # print("cluster_predefine.keys(): ", cluster_predefine.keys())   # dict_keys([0, 1, 2, 3, 4])
        # print("cluster_predefine[0]:",type(cluster_predefine[0]))
    return cluster_predefine


if __name__ == '__main__':
    # gene_path = './data/CellLines_DepMap/CCLE_580_18281/census_706'
    # save_path = './data/Test_preprocess_gene'
    # # get_genes_graph(gene_path,save_path, thresh=0.53)
    # save_cell_graph(gene_path, save_path)

    # def get_STRING_graph(genes_path, thresh=0.95):
    gene_path = '/home/ubuntu/my_code/TGSA/data/CellLines_DepMap/CCLE_580_18281/census_706'
    get_STRING_graph(gene_path,thresh=0.95)