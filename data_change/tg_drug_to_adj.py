
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import time
import argparse
import fitlog
import torch_geometric
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
import scipy

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from scipy.sparse import coo_matrix
def main():
    drug_dict = np.load \
        ('/home/ubuntu/my_code/TGSA/data/Drugs/drug_feature_graph.npy'
        ,allow_pickle=True).item()

    drug_dict_keys = list(drug_dict.keys())
    print(drug_dict_keys)

    drug_3_dict_all_form_tgsa_graph ={}
    for drug in drug_dict_keys:

        print("drug_dict[drug].num_edges: " ,drug_dict[drug].num_edges)   # 52  74   56
        print("drug_dict[drug].num_nodes: ", drug_dict[drug].num_nodes)  # 23   32   26
        compound_adj_matrix = coo_matrix(
            (np.ones(drug_dict[drug].num_edges), (drug_dict[drug].edge_index[0].numpy(), drug_dict[drug].edge_index[1].numpy())),
            shape=(drug_dict[drug].num_nodes, drug_dict[drug].num_nodes)).toarray()
        print("type(compound_adj_matrix): ",type(compound_adj_matrix))  # <class 'numpy.ndarray'>
        print("compound_adj_matrix.shape[0]: ",compound_adj_matrix.shape[0])
        print("compound_adj_matrix.shape: ", compound_adj_matrix.shape)


        compound_node_features =drug_dict[drug].x.numpy()
        print("compound_node_features.shape: ",compound_node_features.shape)   # (23, 77)

        # cell_info ={'name' :cell ,'cell_feat' :cell_feat ,'cell_adj_duijiao' :cell_adj_duijiao}
        drug_info = {'name': drug,'compound_node_features' :compound_node_features ,'compound_adj_matrix' :compound_adj_matrix}
# drug_info ={'name' :drug_name ,'smile':uni_mol_feature_dict[drug_name]['smile'],'compound_node_features' :compound_node_features ,'compound_adj_matrix' :compound_adj_matrix}
        drug_3_dict_all_form_tgsa_graph[drug] =drug_info
    print("drug_3_dict_all_form_tgsa_graph: " ,drug_3_dict_all_form_tgsa_graph)
    # np.save("/home/ubuntu/my_code/TGSA/data/drug_3_dict_all_form_tgsa_graph.npy",drug_3_dict_all_form_tgsa_graph)

    # print(cell_dict['ACH-000001'].x[0])






if __name__ == "__main__":



    main()

