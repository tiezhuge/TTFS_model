
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
from compound import get_mol2vec_features, get_mol_features
def main():
    uni_mol_feature_dict=np.load("/home/ubuntu/my_code/TGSA/data/mol_repr_all_no_h_170.npy", allow_pickle=True).item()
    # print(cell_dict)
    drug_dict_keys = list(uni_mol_feature_dict.keys())





    drug_3_dict_all ={}
    for drug_name in drug_dict_keys:
        uni_mol_feature_dict[drug_name].keys()
        # print("uni_mol_feature_dict[drug_name].keys(): ",uni_mol_feature_dict[drug_name].keys())
        # print("uni_mol_feature_dict[drug_name]['drug_name']: ", uni_mol_feature_dict[drug_name]['drug_name'])
        # print("uni_mol_feature_dict[drug_name]['smile']: ", uni_mol_feature_dict[drug_name]['smile'])
        compound_node_features, compound_adj_matrix, _ = get_mol_features(uni_mol_feature_dict[drug_name]['smile'],34) # 维度为34
        print("compound_node_features: ", type(compound_node_features))  # <class 'numpy.ndarray'>
        print("compound_node_features.shape: ", compound_node_features.shape)   # (23, 34)
        print("compound_adj_matrix: ", type(compound_adj_matrix)) # <class 'numpy.ndarray'>
        # print("compound_adj_matrix: ", compound_adj_matrix)
        print("compound_adj_matrix.shape: ", compound_adj_matrix.shape)   #  (23, 23)

        drug_info ={'name' :drug_name ,'smile':uni_mol_feature_dict[drug_name]['smile'],'compound_node_features' :compound_node_features ,'compound_adj_matrix' :compound_adj_matrix}

        drug_3_dict_all[drug_name ] =drug_info
    print("cell_3_dict_all: " ,drug_3_dict_all)
    np.save("/home/ubuntu/my_code/TGSA/data/drug_3_dict_all.npy" ,drug_3_dict_all)

    # print(cell_dict['ACH-000001'].x[0])






if __name__ == "__main__":



    main()

