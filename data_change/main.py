


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
        cell_dict = np.load('/home/ubuntu/my_code/TGSA/data/CellLines_DepMap/CCLE_580_18281/census_706/cell_feature_all.npy',allow_pickle=True).item()
        # print(cell_dict)
        cell_dict_keys = list(cell_dict.keys())


        edge_index = np.load('/home/ubuntu/my_code/TGSA/data/CellLines_DepMap/CCLE_580_18281/census_706/edge_index_PPI_0.95.npy')

        cell_3_dict_all={}
        for cell in cell_dict_keys:
            edge_index=torch.tensor(edge_index, dtype=torch.long)
            # print("type(edge_index)_2: ", type(edge_index))
            cell_dict[cell].edge_index = edge_index
            print("cell_dict[cell].num_edges: ",cell_dict[cell].num_edges)
            print("cell_dict[cell].edge_index[0].numpy(): ", cell_dict[cell].edge_index[0].numpy())
            print("cell_dict[cell].edge_index[1].numpy(): ", cell_dict[cell].edge_index[1].numpy())
            print("cell_dict[cell].num_nodes: ", cell_dict[cell].num_nodes)
            cell_adj = coo_matrix(
                        (np.ones(cell_dict[cell].num_edges), (cell_dict[cell].edge_index[0].numpy(), cell_dict[cell].edge_index[1].numpy())),
                        shape=(cell_dict[cell].num_nodes, cell_dict[cell].num_nodes)).toarray()
            # print("type(cell_adj): ",type(cell_adj))  # <class 'numpy.ndarray'>
            # print("cell_adj.shape[0]: ",cell_adj.shape[0])  # 706
            # print("cell_adj: ", cell_adj)
            cell_adj_duijiao=np.array(cell_adj)+np.eye(cell_adj.shape[0])
            # print("cell_adj_duijiao: ", cell_adj_duijiao)
            # print("type(cell_adj_duijiao): ", type(cell_adj_duijiao))
            # print("cell_adj_duijiao.shape[0]: ", cell_adj_duijiao.shape[0])

            cell_feat=cell_dict[cell].x.numpy()
            # print("cell_feat: ",cell_feat)

            cell_info={'name':cell,'cell_feat':cell_feat,'cell_adj_duijiao':cell_adj_duijiao}

            cell_3_dict_all[cell]=cell_info
        print("cell_3_dict_all: ",cell_3_dict_all)
        # np.save("/home/ubuntu/my_code/TGSA/data/cell_3_dict_all_duijiao.npy",cell_3_dict_all)

        # print(cell_dict['ACH-000001'].x[0])


"""

[[-0.06600455  1.0091971   0.        ]
[[-1.2347628  1.0090734  0.       ]
[[-0.00374265  1.1003444   0.        ]

        

cell_3_dict_all: 
 {'ACH-000001': {'name': 'ACH-000001', 'cell_feat': array([[-0.06600455,  1.0091971 ,  0.        ],
ACH-000002': {'name': 'ACH-000002', 'cell_feat': array([[-1.2347628,  1.0090734,  0.       ],
ACH-000006': {'name': 'ACH-000006', 'cell_feat': array([[-0.00374265,  1.1003444 ,  0.        ],
"""




if __name__ == "__main__":



        main()

