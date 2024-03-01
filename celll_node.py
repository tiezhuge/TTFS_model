import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from utils import load_data
from utils import EarlyStopping, set_random_seed
from utils import train, validate
from preprocess_gene import get_STRING_graph, get_predefine_cluster
from models.TGDRP import TGDRP
import time
import argparse
import fitlog



cell_3_dict_all_dict=np.load("/home/kk/my_code/TGSA-master/data/cell_3_dict_all.npy", allow_pickle=True).item()
cell_3_dict_all_dict_keys=cell_3_dict_all_dict.keys()
print("cell_3_dict_all_dict_keys: ",len(cell_3_dict_all_dict_keys))

cell_node3_dict={}
for keys in cell_3_dict_all_dict_keys:
    # cell_info = {'name': cell, 'cell_feat': cell_feat, 'cell_adj_duijiao': cell_adj_duijiao}

    # cell_3_dict_all[cell] = cell_info
    # cell_feat_list = [cell_3_dict['cell_feat'] for cell_3_dict in cell_3_dict_all]
    # cell_adj_list = [cell_3_dict['cell_adj'] for cell_3_dict in cell_3_dict_all]

    # cell_info = {'name': cell, 'cell_feat': cell_feat, 'cell_adj_duijiao': cell_adj_duijiao}
    #
    # cell_3_dict_all[cell] = cell_info
    # print(cell_3_dict_all_dict[keys]['name'])

    cell_info = {'name': cell_3_dict_all_dict[keys]['name'], 'cell_feat': cell_3_dict_all_dict[keys]['cell_feat']}
    cell_node3_dict[keys]=cell_info
np.save("/home/kk/my_code/TGSA-master/data/cell_node3_dict.npy",cell_node3_dict)