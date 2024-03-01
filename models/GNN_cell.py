import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, max_pool,GCNConv


class GNN_cell(torch.nn.Module):
    def __init__(self, num_feature, layer_cell, dim_cell, cluster_predefine):
        super().__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.cluster_predefine = cluster_predefine
        self.final_node = len(self.cluster_predefine[self.layer_cell - 1].unique())
        self.convs_cell = torch.nn.ModuleList()
        self.bns_cell = torch.nn.ModuleList()
        # self.activations = torch.nn.ModuleList()

        for i in range(self.layer_cell):
            if i:
                conv = GATConv(self.dim_cell, self.dim_cell) # 原来的
                # conv = GCNConv(self.dim_cell, self.dim_cell)  # 新的
            else:
                conv = GATConv(self.num_feature, self.dim_cell) # 原来的
                # conv = GCNConv(self.num_feature, self.dim_cell)  # # 新的

            bn = torch.nn.BatchNorm1d(self.dim_cell, affine=False)  # True or False
            # activation = nn.PReLU(self.dim_cell)

            self.convs_cell.append(conv)


            self.bns_cell.append(bn)
            """
               print("a_self.convs_cell: ", self.convs_cell)
                       i=1
                       ModuleList(
                             (0): GATConv(3, 8, heads=1)
                           )
                       i=2
                        ModuleList(
                           (0): GATConv(3, 8, heads=1)
                           (1): GATConv(8, 8, heads=1)
                           )
                       i=3
                       ModuleList(
                             (0): GATConv(3, 8, heads=1)
                             (1): GATConv(8, 8, heads=1)
                             (2): GATConv(8, 8, heads=1)
                           )        
            """
            # self.activations.append(activation)

    def forward(self, cell):
        for i in range(self.layer_cell):
            # print("self.layer_cell: ",self.layer_cell) # 3
            # print("1_self.convs_cell: ", self.convs_cell) #  ModuleList((0): GATConv(3, 8, heads=1) (1): GATConv(8, 8, heads=1)(2): GATConv(8, 8, heads=1))
            # print("i: ",i)
            #
            # print("1_cell.x.size(): ", cell.x.size())                   # epoch=128: i:0 torch.Size([90368, 3])  128*706=90368    # i:1 torch.Size([68224, 8])  # 2: torch.Size([56064, 8])
            # print("cell.edge_index.size(): ", cell.edge_index.size())   # epoch=128: i:0 torch.Size([2, 403200]) 128*3150=403200  # i:1 torch.Size([2, 270336])  # 2: torch.Size([2, 193024])
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            # print("2_cell.x.size(): ", cell.x.size())         # 0: torch.Size([90368, 8])  1: torch.Size([68224, 8])  2: torch.Size([56064, 8])
            # print("cell.x.size(0): ", cell.x.size(0))         # 0: 90368  1: 68224  2: 56064
            # print("cell.num_graphs: ", cell.num_graphs)       # 128
            num_node = int(cell.x.size(0) / cell.num_graphs)
            # print("num_node: ",num_node)                      # 0: 706  1: 533  2:438
            # print("self.cluster_predefine: ",self.cluster_predefine)
            # print("type(self.cluster_predefine): ", type(self.cluster_predefine))    # <class 'dict'>
            # print("self.cluster_predefine.keys(): ", self.cluster_predefine.keys())  #  dict_keys([0, 1, 2, 3, 4])
            # print("self.cluster_predefine[0].size(): ", self.cluster_predefine[0].size())  # torch.Size([706])
            # print("self.cluster_predefine[1].size(): ", self.cluster_predefine[1].size())  # torch.Size([533])
            # print("self.cluster_predefine[2].size(): ", self.cluster_predefine[2].size())  # torch.Size([438])
            # print("self.cluster_predefine[3].size(): ", self.cluster_predefine[3].size())  # torch.Size([382])
            # print("self.cluster_predefine[4].size(): ", self.cluster_predefine[4].size())  # torch.Size([345])
            # print("len([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)]): ",len([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])) # len=128=cell.num_graphs
            # print("[self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)][0].size(): ",
            #       [self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)][0].size())     # i:0 torch.Size([706])  i:1 torch.Size([533])  i:2 torch.Size([438])
            # print("[self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)][1].size(): ",
            #       [self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)][1].size())     # i:0 torch.Size([706])  i:1 torch.Size([533])  i:2 torch.Size([438])
            # print("[self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)][127].size(): ",
            #       [self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)][127].size())   # i:0 torch.Size([706])  i:1 torch.Size([533])  i:2 torch.Size([438])

            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            # print("cluster.size(): ",cluster.size())  # i:0 torch.Size([90368])  1: torch.Size([68224])   2: torch.Size([56064])
            # print("cluster: ", cluster)  # tensor([0,1,2,  ..., 90365,90324,89734], device='cuda:0')  tensor([0,1,2,.., 68221,68222,68223],device='cuda:0') tensor([0,1,2,  ..., 56061, 55790, 56063], device='cuda:0')
            cell = max_pool(cluster, cell, transform=None)
            # print("cell: ", cell) #  0: Batch(batch=[68224], edge_index=[2, 270336], x=[68224, 8])  1:Batch(batch=[56064], edge_index=[2, 193024], x=[56064, 8]) 2: Batch(batch=[48896], edge_index=[2, 130304], x=[48896, 8])
            cell.x = self.bns_cell[i](cell.x)
            # print("cell.x.size(): ", cell.x.size())  # 0: torch.Size([68224, 8])  1: torch.Size([56064, 8]) 2:  torch.Size([48896, 8])
        # print("cell.x.size(): ",cell.x.size())   # torch.Size([49152, 8])
        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)
        # print("node_representation.size(): ", node_representation.size())    # torch.Size([128, 3072])
        return node_representation
    
    def grad_cam(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            if i == 0:
                cell_node = cell.x
                cell_node.retain_grad()
            num_node = int(cell.x.size(0) / cell.num_graphs)
            cluster = torch.cat([self.cluster_predefine[i] + j * num_node for j in range(cell.num_graphs)])
            cell = max_pool(cluster, cell, transform=None)
            cell.x = self.bns_cell[i](cell.x)

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        return cell_node, node_representation