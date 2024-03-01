import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_max_pool,GATConv,GCNConv

class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()

        for i in range(self.layer_drug):
            if i:
                block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
                block = nn.Sequential(nn.Linear(77, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            # print("block: ",block)
            conv = GINConv(block)  # 原来的
            # print("conv: ", conv)
            # conv = GCNConv(block) #新的

            bn = torch.nn.BatchNorm1d(self.dim_drug)
            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x_drug_list = []
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        # print("node_representation: ",node_representation.size())   # torch.Size([4298, 384])
        # print("batch: ", batch.size())    #torch.Size([4298])
        x_drug = global_max_pool(node_representation, batch)        # 三种图结构数据的预测任务（node-level task,edge-level task,global-level task）
        return x_drug



