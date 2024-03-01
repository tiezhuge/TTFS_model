import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, max_pool, GCNConv,global_max_pool


class T_fusion(torch.nn.Module):
    def __init__(self ):
        super().__init__()
        self.druglayer_3D = nn.Linear(512, 100)
        self.druglayer_2D = nn.Linear(512, 100)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5)),
            nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
            nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(22 * 22 * 8, 100), nn.BatchNorm1d(100),
                                 nn.ReLU(True))
        self.fc2_global = nn.Sequential(nn.Linear(100 * 100 + 100, 100), nn.ReLU(True))
        self.fc2_global_reverse = nn.Sequential(nn.Linear(100 * 100 + 100, 100), nn.ReLU(True))
        self.activate = nn.ReLU()
        self.multi_drug = nn.Sequential(nn.Linear(100, 100))

    def forward(self, drug_3D,drug_2D):
        drug_repre_3D = self.druglayer_3D(drug_3D)
        drug_repre_2D = self.druglayer_2D(drug_2D)
        drug_repre_3D_embed_reshape = drug_repre_3D.unsqueeze(-1)
        drug_repre_2D_embed_reshape = drug_repre_2D.unsqueeze(-1)
        drug_repre_2D_matrix = drug_repre_3D_embed_reshape * drug_repre_2D_embed_reshape.permute((0, 2, 1))
        drug_repre_2D_reverse = drug_repre_2D_embed_reshape * drug_repre_3D_embed_reshape.permute((0, 2, 1))
        drug_repre_2D_global = drug_repre_2D_matrix.view(drug_repre_2D_matrix.size(0), -1)
        drug_repre_2D_global_reverse = drug_repre_2D_reverse.view(drug_repre_2D_matrix.size(0), -1)
        drug_repre_2D_reshape = drug_repre_2D_matrix.unsqueeze(1)
        drug_repre_2D_data = drug_repre_2D_reshape
        out = self.conv1(drug_repre_2D_data)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        embedding_data = out
        global_local_before = torch.cat((embedding_data, drug_repre_2D_global), 1)
        cross_embedding_pre = self.fc2_global(global_local_before)
        entity_matrix_reshape_reverse = drug_repre_2D_reverse.unsqueeze(1)
        entity_reverse = entity_matrix_reshape_reverse
        out = self.conv1(entity_reverse)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        embedding_data_reverse = out
        global_local_before_reverse = torch.cat((embedding_data_reverse, drug_repre_2D_global_reverse), 1)
        cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)
        out3 = self.activate(self.multi_drug(drug_repre_3D * drug_repre_2D))


        out_2D_3D_concat = torch.cat((drug_3D, drug_2D, cross_embedding_pre, cross_embedding_pre_reverse, out3), 1)
        return out_2D_3D_concat

