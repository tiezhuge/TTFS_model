from rdkit import Chem
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data
from dgllife.utils import *
import sys

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    8 features are canonical, 2 features are from OGB
    """
    featurizer_funcs = ConcatFeaturizer([atom_type_one_hot,
                                         atom_degree_one_hot,
                                         atom_implicit_valence_one_hot,
                                         atom_formal_charge,
                                         atom_num_radical_electrons,
                                         atom_hybridization_one_hot,
                                         atom_is_aromatic,
                                         atom_total_num_H_one_hot,
                                         atom_is_in_ring,
                                         atom_chirality_type_one_hot,
                                         ])
    atom_feature = featurizer_funcs(atom)
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    featurizer_funcs = ConcatFeaturizer([bond_type_one_hot,
                                         # bond_is_conjugated,
                                         # bond_is_in_ring,
                                         # bond_stereo_one_hot,
                                         ])
    bond_feature = featurizer_funcs(bond)

    return bond_feature


def smiles2graph(mol):
    """
    Converts SMILES string or rdkit's mol object to graph Data object without remove salt
    :input: SMILES string (str)
    :return: graph object
    """

    if isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        mol = Chem.MolFromSmiles(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = Data(x=torch.tensor(x, dtype=torch.float),
                 edge_index=torch.tensor(edge_index, dtype=torch.long),
                 edge_attr=torch.tensor(edge_attr), dtype=torch.float)
    print("graph: ",graph)
    print("type(graph): ", type(graph))  #  <class 'torch_geometric.data.data.Data'>
    return graph


def save_drug_graph():
    smiles = pd.read_csv('./data/IC50_GDSC/drug_smiles.csv')
    print("smiles: ",smiles)
    drug_dict = {}
    for i in range(len(smiles)):
        print("smiles.iloc[i, 0]: ", smiles.iloc[i, 0])   #  MK-8776
        print("smiles.iloc[i, 2]: ",smiles.iloc[i, 2])      # CanonicalSMILES  CN1C=C(C=N1)C2=C3N=C(C(=C(N3N=C2)N)Br)C4CCCNC4
        print("smiles2graph(smiles.iloc[i, 2]): ", smiles2graph(smiles.iloc[i, 2]))   # Data(dtype=torch.float32, edge_attr=[52, 4], edge_index=[2, 52], x=[23, 77])
        drug_dict[smiles.iloc[i, 0]] = smiles2graph(smiles.iloc[i, 2])
    np.save('./drug_feature_graph1.npy', drug_dict)
    return drug_dict


if __name__ == '__main__':
    # graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    # print(graph.x.shape)
    # print(graph.edge_attr.shape)
    # save_drug_graph()

    drug_dict = np.load('./drug_feature_graph1.npy', allow_pickle=True).item()
    print("drug_dict: ", drug_dict)   # {'MK-8776': Data(dtype=torch.float32, edge_attr=[52, 4], edge_index=[2, 52], x=[23, 77]), 'Linsitinib': Data(dtype=torch.float32, edge_attr=[74, 4], edge_index=[2, 74], x=[32, 77]),
    drug_dict_keys = list(drug_dict.keys())
    #
    print("len(drug_dict_keys): ", len(drug_dict_keys))  # len(drug_dict_keys):  170
    print("drug_dict_keys: ", drug_dict_keys)  # 列表中的是药物名称  170 个
    # print("drug_dict['MK-8776']: ",drug_dict['MK-8776'])  #  Data(dtype=torch.float32, edge_attr=[52, 4], edge_index=[2, 52], x=[23, 77])
    # print("drug_dict['Linsitinib']: ",drug_dict['Linsitinib'])  # Data(dtype=torch.float32, edge_attr=[74, 4], edge_index=[2, 74], x=[32, 77])
