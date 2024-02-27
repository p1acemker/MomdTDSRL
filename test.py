import torch
# cuda 10.2
# pytorch 1.12

import torch
from Model.gat import GAT,ValueGAT
from Modules.amid_graph_encoding import read_smile_from_file, create_pytorch_geometric_graph_data_list_from_smiles
from Modules.edge_connect_update import create_edge_connect,update_edge_connect,edge_list_to_peptides
from Modules.generate_mol import genertate_peptides_by_edge_index_list
from Modules.rewards import evaleuate_peptides
from rdkit import Chem
from torch import optim,nn
import numpy as np
from rdkit.Chem import Descriptors
import numpy as np
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.rdMolDescriptors as MolDescriptors
if __name__ == '__main__':
    filename = "Data/acid_1.csv"
    smiles = read_smile_from_file(filename)
    print(smiles)
    print(len(smiles))
    graph_2_vec = torch.from_numpy(create_pytorch_geometric_graph_data_list_from_smiles(smiles))
    s = create_edge_connect(edge_num=20)

    print(GAT(dim_in=128,dim_h=64,dim_out=4)(graph_2_vec,s))