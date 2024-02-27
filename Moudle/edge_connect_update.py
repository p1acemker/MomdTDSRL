# -*- coding: utf-8 -*-
# @Time    : 2023/2/28 11:56
# @Author  : hxt
# @FileName: edge_connect_update.py
# @Software: PyCharm
import numpy as np
import torch
from torch import tensor
from torch import randint
from rdkit import Chem
import random
from .generate_mol import genertate_peptides_by_edge_index_list
from numpy import array
def create_edge_connect(edge_num=219):
    edge_index_list=randint(high=edge_num,size=(1,4))
    edge_index = tensor([list(edge_index_list[0,:3]), list(edge_index_list[0,1:])])
    return edge_index

def update_edge_connect(edge_index,op):
    pos=op//20
    replaced=op%20
    for row in range(2):
        for col in range(3):
            if row+col==pos:
                edge_index[row][col]=replaced
    return edge_index


def edge_list_to_peptides(edge_index):
    l=[]

    dim0, dim1 = edge_index.shape
    for i in range(dim0):
        for j in range(dim1):
            if i==0:
                l.append(int(edge_index[i][j]))
            else:
                if j==2:
                    l.append(int(edge_index[i][j]))
    return array(l).reshape(1, 4)



if __name__ == '__main__':
    update_index_list=create_edge_connect()
    for i in range(100):
        op = random.randint(0, 219 * 4)
        update_index_list = update_edge_connect(update_index_list, op)
        Chem.MolToSmiles(genertate_peptides_by_edge_index_list( edge_list_to_peptides(update_index_list),))
