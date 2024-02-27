import os

import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from sklearn.preprocessing import LabelEncoder
from tensorflow import one_hot


def dir_convert_to_index(path):
    convert_dict = {}
    mol_files = os.listdir(path)
    for index, file in enumerate(mol_files):
        convert_dict[index] = file
    return convert_dict


def one_hot_encode(path):
    # 得到文件夹对应的序号
    convert_dict = dir_convert_to_index(path)
    mol_index = convert_dict.keys()
    # mol_files=tf.convert_to_tensor(mol_index)
    print(mol_index)
    print(type(mol_index))
    labels_ = one_hot(list(mol_index), depth=len(mol_index), dtype=tf.int64)
    return labels_, convert_dict


mol_path = "Data/"
labels_, convert_dict = one_hot_encode(mol_path)

# 从action中获得的经过处理后找到对应的mol2文件
def find_index(label, labels):
    for i, v in enumerate(labels):
        print(i,v)
        if all(label == v):
            return convert_dict[i]
if __name__ == '__main__':
    df=pd.read_csv("Data/smile.CSV",names=['Name','smile'])
    # le = LabelEncoder()
    # df['Name']=le.fit_transform(df['Name'])
    print(df.loc[:,'smile'])
    # print(type(df.loc[:,'smile']))
    # dict={}
    for index,smile in df.loc[:,'smile'].items():
        try:
            print(Chem.MolFromSmiles(smile))
            print(f"index:{index},smile:{smile}")
        except:
            print(f"END,index:{index},smile:{smile}")
    print(dict)
    # print(dict)
    # np.save('Data/dd.npy', dict)
    # dict_load = np.load('Data/dd.npy',allow_pickle=True)
    # print(dict_load.item())
    # print(type(dict_load.item()))
    # print(dict_load.item()[0])
    # 先导分子生成
    # lead_labels=np.random.randint(low=0,high=222,size=[150,4])
    # print(lead_labels)



