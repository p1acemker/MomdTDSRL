import sys

from Modules.models_alter_transformer import build_models

sys.path.insert(0, './Modules/')

import numpy as np
import pandas as pd
from Modules.file_reader import read_file
from Modules.mol_utils import get_fragments
from one_hot_encoder import one_hot_encode
# from Modules.models_alter_transformer import build_models
from Modules.training import train
from Modules.generate_mol import genertate_long_mol
from Modules.rewards import clean_good
from rdkit import rdBase
import logging

logging.getLogger().setLevel(logging.INFO)
rdBase.DisableLog('rdApp.error')


def str_to_list(string):
    # [26 154 151 129]->[ 26, 154, 151, 129]
    str_list = []
    for str in string.split('[')[1].split(']')[0].strip().split(' '):
        if str != '':
            str_list.append(int(str))
    return str_list


def read_lead_peptides(path):
    peptides_index_list = []
    peptides_csv = pd.read_csv(path)
    peptides_series = peptides_csv.loc[:, "mol_index"]
    for i, peptide_index in peptides_series.items():
        print(type(peptide_index))
        print(peptide_index)
        peptide_index_list = str_to_list(peptide_index)
        peptides_index_list.append(peptide_index_list)
    return peptides_index_list


def main():
    # 随机生成150个先导的多肽
    # 150*4
    lead_peptide_path = "Data/lead_smile.csv"
    peptides_index_list = read_lead_peptides(lead_peptide_path)
    peptides_index_list = np.array(peptides_index_list).reshape((138, 4, 1))
    # fragment_mols = read_file(fragment_file)
    # print(fragment_mols[0])
    # lead_mols = read_file(lead_file)
    # # fragment将所有分子存入
    # fragment_mols += lead_mols
    # #print(fragment_mols)   rdkit.Chem.rdchem.Mol object at 0x0000013B02F4DE70
    # print("Read {} molecules for fragmentation library", len(fragment_mols))   #logging.info（）  输出日志的信息
    # print("Read {} lead moleculs", len(lead_mols))
    #
    # fragments, used_mols = get_fragments(fragment_mols)
    # print("Num fragments: {}".format(len(fragments)) )
    # print("Total molecules used: {}", len(used_mols))
    # assert len(fragments)
    # assert len(used_mols)
    # mol_path = "Data/"
    #
    # lead_mols = np.asarray(fragment_mols[-len(lead_mols):])[used_mols[-len(lead_mols):]]
    # # #print(lead_mols) <rdkit.Chem.rdchem.Mol object at 0x0000022C01F30490>   47个
    # print(labels_)
    # print(labels_.shape)
    # # logging.info("Building models")
    # print(labels_.shape[1:])
    # 1->[NH3+]c1c(O)cccc1C(=O)[NH+][C@H](CC(=O)O)C(=O)[O-]
    decodings = np.load("Data/dd.npy", allow_pickle=True).item()
    actor, critic = build_models(peptides_index_list.shape[1:])
    print(actor.summary())
    #
    # # #print(X)
    #
    # logging.info("Training")
    history = train(peptides_index_list, actor, critic, decodings)
    #
    # # logging.info("Saving")
    np.save("History/history.npy", history)


if __name__ == "__main__":
    main()
