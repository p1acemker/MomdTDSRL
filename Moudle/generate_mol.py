import numpy as np
import pandas as pd
# from Cython import inline
from PIL.ImageDraw2 import Draw
from numpy import array
from rdkit import Chem
from rdkit import RDConfig
import unittest
import random
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDepictor, MolFromSmarts
from pandas import read_excel
from rdkit.Chem.Draw import rdMolDraw2D, MolDraw2DCairo
from rdkit import Geometry
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
# from IPython.display import SVG, display
# import seaborn as sns

# sns.set(color_codes=True)
from rdkit.Chem import rdChemReactions
from rdkit import Chem

# C[C@H](C1=NC(C([O-])=O)=C(O1)C)[NH3+]
# CC[C@@H]([C@H]([NH3+])C(C([O-])=O)=O)C


smi1 = "CC[C@H](C)[C@H](NC(=O)c1nc([C@@H](C)N)oc1C)C(=O)C(=O)O"
smi2 = "CC[C@@H]([C@H]([NH2])C(C([OH])=O)=O)C"
rxn = rdChemReactions.ReactionFromSmarts('[C:1](=[O:2])[O].[NH3+:3]>>[C:1](=[O:2])[NH:3]')


def generate_mol_query(smi1, smi2, rxn):
    # 首先判断有多少羧基,多少氨基
    # 第二个判断有多少氨基
    COOH = 'C([O-])=O'
    NH3 = '[N+]'
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    CooH_num = len(mol1.GetSubstructMatches(MolFromSmarts(COOH)))
    NH3_num = len(mol2.GetSubstructMatches(MolFromSmarts(NH3)))
    print(CooH_num, NH3_num)

    # 如果羧基大于一个，氨基大于一个，则进行处理
    if CooH_num > 1:
        for match in smi1.GetSubstructMatches(COOH):
            mol1.GetAtomWithIdx(match[0]).SetProp('_protected', '1')
    if NH3_num > 1:
        for match in smi1.GetSubstructMatches(NH3):
            mol2.GetAtomWithIdx(match[0]).SetProp('_protected', '1')
    reacts = (Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2))
    products = rxn.RunReactants(reacts)
    print(products)
    return Chem.MolToSmiles(products[0][0])
def neutralize_protonated_amides(mol):
    pattern = Chem.MolFromSmarts('[NH+:1][C:2](=[O:3])')
    matches = mol.GetSubstructMatches(pattern)
    for match in matches:
        nitrogen_atom = mol.GetAtomWithIdx(match[0])
        nitrogen_atom.SetFormalCharge(0)
        nitrogen_atom.SetNumExplicitHs(1)
    AllChem.Compute2DCoords(mol)
    Chem.SanitizeMol(mol)
    return mol

def generate_mol(smi1, smi2, rxn):
    # 首先判断有多少羧基,多少氨基
    # 第二个判断有多少氨基
    reacts = (Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2))
    products = rxn.RunReactants(reacts)
    # print(products)
    return Chem.MolToSmiles(products[0][0])

def genertate_long_mol(smiles, rxn):
    #  [217 117 141 221] smiles
    # 1.脱水缩合
    # ac_smile_dict=np.load("Data/dd.npy",allow_pickle=True).item()
    # smiles=[]
    # for ac_smile in index_smiles:
    #     smiles.append(ac_smile_dict[ac_smile])
    # print(smiles)
    result = smiles[0]
    for i, smile in enumerate(smiles):
        # 倘若是终止，则返回当前分子
        if smile == '<End>':
            break
        # 在位置一先跳过
        if i == 0:
            continue
        else:
            # 递推
            result = generate_mol(result, smile, rxn)
            # print(result)
    # 2.去质子，规范形态
    if result!='<End>':
        # print(result)
        mol = neutralize_protonated_amides(Chem.MolFromSmiles(result))
        # for atom in mol.GetAtoms():
        #     if atom.GetTotalNumHs()==3:
        #
        # # H_Idx_list = mol.GetSubstructMatches(Chem.MolFromSmarts('[h]'))
        # selected_H_index_list=np.intersect1d(H_Idx_list,NH3_Idx_list)
        # print(NH3_Idx_list,H_Idx_list)
        # print(NH3_Idx_list)
        # mol.RemoveAtom(selected_H_index_list[0])
        # Chem.RemoveHs(mol)
        return neutralize_protonated_amides(mol),Chem.MolToSmiles(mol)



def index_genertate_long_mol(index_smiles, rxn):
    #  [217 117 141 221] smiles
    # 1.脱水缩合
    # ac_smile_dict=np.load("Data/dd.npy",allow_pickle=True).item()
    # smiles=[]
    # for ac_smile in index_smiles:
    #     smiles.append(ac_smile_dict[ac_smile])
    # print(smiles)
    result = index_smiles[0]
    for i, smile in enumerate(index_smiles):
        # 倘若是终止，则返回当前分子
        if smile == '<End>':
            break
        # 在位置一先跳过
        if i == 0:
            continue
        else:
            # 递推
            result = generate_mol(result, smile, rxn)
            # print(result)
    # 2.去质子，规范形态
    if result!='<End>':
        # print(result)
        mol = Chem.MolFromSmiles('CC[C@H](C)[C@@H]([NH+]C(=O)C[C@H](O)[C@@H]([NH3+])CC(C)C)[C@@H](O)CC(=O)[NH+]1C=C(Br)C=C1C(=O)[NH](O)CCC[C@H]([NH3+])C(=O)[O-]')
        nIdx_list = mol.GetSubstructMatches(Chem.MolFromSmarts('N'))
        # print(nIdx_list)
        for nIdex in nIdx_list:
            mol.GetAtomWithIdx(*nIdex).SetFormalCharge(0)
        OIdx_list = mol.GetSubstructMatches(Chem.MolFromSmarts('O'))
        # print(OIdx_list)
        for OIdex in OIdx_list:
            mol.GetAtomWithIdx(*OIdex).SetFormalCharge(0)
        NH3_Idx_list = mol.GetSubstructMatches(Chem.MolFromSmarts('[NH3]'))
        for Nh3_index in NH3_Idx_list:
            mol.GetAtomWithIdx(*Nh3_index).SetNumExplicitHs(2)

        # for atom in mol.GetAtoms():
        #     if atom.GetTotalNumHs()==3:
        #
        # # H_Idx_list = mol.GetSubstructMatches(Chem.MolFromSmarts('[h]'))
        # selected_H_index_list=np.intersect1d(H_Idx_list,NH3_Idx_list)
        # print(NH3_Idx_list,H_Idx_list)
        # print(NH3_Idx_list)
        # mol.RemoveAtom(selected_H_index_list[0])
        # Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)
    else:
        return 'End'

def hydrolysis_reation(smi):
    rxn = rdChemReactions.ReactionFromSmarts('[C:1](=[O:2])[N:3].[H:4][OH:5]>>[C:1](=[O:2])[OH:5].[H:4][N:3]')
    reacts = Chem.MolFromSmiles(smi)
    products = rxn.RunReactants(reacts)
    print(products)
    # return Chem.MolToSmiles(products[0][0])
# mol->[10,10,10,10]
def decode(mol,decodings,rxn):
    smile_lib=[]
    smiles=[]
    for key, value in decodings.items():
        # print(key)
        smile_lib.append(value)
    for smile_id in list(np.array(mol).flatten()):
        # print(smile_id)
        smile_id=int(smile_id)
        smiles.append(smile_lib[smile_id])
    try:
        result = genertate_long_mol(smiles, rxn)
        return result
    except:
        return False


def genertate_peptides_by_edge_index_list(index_list,ac_smile_dict):
    smile_lib = []
    for key, value in ac_smile_dict.items():
        smile_lib.append(value)
    for i, line in enumerate(index_list):
        result = False
        smile=False
        try:
            smiles = array(smile_lib)[line]
            result,smile = genertate_long_mol(smiles, rxn)
        except:
            print("generate_error")
        return result, smile


if __name__ == '__main__':
    # rxn = rdChemReactions.ReactionFromSmarts('[C:1](=[O:2])[O].[N:3]>>[C:1](=[O:2])[NH1:3]')
    # result=genertate_long_mol([145,216,48,141],rxn)
    # print(result)
    # generate_mol('C[C@H]([NH3+])C1=[NH](C(=O)[C@@H]2CCCN[NH2+]2)[C@H](C(=O)[O-])CS1','O=C([C@H](CCCN)[NH3+])[O-]',rxn)
    # 生成多肽
    # smile1 = "C[C@H](C1=NC(C([O-])=O)=C(O1)C)[NH3+]"
    # smile2 = "CC[C@@H]([C@H]([NH3+])C(C([O-])=O)=O)C"
    # End = '<End>'
    smile_lib = []
    ac_smile_dict=np.load("../Data/dd.npy",allow_pickle=True).item()
    for key,value in ac_smile_dict.items():
        # print(key)
        smile_lib.append(value)
    index_list=np.random.randint(low=0,high=222, size=[1,4])
    for i,line in enumerate(index_list):
        try:
            print(line)
            print(type(line))
            smiles=array(smile_lib)[line]
            print(smiles)
            result,smile=genertate_long_mol(smiles,rxn)

            print(f"succ{i},{result},{smile}")
        except:
            print("generate_error")

    # rxn = rdChemReactions.ReactionFromSmarts('[C:1](=[O:2])[O].[N:3]>>[C:1](=[O:2])[NH1:3]')
    # inital_index_list=np.random.randint(low=0,high=222, size=[150,4,1])
    # ac_smile_dict=np.load("../Data/dd.npy",allow_pickle=True).item()
    # result_smiles=[]
    # mol_list=[]
    # for i in range(inital_index_list.shape[0]):
    #     try:
    #         mol,result_smile=(decode(inital_index_list[i],ac_smile_dict,rxn))
    #         result_smiles.append(result_smile)
    #         mol_list.append(mol)
    #     except:
    #         print("E")
    # mol_list=list(set(mol_list))
    # result_smiles=list(set(result_smiles))
    # dict={}
    # dict["mol_index"]=mol_list
    # dict["smile_list"]=result_smiles
    # print(len(result_smiles))


    # rxn2 = rdChemReactions.ReactionFromSmarts('[C:1](=[O:2]).[N:3]>>[C:1](=[O:2])[NH1:3]')
    # # result = generate_mol_query(smile1, smile2, rxn2)
    # 水解多肽
    # smi="CC[C@H](C)[C@H](N)C(=O)C(=O)[NH][C@H](C)c1nc(C(O)=O)c(C)o1"
    # hydrolysis_reation(smi)
    # # print the protonated SMILES
    # # 画result
    # result="Cn1c(c(nc1/N=C\\1/C(=O)N(C(=O)N1)C)Cc1ccc(cc1)OC)Cc1cc(c(c(c1)OC)O)OC"
    # print(str(result).strip())
    # mol = Chem.MolFromSmiles(str(result).strip())
    # rdDepictor.Compute2DCoords(mol)
    # # mols.append(mol)
    # Draw.MolToFile(mol, 'mol{}.png'.format(0))
#

# d = MolDraw2DCairo(500, 500)
# d.Draw(rxn)
# d.FinishDrawing()
# d.WriteDrawingText('D.png')
