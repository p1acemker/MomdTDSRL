import rdkit.Chem as Chem
from numpy import array
from rdkit.Chem import Descriptors
import numpy as np
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.rdMolDescriptors as MolDescriptors
from .generate_mol import genertate_long_mol, rxn,decode
from rdkit.Chem import Descriptors
import os, time
import numpy as np
# import h5py
# from pybel import *
import subprocess
import logging
import _thread
import threading
from .remote_server import sftpClient
# from rdkit.Chem import rdChemReactions
from pandas import DataFrame as df
# Cache evaluated molecules (rewards are only calculated once)

def modify_fragment(f, swap):
    f[-(1 + swap)] = (f[-(1 + swap)] + 1) % 2
    return f
def get_bit(a):
    bit=1
    # 9999
    while(a//10):
        a=a//10
        bit+=1
    return bit
def get_key(fs):
    key=""
    for a in list(np.array(fs).flatten()):
        bit=3-get_bit(a)
        key+=bit*"0"+str(a)
    return key
def Client(filename):
    result_list_str = sftpClient(
        _ligandListFilePath=filename,
        _remotepath='/lustre/home/weizhiqiang/',
        _hostname='10.130.2.222',
        _username='weizhiqiang',
        _passwd='IAOS1234'
    )
    return result_list_str
import numpy as np

def compute_reward(docking_score, threshold=-8, bonus=10, penalty=-40, alpha=2):
    base_reward = np.power(docking_score, alpha)

    if docking_score < threshold:
        threshold_reward = bonus
    else:
        threshold_reward = penalty

    reward = base_reward + threshold_reward
    return reward




def evaleuate_peptides(mol,epoch,count):
    global evaluated_molelues
    evaluated_molelues={}
    try:
        key = str(mol).split("<rdkit.Chem.rdchem.Mol object at")[1].split(">")[0]
    except:
        return -2
    if key in evaluated_molelues:
        print("rewards : this is if----")
        reward=evaluated_molelues[key]
    else:
        rootPath = "/home/b519/wq/rlg_1"
        path = rootPath + "/ledock/ligands/" + str(epoch)+"/"+str(count) + "/"
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        filename = path + 'ligands'

        try:
            Chem.GetSSSR(mol)
            clogp = Crippen.MolLogP(mol)
            mw = MolDescriptors.CalcExactMolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            # print(f"smi:{Chem.MolToSmiles(mol)},lpgp:{clogp},tpsa:{tpsa},mw:{mw}")
            Smiles=Chem.MolToSmiles(mol)
            #     启动对接
            os.system('obabel -:"' + Smiles + '" --gen3d -omol2 -O ' + path + str(count) + '.mol2')
            with open(filename, 'a') as file_object:
                file_object.write(path + str(count) + ".mol2\n")
            result_list_str = Client(filename)
            while True:
                if (result_list_str == None):
                    # 当result_list_str为None的时候，走if
                    time.sleep(6)  # 等待6s
                    # 日志输出
                    _now_time = time.time()
                    logging.info('=' * 100)
                    logging.info(str((time.strftime('%Y-%m-%d %H:%M:%S'))))
                    logging.info('The result_list_str is None')
                    logging.info('=' * 100)
                    # 重新执行
                    result_list_str = Client(filename)
                else:
                    # 如果result_list_str不为None的话，就尝试
                    try:
                        result_list = result_list_str.strip('[]').split('], [')
                        break
                    except BaseException as e:
                        time.sleep(6)  # 等待6s
                        # 日志输出
                        _now_time = time.time()
                        logging.info('=' * 100)
                        logging.info(str((time.strftime('%Y-%m-%d %H:%M:%S'))))
                        logging.info('The interrupted service has been restarted')
                        logging.error(e)
                        logging.info('=' * 100)
                        # 重新执行
                        result_list_str = Client(filename)
            print(result_list)
            for result in result_list:
                strs = result.split(', ')
                value = float(strs[2])
                print(value)
            reward=compute_reward(value)
        except:
            reward=-10
        evaluated_molelues[key]=reward
    return reward

# **# 批量处理分子奖励
def evaluate_batch_mol(org_mols, batch_mol, epoch, decodings):
    fr = [False] * FEATURES
    rootPath = "/home/developer/wq/rlg_1"
    frs = [fr] * batch_mol.shape[0]
    global evaluated_mols
    path = rootPath + "/ledock/ligands/" + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + "/"
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        # os.system('sudo mkdir '+path)
    filename = path + 'ligands'
    Times_result_dir = "/home/developer/wq/rlg_1/Times/" + str(TIMES) + "/"
    Inital_smi_list = []
    Modified_smi_list = []
    dict_smi = {}
    if not os.path.exists(Times_result_dir):
        os.mkdir(Times_result_dir)
    if epoch == -1:
        for i in range(batch_mol.shape[0]):
            try:
                mol = decode(batch_mol[i], decodings, rxn)
                Inital_smi_list.append(str("E"))
                Modified_smi_list.append(str((mol)))
            except:
                Inital_smi_list.append("E")
                Modified_smi_list.append("error")
    else:
        for i in range(batch_mol.shape[0]):
            try:
                ori_mol = decode(org_mols[i], decodings,rxn=rxn)
                mol = decode(batch_mol[i], decodings,rxn)
                Inital_smi_list.append(Chem.MolToSmiles(ori_mol))
                Modified_smi_list.append(Chem.MolToSmiles((mol)))
            except:
                ori_mol = decode(org_mols[i], decodings,rxn)
                Inital_smi_list.append(Chem.MolToSmiles(ori_mol))
                Modified_smi_list.append("error")
    dict_smi = {}
    dict_smi["Inital_smi" ]=Inital_smi_list
    dict_smi["Modified_smi"]=Modified_smi_list
    df(dict_smi).to_csv(Times_result_dir+str(epoch)+".csv")
    # print(filename)
    # 构建需要计算的小分子的列表
    mollist = []
    keylist = []
    for i in range(batch_mol.shape[0]):
        # 150
        if not np.all(org_mols[i] == batch_mol[i]):
            # print("i=" + str(i))
            key = get_key(batch_mol[i])
            if key in evaluated_mols:
                print("rewards : this is if----")
                frs[i] = evaluated_mols[key][0]
                continue
            try:
                # mol = decode(batch_mol[i], decodings)
                mol=decode(batch_mol[i],decodings,rxn)
                Smiles=Chem.MolToSmiles(mol)
                print("smiles{}".format(Smiles))
                # print(f"succ{i},{result}")
                # print("this is try .....,smile:{}".format(Smiles))
                # 转换成mol2文件
                # mol = Chem.MolFromSmiles('COc1cc(ccc1OCC2CN(CCCO2)Cc3ccc(cc3)Br)Cl')
                # print("++++++++++++++++++++++")
                # 计算前4个属性值
                Chem.GetSSSR(mol)
                clogp = Crippen.MolLogP(mol)
                mw = MolDescriptors.CalcExactMolWt(mol)
                tpsa = Descriptors.TPSA(mol)
                # # 计算相似性
                # # 计算相似性
                # m1 = Chem.MolFromMol2File('/home/developer/wq/3.6/sdy-14.mol2')
                # m2 = Chem.MolFromMol2File('/home/developer/wq/3.6/sdy-95.mol2')
                # ms = [m1, m2, Chem.Mol(mol)]
                # fps = [Chem.RDKFingerprint(x) for x in ms]
                # sim1 = DataStructs.FingerprintSimilarity(fps[0], fps[2])
                # sim2 = DataStructs.FingerprintSimilarity(fps[1], fps[2])
                # if(sim1 >= 0.8) or (sim2 >= 0.8):
                #     print("target molecule is {},sim1 is{},sim2 is {}".format(str(Chem.MolToSmiles(mol))),sim1,sim2)
                frs[i][0] = True
                frs[i][1] = 100 < mw < 1000
                # frs[i][2] = 4 < clogp < 7
                # frs[i][3] = 80 < tpsa < 102
                print("mw:{},clogP:{},tpsa:{}".format(mw, clogp, tpsa))
                # os.system("obabel -ismiles /home/b519/lyy/lyy2/deep/ligand/temp.smi -omol2 -O /home/b519/lyy/lyy2/deep/ledock/ligand/temp.mol2 --gen3D")
                # print("*****************************")
                # mymol = readstring("smi", smi)
                # mymol.write("smi", path+str(i)+".smi",overwrite=True)
                # os.system("obabel -ismiles "+path+str(i)+".smi -omol2 -O "+path+str(i)+".mol2 --gen3D")
                print('obabel -:"' + Smiles + '" --gen3d -omol2 -O ' + path + str(i) + '.mol2')
                os.system('obabel -:"' + Smiles + '" --gen3d -omol2 -O ' + path + str(i) + '.mol2')
                # 将路径写入文件
                # print("+++++++++++++++++"+filename)
                with open(filename, 'a') as file_object:
                    file_object.write(path + str(i) + ".mol2\n")
                mollist.append(i)
                keylist.append(key)
            except:
                frs[i] = [False] * FEATURES
        else:
            frs[i] = [False] * FEATURES
    # print("rewards1:")
    # print(frs)
    # 调用对接接口
    if len(mollist) == 0:
        # print('len(mollist) == 0'+str(len(evaluated_mols)))
        return frs
    # result_list_str = Client(filename)
    # result_list_str = sftpClient(
    #    _ligandListFilePath=filename,
    #    _remotepath='/lustre/home/weizhiqiang/',
    #    _hostname='10.130.2.222',
    #    _username='weizhiqiang',
    #    _passwd='IAOS1234'
    # )
    # print(result_list_str)ss
    # print(type(result_list_str))
    # 获取对接结果，拼接成模型输入
    # 将获得的结果转换成字典
    # 死循环，只有当try到正确结果时，才会退出循环，否则将等待6s后重新执行，同时将这次的中断输出到日志
    result_list_str = Client(filename)
    while True:
        if (result_list_str == None):
            # 当result_list_str为None的时候，走if
            time.sleep(6)  # 等待6s
            # 日志输出
            _now_time = time.time()
            logging.info('=' * 100)
            logging.info(str((time.strftime('%Y-%m-%d %H:%M:%S'))))
            logging.info('The result_list_str is None')
            logging.info('=' * 100)
            # 重新执行
            result_list_str = Client(filename)
        else:
            # 如果result_list_str不为None的话，就尝试
            try:
                result_list = result_list_str.strip('[]').split('], [')
                break
            except BaseException as e:
                time.sleep(6)  # 等待6s
                # 日志输出
                _now_time = time.time()
                logging.info('=' * 100)
                logging.info(str((time.strftime('%Y-%m-%d %H:%M:%S'))))
                logging.info('The interrupted service has been restarted')
                logging.error(e)
                logging.info('=' * 100)
                # 重新执行
                result_list_str = Client(filename)
    # result_list = result_list_str.strip('[]').split('], [')
    # TDO 容错机制，若没有结果，则继续检测
    # del(result_list[-1])#删除掉最后的空格元素
    # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    # print(len(result_list))
    # print(result_list)
    result_key_value = {}
    # print("---------------------------------------")
    for result in result_list:
        # result= result.strip('[]')
        # print(result)
        strs = result.split(', ')
        # print(strs)
        molNo = strs[0].strip("'")
        pdbId = strs[1].strip("'").split('/')[1]
        key = molNo + "_" + pdbId
        # print("key="+key)
        value = float(strs[2])
        result_key_value[key] = value
        # print("key")
        # print(key)
        # print("value")
        # print(value)
    ##拼接结果为感知机的输入
    PDBpath = rootPath + '/ledock/Mpro/Mpro/ledock_in.list'
    print(len(result_key_value))
    for i, key in zip(mollist, keylist):
        with open(PDBpath, "r") as f:
            for line in f:
                # ('./SARS-CoV-2/7JQ4', '7JQ4_ledock.in')
                p1, f1 = os.path.split(line)
                # ('./SARS-CoV-2', '7JQ4')
                p2, pdbID = os.path.split(p1)
                k = str(i) + ".dok" + "_" + pdbID
                try:
                    val = result_key_value[k]
                except:
                    val = 0.0
                # print(len(X))
            # print("i:"+str(i))
            # print(X)
        # 预测activity
        # print(len(X))
        # print(len(W))
        # print(len(b[0]))
        frs[i][2] = (val<=-8.5)
        evaluated_mols[key] = (np.array(frs[i]), epoch)
        # if np.sum(frs[i]) >= 4:
        #     # mol = decode(batch_mol[i], decodings)
        #     smile_lib = []
        #     ac_smile_dict = decodings
        #     for key, value in ac_smile_dict.items():
        #         # print(key)
        #         smile_lib.append(value)
        #     smiles = array(smile_lib)[batch_mol[i]]
        #     result = genertate_long_mol(smiles, rxn)
        #     print(f"succ{i},{result}")
        #     mol = Chem.MolFromSmiles(result)
        #     Smiles = str(Chem.MolToSmiles(mol))
        #     with open(smile_path, 'a') as smile_txt:
        #         smile_txt.write(Smiles + '\n')
    # print("final_frs")
    # print(frs)
    return frs
'''
#**# Same as above but decodes and check if a cached value could be used.
def evaluate_mol(fs, epoch, decodings):

    #print("This is evaluate_mol.....")

    global evaluated_mols

    key = get_key(fs)

    if key in evaluated_mols:
        return evaluated_mols[key][0]

    try:
        mol = decode(fs, decodings)
        ret_val = evaluate_chem_mol(mol)
    except:
        ret_val = [False] * 5

    evaluated_mols[key] = (np.array(ret_val), epoch)

    return np.array(ret_val)
'''

# Get initial distribution of rewards among lead molecules
def get_init_dist(X, decodings):
    # arr = np.asarray([evaluate_mol(X[i], -1, decodings) for i in range(X.shape[0])])
    # X_mat=np.full(X.shape,np.nan)
    # arr = np.asarray(evaluate_batch_mol(X_mat,X,-1,decodings))
    X_mat = np.full(X.shape, np.nan)
    frs = evaluate_batch_mol(X_mat, X, -1, decodings)
    133*3
    arr = np.asarray(frs)
    dist = arr.shape[0] / (1.0 + arr.sum(0))
    open("dist.txt",'w').write(dist)
    return dist
# Discard molecules which fulfills all targets (used to remove to good lead molecules).
def clean_good(X, decodings):
    X_mat = np.full(X.shape, np.nan)
    frs = evaluate_batch_mol(X_mat, X, -1, decodings)
    X = [X[i] for i in range(X.shape[0]) if not
    np.array(frs[i]).all()]
    return np.asarray(X), frs
'''
# 调用测试
if __name__ == '__main__':
    batch_mol = np.array([[[1, 1, 0, 1, 1, 0, 1, 0,],[1, 0, 1, 0, 1, 1, 0, 1,],[1, 1, 0, 1, 1, 1, 1, 0,],[1, 0, 1, 1, 0, 0, 1, 1,],[1, 0, 1, 0, 1, 1, 1, 1,],[1, 1, 0, 0, 0, 1, 0, 1,],[1, 0, 1, 0, 1, 0, 1, 1,],[1, 1, 1, 1, 1, 0, 0, 1,],[0, 0, 0, 0, 0, 0, 0, 0,],[0, 0, 0, 0, 0, 0, 0, 0,],[0, 0, 0, 0, 0, 0, 0, 0,],[0, 0, 0, 0, 0, 0, 0, 0,]],[[1, 1, 0, 1, 1, 0, 1, 0,],[1, 0, 1, 0, 1, 1, 0, 1,],[1, 1, 0, 1, 1, 1, 1, 0,],[1, 0, 1, 1, 1, 0, 1, 1,],[1, 0, 1, 1, 1, 1, 1, 1,],[1, 1, 0, 0, 0, 1, 0, 0,],[1, 0, 1, 0, 1, 1, 1, 1,],[1, 1, 0, 1, 1, 1, 1, 0,],[1, 1, 1, 1, 0, 0, 0, 0,],[0, 0, 0, 0, 0, 0, 0, 0,],[0, 0, 0, 0, 0, 0, 0, 0,],[0, 0, 0, 0, 0, 0, 0, 0,]]])
    decoding = {'0000000': 'O=C1N[C@@H]([Lu])C(=O)N[C@H]1[Yb]', '0000001': 'O=C1OCCN1[Yb]', '0000010': 'C=C1NC(=O)[C@@H]([Yb])NC1=O', '0000011': 'C=C1NC(=O)[C@H]([Yb])NC1=O', '0001000': 'CS(=O)(=O)N([Yb])/C(=C\\[Lu])SC#N', '0001001': 'CS(=O)(=O)N([Yb])/C(=C\\[Lu])[Se]C#N', '0001100': 'CS(=O)(=O)N([Yb])[Lu]', '0001101': 'O=S(=O)([Lu])N([Yb])[Ta]', '0001110': 'CN([Yb])S(=O)(=O)[Lu]', '0001111': 'CN([Yb])S(C)(=O)=O', '0100000': '[Yb]c1oc2c(c1[Lu])CCCC2', '0100010': '[Yb]C1CCCCC1', '0100011': '[Yb][C@@H]1C[C@H]([Lu])CC[C@H]1[Ta]', '0100100': '[Yb]c1ccsc1', '0100101': '[Yb]c1cccs1', '0100110': '[Yb]c1scc([Lu])c1[Ta]', '0100111': '[Yb]c1occ([Lu])c1[Ta]', '0101000': '[Yb]c1cccc([Lu])c1', '0101001': '[Yb]c1ccc([Ta])c([Lu])c1', '0101010': '[Yb]c1cccnc1', '0101011': '[Yb]c1ccccc1', '0101100': '[Yb]c1ccccc1[Lu]', '0101101': '[Yb]c1ccc([Lu])c([Ta])c1', '0101110': '[Yb]c1ccc([Lu])cc1[Ta]', '0101111': '[Yb]c1ccc([Lu])cc1', '0110000': '[Yb]c1cc([Lu])ccn1', '0110001': '[Yb]c1ncc([Lu])o1', '0110010': '[Yb]C1CC([Lu])=NN1', '0110011': '[Yb]C1CC([Ta])=NN1[Lu]', '0110100': '[Yb]n1ccc2ccccc21', '0110101': '[Yb]c1ccc2c(c1)OCO2', '0110110': '[Yb]c1cc2ccccc2o1', '0110111': '[Yb]c1nc2ccccc2s1', '0111000': '[Yb]c1cnc2ccccc2c1', '0111001': '[Yb]c1c[nH]c2ccccc12', '0111010': '[Yb]c1ccc2ccccc2c1', '0111011': '[Yb]c1ccc2cc([Lu])ccc2c1', '0111100': '[Yb]c1c([Ta])ccc2c1ccn2[Lu]', '0111101': '[Yb]c1cc([Ta])cc2c1ccn2[Lu]', '0111110': '[Yb]c1cccc2c1c([Ta])cn2[Lu]', '0111111': '[Yb]c1cccc2c1ccn2[Lu]', '1000000': 'O=P([Yb])(OC[Lu])OC[Ta]', '1000001': 'COP(=O)([Yb])OC', '1000010': 'CC(C)OP(=O)([Yb])OC(C)C', '1000011': 'CCOP(=O)([Yb])OCC', '1000100': '[Yb]SC[Lu]', '1000101': '[Yb]O[Lu]', '1000110': '[Yb]OC[Lu]', '1000111': '[Yb]C[Lu]', '1010000': 'O=[N+]([O-])[Yb]', '1010001': 'Br[Yb]', '1010010': 'C=CC(C)(C)C(=O)[Yb]', '1010011': 'CC(C)(C)P(=O)([Yb])C(C)(C)C', '1010100': 'O=C[Yb]', '1010101': 'COCC[Yb]', '1010110': 'N#C[Yb]', '1010111': 'N#CC[Yb]', '1011000': 'I[Yb]', '1011001': 'F[Yb]', '1011010': 'O[Yb]', '1011011': 'C[Yb]', '1011100': 'CC[Yb]', '1011101': 'Cl[Yb]', '1011110': 'CCO[Yb]', '1011111': 'CO[Yb]', '1100000': '[Yb]C1=NN([Ta])C([Lu])C1', '1100001': 'CC(C[Yb])S[Lu]', '1100010': 'FC(S[Lu])=C([Yb])[Ta]', '1100011': 'F/C(=C\\[Yb])S[Lu]', '1100100': '[Yb]C=C(S[Lu])S[Ta]', '1100101': 'O=S(=O)([Lu])C(F)C[Yb]', '1100110': 'O=S(=O)([Lu])/C([Ta])=C/[Yb]', '1100111': 'O=S(=O)([Lu])/C(F)=C/[Yb]', '1101000': 'CC(C)C[Yb]', '1101001': 'CC(C)(C)[Yb]', '1101010': 'FC(F)(F)O[Yb]', '1101011': 'FC(F)(F)[Yb]', '1101100': 'OCCCCCCS[Yb]', '1101101': 'CCC(C)[Yb]', '1101110': '[Yb]C1CC1', '1101111': 'CC(C)[Yb]', '1110000': 'O=C1CSC([Yb])=N1', '1110001': 'CCOC(=O)/C(F)=C(/[Yb])C(=O)[Lu]', '1110010': 'O=C([Lu])/C=C/[Yb]', '1110011': 'O=C([Lu])CS[Yb]', '1110100': 'O=C(CS[Yb])O[Lu]', '1110101': 'O=C([Yb])O[Lu]', '1110110': 'COC(=O)CC[Yb]', '1110111': 'COC(=O)CS[Yb]', '1111000': 'CCCCOC(=O)[Yb]', '1111001': 'CS(=O)(=O)[Yb]', '1111010': 'CCCC(=O)[Yb]', '1111011': 'CCOC(=O)[Yb]', '1111100': 'CC(=O)[Yb]', '1111101': 'NC(=S)[Yb]', '1111110': 'CCC(=O)[Yb]', '1111111': 'COC(=O)[Yb]'}
    evaluate_batch_mol(batch_mol,1,decoding)
'''