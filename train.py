# -*- coding: utf-8 -*-
# @Time    : 2023/2/28 17:06
# @Author  : hxt
# @FileName: train.py
# @Software: PyCharm
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












LR_ACTOR = 0.01  # 策略网络的学习率
LR_CRITIC = 0.001  # 价值网络的学习率
GAMMA = 0.9  # 奖励的折扣因子
EPSILON = 0.9  # ϵ-greedy 策略的概率
TARGET_REPLACE_ITER = 100  # 目标网络更新的频率
N_ACTIONS = 20*4  # 动作数



# A2C 的主体函数
class A2C:
    def __init__(self):
        # 初始化策略网络，价值网络和目标网络。价值网络和目标网络使用同一个网络
        self.actor_net, self.critic_net, self.target_net =GAT(dim_in=128,dim_h=64,dim_out=4) , ValueGAT(dim_in=128,dim_h=64,dim_out=4), ValueGAT(dim_in=128,dim_h=64,dim_out=4)
        self.learn_step_counter = 0  # 学习步数
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr=LR_ACTOR)  # 策略网络优化器
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=LR_CRITIC)  # 价值网络优化器
        self.criterion_critic = nn.MSELoss()  # 价值网络损失函数

    def choose_action(self, graph_2_vec,s):
        if np.random.uniform() < EPSILON:  # ϵ-greedy 策略对动作进行采取
            action_value = self.actor_net(graph_2_vec,s)
            return torch.argmax(action_value)
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def learn(self, s, a, r, s_,graph_2_vec):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 更新目标网络
            self.target_net.load_state_dict(self.critic_net.state_dict())

        self.learn_step_counter += 1



        q_actor = self.actor_net(graph_2_vec,s)  # 策略网络
        q_critic = self.critic_net(graph_2_vec,s)  # 价值对当前状态进行打分
        q_next = self.target_net(graph_2_vec,s_).detach()  # 目标网络对下一个状态进行打分
        q_target = r + GAMMA * q_next  # 更新 TD 目标
        td_error = (q_critic - q_target).detach()  # TD 误差

        # 更新价值网络
        loss_critic = self.criterion_critic(q_critic, q_target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        # 更新策略网络
        log_q_actor = torch.log(q_actor)
        actor_loss = log_q_actor[a] * td_error
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
a2c = A2C()
def step(s,a,ac_smile_dict,epoch,count):
    done=False
    s_=update_edge_connect(s,a)
    try:
        mol,peptides=genertate_peptides_by_edge_index_list(edge_list_to_peptides(s_),ac_smile_dict)
        r = evaleuate_peptides(mol, epoch, count)
        if r >= 100:
            done = True
        return s_, r, done, peptides, mol
    except:
        r=-10
        return s_, r, done, "generate error", "error"

s=create_edge_connect(edge_num=20)
print(s)
filename = "Data/acid_1.csv"
ep_r = 0
smiles = read_smile_from_file(filename)
print(smiles)
print(len(smiles))
graph_2_vec = torch.from_numpy(create_pytorch_geometric_graph_data_list_from_smiles(smiles))
ac_smile_dict={}
for i,smile in enumerate(smiles):
    ac_smile_dict[i]=smile
print(graph_2_vec.shape)
print(ac_smile_dict)
for epoch in range(2000):
    if epoch%100==0:
        path="saved_model/"+str(epoch)+".pt"
        torch.save(a2c,path)
    print(f"Epoch{epoch}:")
    count=0
    while True:
        print(s)
        a = a2c.choose_action(graph_2_vec,s)  # 选择动作
        s_, r,done ,peptide,mol= step(s,a,ac_smile_dict,epoch,count)  # 执行动作
        print(s_, r,done ,peptide,mol)
        # 学习
        a2c.learn(s, a, r, s_,graph_2_vec)
        count+=1
        if done:
            break
        s = s_
    Chem.GetSSSR(mol)
    clogp = Crippen.MolLogP(mol)
    mw = MolDescriptors.CalcExactMolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    reward = (300 < mw < 800) + (0 < clogp < 2) + (160 < tpsa < 200)
    print(f"peptides:{peptide},smi:{Chem.MolToSmiles(mol)},lpgp:{clogp},tpsa:{tpsa},mw:{mw}")
