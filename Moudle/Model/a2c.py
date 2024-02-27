import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np

LR_ACTOR = 0.01     # 策略网络的学习率
LR_CRITIC = 0.001   # 价值网络的学习率
GAMMA = 0.9         # 奖励的折扣因子
EPSILON = 0.9       # ϵ-greedy 策略的概率
TARGET_REPLACE_ITER = 100                 # 目标网络更新的频率
env = gym.make('CartPole-v0')             # 加载游戏环境
N_ACTIONS = env.action_space.n            # 动作数
N_SPACES = env.observation_space.shape[0] # 状态数量
env = env.unwrapped

# 网络参数初始化，采用均值为 0，方差为 0.1 的高斯分布
def init_weights(m) :
    if isinstance(m, nn.Linear) :
        nn.init.normal_(m.weight, mean = 0, std = 0.1)

# 策略网络
class Actor(nn.Module) :
    def __init__(self):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_SPACES, 50),
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS) # 输出为各个动作的概率，维度为 3
        )

    def forward(self, s):
        output = self.net(s)
        output = F.softmax(output, dim = -1) # 概率归一化
        return output

# 价值网络
class Critic(nn.Module) :
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_SPACES, 20),
            nn.ReLU(),
            nn.Linear(20, 1) # 输出值是对当前状态的打分，维度为 1
        )

    def forward(self, s):
        output = self.net(s)
        return output

# A2C 的主体函数
class A2C :
    def __init__(self):
        # 初始化策略网络，价值网络和目标网络。价值网络和目标网络使用同一个网络
        self.actor_net, self.critic_net, self.target_net = Actor().apply(init_weights), Critic().apply(init_weights), Critic().apply(init_weights)
        self.learn_step_counter = 0 # 学习步数
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = LR_ACTOR)    # 策略网络优化器
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = LR_CRITIC) # 价值网络优化器
        self.criterion_critic = nn.MSELoss() # 价值网络损失函数

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), dim = 0) # 增加维度
        if np.random.uniform() < EPSILON :                 # ϵ-greedy 策略对动作进行采取
            action_value = self.actor_net(s)
            action = torch.max(action_value, dim = 1)[1].item()
        else :
            action = np.random.randint(0, N_ACTIONS)

        return action

    def learn(self, s, a, r, s_):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0 :          # 更新目标网络
            self.target_net.load_state_dict(self.critic_net.state_dict())

        self.learn_step_counter += 1

        s = torch.FloatTensor(s)
        s_ = torch.FloatTensor(s_)

        q_actor = self.actor_net(s)               # 策略网络
        q_critic = self.critic_net(s)             # 价值对当前状态进行打分
        q_next = self.target_net(s_).detach()     # 目标网络对下一个状态进行打分
        q_target = r + GAMMA * q_next             # 更新 TD 目标
        td_error = (q_critic - q_target).detach() # TD 误差

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

for epoch in range(10000) :

    s = env.reset()
    ep_r = 0
    while True :
        env.render()
        a = a2c.choose_action(s)       # 选择动作

        s_, r, done, info = env.step(a)# 执行动作

        x, x_dot, theta, theta_dot = s_
        # 修改奖励，为了更快收敛
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        ep_r += r

        # 学习
        a2c.learn(s, a, r, s_)

        if done :
            break

        s = s_
    print(f'Ep: {epoch} | Ep_r: {round(ep_r, 2)}')
