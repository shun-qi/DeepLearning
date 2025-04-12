import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# 实现了一个使用深度 Q 网络 (Deep Q-Network, DQN) 算法的智能体（Agent），用于解决一个经典的强化学习问题——网格世界 (Grid World)
# Policy 是在当前状态下选择什么动作的
'''
Q 值 (Action-Value): Q(s, a) 表示在状态 s 下执行动作 a 后，遵循当前策略所能获得的预期未来累积奖励。
DQN 的核心就是学习这个 Q 函数。
'''

'''
重点：
1.机器学习3要素：1）数据     2）模型    3）损失
    这里数据是合成的，通过策略合成的
    模型是一个简单的FNN
    损失是MSE
2
'''


# 定义网格世界环境
class GridWorldEnv:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = [0, 0]  # 智能体初始位置
        self.goal_pos = [size-1, size-1]  # 目标位置
        self.obstacles = [[1, 1], [2, 2], [3, 1]]  # 障碍物位置
        
    def reset(self):
        self.agent_pos = [0, 0]
        return self._get_state()
    
    def _get_state(self):
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
        # 动作: 0-上, 1-右, 2-下, 3-左
        directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        new_pos = [self.agent_pos[0] + directions[action][0], 
                   self.agent_pos[1] + directions[action][1]]
        
        # 检查是否超出网格边界或撞到障碍物
        if (new_pos[0] < 0 or new_pos[0] >= self.size or 
            new_pos[1] < 0 or new_pos[1] >= self.size or
            new_pos in self.obstacles):
            reward = -1  # 惩罚
            return self._get_state(), reward, False
        
        # 更新智能体位置
        self.agent_pos = new_pos
        
        # 检查是否到达目标
        if self.agent_pos == self.goal_pos:
            return self._get_state(), 10, True  # 大奖励，完成任务
        else:
            return self._get_state(), -0.1, False  # 小惩罚，鼓励快速到达目标
    
    def render(self):
        grid = np.zeros((self.size, self.size))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1  # 智能体
        grid[self.goal_pos[0], self.goal_pos[1]] = 2    # 目标
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 3  # 障碍物
            
        cmap = ListedColormap(['white', 'blue', 'green', 'red'])
        plt.figure(figsize=(5, 5))
        plt.imshow(grid, cmap=cmap)
        plt.grid(True)
        plt.title('Grid World')
        plt.show()

# 定义Q网络，本质就是一个FNN
# 输入状态，输出一个action
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__() # 调用父类的构造函数
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)
        
    def forward(self, x):  #前馈网络，告诉在这个网络中数据如何前向传播的
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 定义深度Q学习智能体
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = [] # 经验回放缓冲区 (Experience Replay Buffer)，用于存储过去的 (state, action, reward, next_state, done) 经验元组。
        self.gamma = 0.95  # 折扣因子，决定了未来奖励相对于当前奖励的重要性。值越接近 1，智能体越看重长期奖励。
        self.epsilon = 1.0  # 探索率 epsilon概率探索，1-epsilon概率采取行动
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # 每次学习后，epsilon 会乘以这个值，逐渐减少探索，增加利用。
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 用于根据损失更新 Q 网络的权重。
        self.criterion = nn.MSELoss()   # 损失
        
    def remember(self, state, action, reward, next_state, done):
        # 将一次交互的经验 (s, a, r, s', done) 存储到 memory 中。
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        # 根据当前状态 state 选择一个动作。
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # 从 0到self.action_size-1中随机选择一个整数
        
        state_tensor = torch.FloatTensor(self._state_to_tensor(state))
        act_values = self.model(state_tensor)
        return torch.argmax(act_values).item() # .item是把tensor索引变为python的整型，这里是下标，也就是action的值
    
    def _state_to_tensor(self, state): # 一般深度学习会这样处理，防止1，2这种数据会让模型认为有顺序关系
        # 将状态转换为one-hot编码
        tensor = np.zeros(self.state_size)
        tensor[state] = 1
        return tensor
    
    def replay(self, batch_size):
        # 核心学习步骤 (经验回放和网络更新)
        # 检查 memory 中的经验数量是否足够进行一次 batch_size 的训练。
        if len(self.memory) < batch_size: 
            return
        
        minibatch = random.sample(self.memory, batch_size) # 从 memory 中随机采样一个 minibatch 的经验。随机采样打破了经验之间的时间相关性，有助于稳定训练。
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(self._state_to_tensor(state))
            next_state_tensor = torch.FloatTensor(self._state_to_tensor(next_state))
            
            """
            如果 done 为 True (回合结束)，目标 Q 值就是即时奖励 reward。因为没有后续状态了。
            如果 done 为 False，目标 Q 值是 reward + self.gamma * max(Q(next_state, a'))。
            其中 max(Q(next_state, a')) 是 Q 网络对 next_state 预测的所有动作 Q 值中的最大值 
            (torch.max(self.model(next_state_tensor)).item())。
            这表示了从下一个状态开始能获得的最大预期未来奖励。    
            """
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            # 损失函数就比较目标 target_f_clone 和原本预测的 target_f

            target_f = self.model(state_tensor)     # 预测的 target_f
            target_f_clone = target_f.clone()
            target_f_clone[action] = target     # 更新后的target_f_clone是真实值
            
            self.optimizer.zero_grad()
            loss = self.criterion(target_f.unsqueeze(0), target_f_clone.unsqueeze(0)) #unsqueeze(0)变为二维： 因为很多 PyTorch 损失函数（比如 MSELoss）要求输入是至少二维的，通常是 (batch_size, 输出大小)这种形状。
            loss.backward()
            self.optimizer.step()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 主函数
def train_and_visualize():
    env = GridWorldEnv(size=5)
    state_size = env.size ** 2
    action_size = 4  # 上, 右, 下, 左
    agent = DQNAgent(state_size, action_size)

    # 训练参数
    episodes = 500
    batch_size = 32
    max_steps = 100
    rewards = []

    # 训练阶段
    print("训练智能体...")
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done or step == max_steps - 1:
                if e % 50 == 0:
                    print(f"回合: {e+1}/{episodes}, 步数: {step+1}, 总奖励: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                rewards.append(total_reward)
                break
                
        agent.replay(batch_size)
    
    # 绘制训练奖励变化
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('训练过程中的奖励变化')
    plt.xlabel('回合数')
    plt.ylabel('总奖励')
    plt.grid(True)
    plt.show()
    
    # 测试阶段 - 可视化学到的路径
    print("\n测试智能体...")
    state = env.reset()
    print("初始状态:")
    env.render()
    
    # 记录路径
    path = [env.agent_pos.copy()]
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        action = agent.act(state)
        state, reward, done = env.step(action)
        path.append(env.agent_pos.copy())
        steps += 1
        
        if done:
            print(f"目标达成! 用了{steps}步")
            env.render()
    
    # 绘制智能体的路径
    grid = np.zeros((env.size, env.size))
    grid[env.goal_pos[0], env.goal_pos[1]] = 2    # 目标
    for obs in env.obstacles:
        grid[obs[0], obs[1]] = 3  # 障碍物
    
    cmap = ListedColormap(['white', 'blue', 'green', 'red'])
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap=cmap)
    plt.grid(True)
    
    # 绘制路径
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    plt.plot(path_x, path_y, 'b-o', linewidth=2, markersize=8)
    plt.title('path')
    plt.show()
    plt.savefig('plot.png')
    
    print("\n输入: 初始状态 (智能体在[0,0])")
    print("输出: 学到的到达目标的路径")
    print(f"路径: {path}")

# 运行模型
train_and_visualize()