import numpy as np
import random

# 网格环境设置
grid_size = 4
n_actions = 4  # 上下左右4个动作
actions = ['up', 'down', 'left', 'right']
goal_state = (3, 3)  # 目标位置

# Q-Learning参数
alpha = 0.8  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率
n_episodes = 1000  # 训练轮数

# 初始化Q表
Q = np.zeros((grid_size, grid_size, n_actions))

# 计算当前位置的奖励
def get_reward(state):
    if state == goal_state:
        return 1  # 到达目标，奖励为1
    else:
        return -0.1  # 每走一步，惩罚为-0.1

# 选择动作（ε-greedy策略）
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(n_actions))  # 随机选择动作
    else:
        return np.argmax(Q[state[0], state[1]])  # 选择最大Q值对应的动作

# 环境状态更新
def next_state(state, action):
    if action == 0:  # 向上
        next_state = (max(state[0] - 1, 0), state[1])
    elif action == 1:  # 向下
        next_state = (min(state[0] + 1, grid_size - 1), state[1])
    elif action == 2:  # 向左
        next_state = (state[0], max(state[1] - 1, 0))
    else:  # 向右
        next_state = (state[0], min(state[1] + 1, grid_size - 1))
    return next_state

# Q-Learning算法
for episode in range(n_episodes):
    state = (0, 0)  # 初始状态为左上角
    done = False
    while not done:
        action = choose_action(state)
        next_s = next_state(state, action)
        reward = get_reward(next_s)
        
        # Q值更新
        Q[state[0], state[1], action] = Q[state[0], state[1], action] + alpha * (reward + gamma * np.max(Q[next_s[0], next_s[1]]) - Q[state[0], state[1], action])
        
        # 打印Q表的动态变化
        if episode % 100 == 0:  # 每100轮输出一次Q表
            print(f"Episode {episode + 1} Q-table:")
            print(Q)
        
        # 如果到达目标状态，结束
        if next_s == goal_state:
            done = True
        
        state = next_s

# 打印最终的Q表
print("训练后的Q表：")
print(Q)

# 显示最终策略
print("\n最终策略：")
for i in range(grid_size):
    for j in range(grid_size):
        action = np.argmax(Q[i, j])
        print(actions[action], end=" ")
    print()
