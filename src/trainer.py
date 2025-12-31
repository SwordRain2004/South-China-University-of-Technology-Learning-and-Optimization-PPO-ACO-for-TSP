import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from .ppo import PPO
from .aco import ACO
from .utils import compute_ant_count, map_action_to_range
from .tsp import *
from config import Config


def train(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, entropy_coef,
          epochs, eps, gamma, device, training_epoch=30, training_iter=50,
          training_minimun_city=10, training_maximun_city=30):
    print("Training Start")
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, entropy_coef, epochs, eps, gamma, device)
    # 训练记录
    episode_rewards = []

    for epoch in range(training_epoch):
        # 生成随机TSP问题
        n_city = random.randint(training_minimun_city, training_maximun_city)
        n_ants = compute_ant_count(n_city)
        dist = distance_matrix(generate_tsp(n_city))

        # 初始化ACO
        aco = ACO(dist, n_ants=n_ants, alpha=1.0, beta=1.0, rho=0.5)

        # 存储转换数据
        transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

        best_prev = float('inf')
        best_global = float('inf')
        stagnation = 0
        entropy_history = []

        state = [1.0, 0.0, 0.0]

        total_reward = 0

        for iter in range(training_iter):
            # 智能体选择动作（参数）
            action = agent.take_action(state)

            # 将动作映射到参数范围
            alpha = map_action_to_range(action[0], Config.alpha_lower_bound, Config.alpha_upper_bound)
            beta = map_action_to_range(action[1], Config.beta_lower_bound, Config.beta_upper_bound)
            rho = map_action_to_range(action[2], Config.pho_lower_bound, Config.pho_upper_bound)

            # 设置ACO参数
            aco.set_params(alpha, beta, rho)

            # 运行ACO迭代
            best_len, entropy = aco.run_iteration()

            # 计算熵变化趋势
            entropy_history.append(entropy)
            if len(entropy_history) >= 2:
                entropy_trend = entropy_history[-1] - entropy_history[-2]
            else:
                entropy_trend = 0.0

            # 计算停滞系数
            if best_len < best_global - 1e-8:
                stagnation = 0
                best_global = best_len
            else:
                stagnation += 1
            stagnation_feature = np.log1p(stagnation)

            # 计算奖励（路径长度的相对改进）
            if best_prev < float('inf'):
                reward = (best_prev - best_len) / (best_prev + 1e-8)
            else:
                reward = -0.01
            reward = np.clip(reward, -1.0, 1.0)

            # 更新最佳长度
            best_current = min(best_prev, best_len)

            # 计算下一状态
            next_state = [entropy, entropy_trend, stagnation_feature]
            if Config.no_entropy:
                next_state[0] = -1
            if Config.no_entropy_trend:
                next_state[1] = -1
            if Config.no_stagnation_feature:
                next_state[2] = -1

            # 存储转换
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(0.0 if iter < training_iter - 1 else 1.0)

            # 更新状态
            state = next_state
            best_prev = best_current
            total_reward += reward

        # 更新智能体
        agent.update(transition_dict)

        episode_rewards.append(total_reward)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{training_epoch}, Total Reward: {total_reward:.4f}, "
                  f"Avg Reward: {np.mean(episode_rewards[-5:]):.4f}")

    # 保存文件名（当前目录）
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    model_name = Config.MODEL_NAME
    if Config.no_entropy or Config.no_entropy_trend or Config.no_stagnation_feature:
        model_name = model_name + '_no'
    if Config.no_entropy:
        model_name = model_name + '_entropy'
    if Config.no_entropy_trend:
        model_name = model_name + '_entropy_trend'
    if Config.no_stagnation_feature:
        model_name = model_name + '_stagnation_feature'
    if Config.fix_alpha or Config.fix_beta or Config.fix_rho:
        model_name = model_name + '_fix'
    if Config.fix_alpha:
        model_name = model_name + '_alpha'
    if Config.fix_beta:
        model_name = model_name + '_beta'
    if Config.fix_rho:
        model_name = model_name + '_rho'
    MODEL_PATH = Config.MODEL_DIR + '/' + model_name + '.pth'

    # 保存模型
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }, MODEL_PATH)

    print(f"Training complete, the model saved as: {MODEL_PATH}")

    # 绘制训练奖励曲线
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curve')
    plt.grid(True)
    plt.savefig(f'{Config.RESULTS_DIR}/{model_name}_training_reward_curve.png')

    return agent
