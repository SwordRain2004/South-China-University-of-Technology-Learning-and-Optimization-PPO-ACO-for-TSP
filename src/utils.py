import os
import random
import numpy as np
import torch
from .aco import ACO
from config import Config

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

def map_action_to_range(action, lower_bound, upper_bound):
    action = np.clip(action, -1.0, 1.0)
    return lower_bound + 0.5 * (action + 1.0) * (upper_bound - lower_bound)

def compute_ant_count(n_city, min_ants=10, max_ants=40, scale=4.0):
    m = int(scale * np.sqrt(n_city))
    return int(np.clip(m, min_ants, max_ants))

def vanilla_aco_until(dist, target_len, n_ants, max_iter=100, params=(1.0, 3.0, 0.3)):
    alpha, beta, rho = params
    aco = ACO(dist, n_ants=n_ants, alpha=alpha, beta=beta, rho=rho)
    best = float('inf')
    for iter in range(max_iter):
        cur, _ = aco.run_iteration()
        best = min(best, cur)
        if best <= target_len:
            return best, iter + 1
    return best, max_iter

def ppo_aco_until(agent, dist, target_len, n_ants, max_iter=100):
    aco = ACO(dist, n_ants, 1.0, 3.0, 0.3)
    best_prev = float('inf')
    best_global = float('inf')
    stagnation = 0
    entropy_history = []
    params_history = []

    state = [1.0, 0.0, 0.0]

    for iter in range(max_iter):
        action = agent.take_action(state)
        alpha = map_action_to_range(action[0], Config.alpha_lower_bound, Config.alpha_upper_bound)
        beta = map_action_to_range(action[1], Config.beta_lower_bound, Config.beta_upper_bound)
        rho = map_action_to_range(action[2], Config.pho_lower_bound, Config.pho_upper_bound)
        aco.set_params(alpha, beta, rho)
        params_history.append((alpha, beta, rho))

        best_len, entropy = aco.run_iteration()

        entropy_history.append(entropy)
        if len(entropy_history) >= 2:
            entropy_trend = entropy_history[-1] - entropy_history[-2]
        else:
            entropy_trend = 0.0

        if best_len < best_global - 1e-8:
            stagnation = 0
            best_global = best_len
        else:
            stagnation += 1

        stagnation_feature = np.log1p(stagnation)

        state = [entropy, entropy_trend, stagnation_feature]
        if Config.no_entropy:
            state[0] = -1
        if Config.no_entropy_trend:
            state[1] = -1
        if Config.no_stagnation_feature:
            state[2] = -1

        best_prev = min(best_prev, best_len)

        if best_prev <= target_len:
            return best_prev, iter + 1, params_history

    return best_prev, max_iter, params_history

def load_agent(model_path, state_dim, hidden_dim, action_dim,
               actor_lr=1e-4, critic_lr=5e-3, lmbda=0.9, entropy_coef=0.01,
               epochs=10, eps=0.2, gamma=0.99, device=None):
    from .ppo import PPO

    print(f"[INFO] Loading PPO agent from: {model_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[ERROR] Model file not found: {model_path}\n"
            f"        Please check:\n"
            f"        1) The path is correct\n"
            f"        2) The model has been trained and saved\n"
            f"        3) The ablation flags match the model name"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                          lmbda, entropy_coef, epochs, eps, gamma, device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] Failed to load checkpoint from {model_path}\n"
            f"        Original error: {e}"
        )
    required_keys = [
        'actor_state_dict',
        'critic_state_dict',
        'actor_optimizer_state_dict',
        'critic_optimizer_state_dict'
    ]
    for k in required_keys:
        if k not in checkpoint:
            raise KeyError(
                f"[ERROR] Missing key '{k}' in checkpoint file: {model_path}"
            )
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    agent.actor.eval()
    agent.critic.eval()

    print("[INFO] Agent loaded successfully")
    return agent