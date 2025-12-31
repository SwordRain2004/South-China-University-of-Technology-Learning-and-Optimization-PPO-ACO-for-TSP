import torch


class Config:
    # 路径设置
    MODEL_DIR = "./models"
    RESULTS_DIR = "./results"
    MODEL_NAME = "ppo_aco"

    # 环境与种子
    seed = 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # PPO 超参数
    state_dim = 3
    action_dim = 3  # [α, β, ρ]
    hidden_dim = 128  # 隐藏层数量
    actor_lr = 1e-4  # actor学习率
    critic_lr = 5e-3  # critic学习率
    gamma = 0.99  # 回报折扣率
    lmbda = 0.9  # 优势函数参数
    entropy_coef = 0.01  # 熵正则参数
    epochs = 10  # PPO训练次数
    eps = 0.2  # PPO裁剪系数
    alpha_lower_bound = 0.5
    alpha_upper_bound = 3.0
    beta_lower_bound = 1.0
    beta_upper_bound = 5.0
    pho_lower_bound = 0.1
    pho_upper_bound = 0.5

    # 训练参数
    training_minimun_city = 30
    training_maximun_city = 70
    training_epoch = 30
    training_iter = 100
    # training_epoch = 1
    # training_iter = 1

    # 测试参数
    testing_minimun_city = 30
    testing_maximun_city = 300
    testing_epoch = 10
    testing_iter = 50
    # testing_epoch = 1
    # testing_iter = 1

    # Vanilla ACO参数设置
    Vanilla_ACO_parameters = (1.0, 3.0, 0.3)

    # 消融实验
    no_entropy = False
    no_entropy_trend = False
    no_stagnation_feature = False
    fix_alpha = False
    fix_beta = False
    fix_rho = False
