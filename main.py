import argparse
from config import Config
from src.utils import set_global_seed
from src.trainer import train
from src.tester import test


def main():
    parser = argparse.ArgumentParser(description="PPO-ACO for TSP Meta-Optimization")

    # ===== 运行模式选择 =====
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Execution mode: train or test')

    # ===== 状态特征消融 =====
    parser.add_argument('--no_entropy', action='store_true',
                        help='Disable entropy feature')
    parser.add_argument('--no_entropy_trend', action='store_true',
                        help='Disable entropy trend feature')
    parser.add_argument('--no_stagnation_feature', action='store_true',
                        help='Disable stagnation feature')

    # ===== 参数控制消融 =====
    parser.add_argument('--fix_alpha', action='store_true',
                        help='Fix alpha (disable RL control)')
    parser.add_argument('--fix_beta', action='store_true',
                        help='Fix beta (disable RL control)')
    parser.add_argument('--fix_rho', action='store_true',
                        help='Fix rho (disable RL control)')

    args = parser.parse_args()

    Config.no_entropy = args.no_entropy
    Config.no_entropy_trend = args.no_entropy_trend
    Config.no_stagnation_feature = args.no_stagnation_feature

    Config.fix_alpha = args.fix_alpha
    Config.fix_beta = args.fix_beta
    Config.fix_rho = args.fix_rho

    # 设置全局种子
    set_global_seed(Config.seed)

    if args.mode == 'train':
        print("=== Training Mode ===")
        print("===== Ablation Settings =====")
        print(f" no_entropy             : {Config.no_entropy}")
        print(f" no_entropy_trend       : {Config.no_entropy_trend}")
        print(f" no_stagnation_feature  : {Config.no_stagnation_feature}")
        print(f" fix_alpha              : {Config.fix_alpha}")
        print(f" fix_beta               : {Config.fix_beta}")
        print(f" fix_rho                : {Config.fix_rho}")
        print("=============================")
        train(
            state_dim=Config.state_dim,
            hidden_dim=Config.hidden_dim,
            action_dim=Config.action_dim,
            actor_lr=Config.actor_lr,
            critic_lr=Config.critic_lr,
            lmbda=Config.lmbda,
            entropy_coef=Config.entropy_coef,
            epochs=Config.epochs,
            eps=Config.eps,
            gamma=Config.gamma,
            device=Config.device,
            training_epoch=Config.training_epoch,
            training_iter=Config.training_iter,
            training_minimun_city=Config.training_minimun_city,
            training_maximun_city=Config.training_maximun_city
        )
    elif args.mode == 'test':
        """加载模型并测试"""
        print("=== Testing Mode ===")
        print("===== Ablation Settings =====")
        print(f" no_entropy             : {Config.no_entropy}")
        print(f" no_entropy_trend       : {Config.no_entropy_trend}")
        print(f" no_stagnation_feature  : {Config.no_stagnation_feature}")
        print(f" fix_alpha              : {Config.fix_alpha}")
        print(f" fix_beta               : {Config.fix_beta}")
        print(f" fix_rho                : {Config.fix_rho}")
        print("=============================")
        test(testing_epoch=Config.testing_epoch, testing_iter=Config.testing_iter,
             testing_minimun_city=Config.testing_minimun_city,
             testing_maximun_city=Config.testing_maximun_city)


if __name__ == '__main__':
    main()