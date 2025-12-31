import random
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from .utils import compute_ant_count, ppo_aco_until, vanilla_aco_until, load_agent
from .tsp import generate_tsp, distance_matrix
from config import Config


def test(testing_epoch=100, testing_iter=200,
    testing_minimun_city=10, testing_maximun_city=30):

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

    agent = load_agent(
        model_path=MODEL_PATH,
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
        device=Config.device
    )
    ppo_bests, vanilla_bests, all_params_records = [], [], []

    for i in range(testing_epoch):
        n_city = int(testing_minimun_city + i * (testing_maximun_city - testing_minimun_city) / (testing_epoch - 1))
        n_ants = compute_ant_count(n_city)
        dist = distance_matrix(generate_tsp(n_city))
        print(f"Testing Epoch: {i}\tCity Account: {n_city}")

        ppo_best_runs, vanilla_best_runs = [], []

        for run_idx in range(5):
            best_p, eval_p, params_history = ppo_aco_until(agent, dist, 0, n_ants=n_ants, max_iter=testing_iter)
            best_v, eval_v = vanilla_aco_until(dist, 0, n_ants=n_ants, max_iter=testing_iter, params=Config.Vanilla_ACO_parameters)

            ppo_best_runs.append(best_p)
            vanilla_best_runs.append(best_v)

            all_params_records.append({
                'epoch': i,
                'run': run_idx,
                'n_city': n_city,
                'history': params_history
            })

        ppo_bests.append(np.mean(ppo_best_runs))
        vanilla_bests.append(np.mean(vanilla_best_runs))

        print(f" PPO-ACO best = {ppo_bests[-1]:.4f}")
        print(f" Vanilla best = {vanilla_bests[-1]:.4f}")

    print("===== Test Complete =====")
    best_improve = []
    for pb, vb in zip(ppo_bests, vanilla_bests):
        best_improve.append((vb - pb) / vb * 100.0)

    with open(f'{Config.RESULTS_DIR}/{model_name}_solution_quality_comparison.txt', 'w') as f:
        f.write(f"Experiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        f.write(f"PPO-ACO vs Vanilla ACO solution quality improvement (%):\n")
        f.write(f" Mean = {np.mean(best_improve):.2f}%\n")
        f.write(f" Min = {np.min(best_improve):.2f}%\n")
        f.write(f" Max = {np.max(best_improve):.2f}%\n")
        f.write(f" Std = {np.std(best_improve):.2f}%\n")

    with open(f'{Config.RESULTS_DIR}/{model_name}_params_history.txt', 'w') as f:
        f.write(f"Parameter History Log\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for record in all_params_records:
            f.write(f"Epoch: {record['epoch']} | Run: {record['run']} | Cities: {record['n_city']}\n")
            f.write(f"{'Step':<6} {'Alpha':<10} {'Beta':<10} {'Rho':<10}\n")
            f.write("-" * 40 + "\n")

            for step, (alpha, beta, rho) in enumerate(record['history']):
                f.write(f"{step + 1:<6} {alpha:<10.4f} {beta:<10.4f} {rho:<10.4f}\n")

            f.write("\n" + "=" * 60 + "\n\n")

    # 绘制比较结果
    plt.figure(figsize=(10, 5))
    plt.plot(ppo_bests, label='PPO-ACO', alpha=0.7)
    plt.plot(vanilla_bests, label='Vanilla ACO', alpha=0.7)
    plt.xlabel('Test Instance')
    plt.ylabel('Best Path Length')
    plt.title('Solution Quality Comparison (High FEs)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{Config.RESULTS_DIR}/{model_name}_solution_quality_comparison.png')
    print("Experiment results saved as: \n"
          f'{Config.RESULTS_DIR}/{model_name}_solution_quality_comparison.txt\n'
          f'{Config.RESULTS_DIR}/{model_name}_params_history.txt\n'
          f'{Config.RESULTS_DIR}/{model_name}_solution_quality_comparison.png')
