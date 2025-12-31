import numpy as np
from config import Config

class ACO:
    """
    原始 ACO
    - 所有蚂蚁参与信息素更新
    - 无精英策略
    - 向量化路径构建 + 更新
    """
    def __init__(self, dist, n_ants, alpha, beta, rho):
        self.dist = dist
        self.n_city = dist.shape[0]
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.pheromone = np.ones((self.n_city, self.n_city))
        with np.errstate(divide='ignore'):
            self.heuristic = 1.0 / (dist + 1e-10)
            np.fill_diagonal(self.heuristic, 0.0)

    def set_params(self, alpha, beta, rho):
        if not Config.fix_alpha:
            self.alpha = alpha
        if not Config.fix_beta:
            self.beta = beta
        if not Config.fix_rho:
            self.rho = rho

    def run_iteration(self):
        n, m = self.n_city, self.n_ants

        # === 1. 构建转移概率矩阵 ===
        prob = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)

        # === 2. 初始化路径 ===
        paths = np.zeros((m, n), dtype=np.int32)
        visited = np.zeros((m, n), dtype=bool)

        start = np.random.randint(0, n, size=m)
        paths[:, 0] = start
        visited[np.arange(m), start] = True

        # === 3. 向量化构建路径 ===
        for t in range(1, n):
            curr = paths[:, t - 1]

            P = prob[curr].copy()  # (m, n)
            P[visited] = 0.0
            row_sum = P.sum(axis=1, keepdims=True)

            # 归一化（避免除 0）
            valid = row_sum.squeeze() > 0
            P[valid] /= row_sum[valid]

            # 多项分布采样（向量化）
            r = np.random.rand(m, 1)
            cdf = np.cumsum(P, axis=1)
            next_city = (cdf > r).argmax(axis=1)

            # 极端退化情况（全部为 0）
            dead = ~valid
            if np.any(dead):
                for i in np.where(dead)[0]:
                    unvisited = np.where(~visited[i])[0]
                    next_city[i] = np.random.choice(unvisited)

            paths[:, t] = next_city
            visited[np.arange(m), next_city] = True

        # === 4. 计算路径长度 ===
        length = np.sum(
            self.dist[paths, np.roll(paths, -1, axis=1)],
            axis=1
        )

        best_len = length.min()

        # === 5. 信息素更新 ===
        self.pheromone *= (1.0 - self.rho)

        delta = 1.0 / (length + 1e-12)
        delta = delta[:, None]

        from_city = paths
        to_city = np.roll(paths, -1, axis=1)

        np.add.at(self.pheromone, (from_city, to_city), delta)
        np.add.at(self.pheromone, (to_city, from_city), delta)

        self.pheromone = np.clip(self.pheromone, 1e-2, 1e2)
        # === 6. 种群熵 ===
        tau = self.pheromone
        p = tau / (tau.sum() + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        norm_factor = 2 * np.log(self.n_city) if self.n_city > 1 else 1.0
        normalized_entropy = entropy / norm_factor

        return best_len, normalized_entropy
