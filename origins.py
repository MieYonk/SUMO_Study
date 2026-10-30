import math

import numpy as np
from scipy.stats import norm


def build_mixture_distribution(components, bfs_result):
    """
    param: components，当前模式的高斯分量列表，包含 mean、std_dev、weight
    param: bfs_result，BFS 排序结果，每项包含 rank、edge_id
    return: 排序号、对应 edge_id、起点概率及有效范围内的原始概率
    """
    rank_to_id = {}
    for row in bfs_result:
        rank = row["rank"]
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
            raise ValueError("排序号必须是整数")
        if rank in rank_to_id:
            raise ValueError(f"排序号重复：{rank}")
        rank_to_id[rank] = row["edge_id"]
    R = len(rank_to_id)
    if R == 0 or set(rank_to_id) != set(range(1, R + 1)):
        raise ValueError("排序号必须从 1 到 R 连续且不重复")
    if len(set(rank_to_id.values())) != R:
        raise ValueError("路段 edge_id 不能重复")
    if not components:
        raise ValueError("高斯分量不能为空")

    raw_weights = [float(c["weight"]) for c in components]
    if any(not math.isfinite(w) or w < 0 for w in raw_weights):
        raise ValueError("分量权重必须是非负有限数")
    weight_sum = math.fsum(raw_weights)
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("分量权重之和必须是正的有限数")

    # 第 j 个排序号对应区间 [j-0.5, j+0.5)。
    bounds = np.arange(R + 1, dtype=np.float64) + 0.5
    mixed_cdf = np.zeros(R + 1, dtype=np.float64)
    for component, raw_weight in zip(components, raw_weights):
        mean = float(component["mean"])
        sigma = float(component["std_dev"])
        if not math.isfinite(mean) or not 1 <= mean <= R:
            raise ValueError("分量均值必须在有效排序号范围内")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("分量标准差必须是正的有限数")
        mixed_cdf += (raw_weight / weight_sum) * norm.cdf(bounds, loc=mean, scale=sigma)

    # CDF 的相邻差值是混合分布落在各整数区间的概率。
    masses = np.maximum(np.diff(mixed_cdf), 0.0)
    valid_mass = float(masses.sum())
    if not math.isfinite(valid_mass) or valid_mass <= 0:
        raise ValueError("混合分布在有效路段范围内没有可计算的概率")
    probabilities = masses / valid_mass
    return {
        "ranks": np.arange(1, R + 1),
        "edge_ids": [rank_to_id[rank] for rank in range(1, R + 1)],
        "probabilities": probabilities,
        "valid_mass": valid_mass,
    }


def sample_origins(distribution, count, rng):
    """
    param: distribution，由 build_mixture_distribution 生成的分布字典
    param: count，需要生成的车辆起点数量
    param: rng，NumPy 随机数生成器，例如 np.random.default_rng(42)
    return: 起点列表，每项包含 rank、edge_id
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("count 必须是非负整数")
    indices = rng.choice(len(distribution["ranks"]), size=count,
                         replace=True, p=distribution["probabilities"])
    return [{"rank": int(distribution["ranks"][i]),
             "edge_id": distribution["edge_ids"][i]} for i in indices]