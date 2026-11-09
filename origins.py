import math

import numpy as np
import random
from scipy.stats import norm


def calculate_edge_weights(edge_map, successors):
    """
    param: edge_map，允许 passenger 通行的路段字典
    param: successors，各路段的后继路段字典
    return: edge_weights，各路段的属性和原始权重字典
    """
    edge_weights = {}

    for edge_id, edge in edge_map.items():
        # 该路段的全部车道数量
        lane_count = edge.getLaneNumber()

        # 该路段的后继路段数量
        successor_count = len(successors[edge_id])

        # 路段长度，单位为米
        length = edge.getLength()

        # 计算原始权重
        weight = lane_count * successor_count * length

        edge_weights[edge_id] = {
            "lane_count": lane_count,
            "successor_count": successor_count,
            "length": length,
            "weight": weight,
        }

    return edge_weights

def classify_days(total_days, event_days, rng):
    """
    param: total_days，模拟总天数，第 1 天默认为星期一
    param: event_days，随机选作突发模式的天数
    param: rng，随机数生成器
    return: 按模式划分的天数列表字典
    """
    if not isinstance(total_days, int) or isinstance(total_days, bool) or total_days < 1:
        raise ValueError("total_days 必须是大于 0 的整数")
    if not isinstance(event_days, int) or isinstance(event_days, bool) or not 0 <= event_days <= total_days:
        raise ValueError("event_days 必须是 0 到 total_days 之间的整数")

    # 在1到total_days之间随机选择event_days个天数作为突发模式
    events = set(rng.sample(range(1, total_days + 1), event_days))
    days = {"weekday": [], "weekend": [], "event": []}
    for day in range(1, total_days + 1):
        if day in events:
            days["event"].append(day)
        elif (day - 1) % 7 < 5:
            days["weekday"].append(day)
        else:
            days["weekend"].append(day)
    return days


def choose_centers(candidates, count, rng):
    """
    param: candidates，候选路段列表，每项包含 rank、edge_id、score
    param: count，需要抽取的不同中心数量
    param: rng，随机数生成器
    return: 按原始权重不放回抽取的中心列表
    """
    pool = [road for road in candidates if road["score"] > 0]
    if len(pool) < count:
        raise ValueError(f"正权重候选路段不足：需要 {count} 条，只有 {len(pool)} 条")
    selected = []
    for _ in range(count):
        weights = [road["score"] for road in pool]
        # 每次只抽 1 个，并移出候选池，实现按权重不放回抽样。
        index = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        selected.append(pool.pop(index))
    return selected


def normalize_components(centers, sigma_by_id):
    """
    param: centers，当前模式的 50 个中心路段
    param: sigma_by_id，中心路段 ID 到标准差的字典
    return: 当前模式归一化后的高斯分量列表
    """
    total = math.fsum(road["score"] for road in centers)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("分量原始权重之和必须是正的有限数")
    return [
        {
            "edge_id": road["edge_id"],
            "rank": road["rank"],
            "mean": road["rank"],
            "score": road["score"],
            "weight": road["score"] / total,
            "std_dev": sigma_by_id[road["edge_id"]],
            "variance": sigma_by_id[road["edge_id"]] ** 2,
        }
        for road in centers
    ]


def create_patterns(bfs_result, edge_weights, seed=None):
    """
    param: bfs_result，BFS 排序结果，每项包含 rank 和 edge_id
    param: edge_weights，路段权重字典，每项的 weight 是原始权重
    param: seed，随机种子；None 表示不固定
    return: 工作日、周末和突发模式的高斯分量字典
    """
    rng = random.Random(seed)
    roads = []
    for row in bfs_result:
        edge_id, rank = row["edge_id"], row["rank"]
        if edge_id not in edge_weights:
            raise ValueError(f"缺少路段权重：{edge_id}")
        score = float(edge_weights[edge_id]["weight"])
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"路段权重必须为非负有限数：{edge_id}")
        roads.append({"edge_id": edge_id, "rank": rank, "score": score})

    R = len(roads)
    ids = [road["edge_id"] for road in roads]
    ranks = [road["rank"] for road in roads]
    if len(set(ids)) != R or set(ranks) != set(range(1, R + 1)):
        raise ValueError("BFS 结果必须包含不重复的 edge_id 和从 1 到 R 的连续排序号")
    roads.sort(key=lambda road: road["rank"])
    d = R / 10

    weekday = choose_centers(roads, 50, rng)
    weekday_ids = {road["edge_id"] for road in weekday}
    base_sigma = 0.4 * d

    weekend_removed = set(rng.sample(sorted(weekday_ids), 25))
    weekend_kept = [road for road in weekday if road["edge_id"] not in weekend_removed]
    weekend_new = choose_centers([road for road in roads if road["edge_id"] not in weekday_ids], 25, rng)
    weekend = weekend_kept + weekend_new
    weekend_ids = {road["edge_id"] for road in weekend}

    event_removed = set(rng.sample(sorted(weekday_ids), 15))
    event_kept = [road for road in weekday if road["edge_id"] not in event_removed]
    event_new = choose_centers([road for road in roads if road["edge_id"] not in weekday_ids | weekend_ids], 15, rng)
    event = event_kept + event_new

    weekday_sigmas = {road["edge_id"]: base_sigma for road in weekday}
    weekend_sigmas = {road["edge_id"]: base_sigma for road in weekend_kept}
    weekend_sigmas.update({road["edge_id"]: rng.uniform(0.36, 0.44) * d for road in weekend_new})
    event_sigmas = {road["edge_id"]: base_sigma for road in event_kept}
    event_sigmas.update({road["edge_id"]: rng.uniform(0.1, 0.8) * d for road in event_new})

    return {
        "weekday": normalize_components(weekday, weekday_sigmas),
        "weekend": normalize_components(weekend, weekend_sigmas),
        "event": normalize_components(event, event_sigmas),
    }



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