import math
def build_destination_candidates(edge_map):
    """
    param: edge_map，已筛选允许 passenger 通行的路段字典
    return: 终点候选列表，每项包含路段 ID、对象和长度乘车道数得到的权重
    """
    candidates = []
    for edge_id, edge in edge_map.items():
        length = float(edge.getLength())
        lane_count = edge.getLaneNumber()
        weight = length * lane_count
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"终点权重必须为非负有限数：{edge_id}")
        if weight > 0:
            candidates.append({"edge_id": edge_id, "edge": edge, "weight": weight})
    if not candidates:
        raise ValueError("没有正权重的终点候选路段")
    candidates.sort(key=lambda item: item["edge_id"])
    return candidates


def select_destination(candidates, origin_id, rng, constraint=None,
                       exclude_origin=False, cumulative_weights=None):
    """
    param: candidates，build_destination_candidates 返回的候选列表
    param: origin_id，当前车辆起点的 edgeID
    param: rng，random.Random 随机数生成器
    param: constraint，可选约束函数，接收起点 ID 和候选路段字典，返回 True 或 False
    param: exclude_origin，是否禁止终点与起点相同
    param: cumulative_weights，与 candidates 对应的预先计算的累计权重
    return: 随机选中的终点 edgeID
    """
    if constraint is None:
        eligible = candidates
    else:
        eligible = [item for item in candidates if constraint(origin_id, item)]
    if not eligible or (exclude_origin and len(eligible) == 1 and eligible[0]['edge_id'] == origin_id):
        raise ValueError(f"起点 {origin_id} 没有满足约束的终点候选")
    # 自定义约束改变候选列表时，必须重新计算权重。
    if constraint is not None or cumulative_weights is None:
        cumulative_weights = []
        total = 0.0
        for item in eligible:
            total += item['weight']
            cumulative_weights.append(total)
    elif len(cumulative_weights) != len(eligible):
        raise ValueError('累计权重必须与候选路段数量一致')
    while True:
        selected = rng.choices(eligible, cum_weights=cumulative_weights, k=1)[0]['edge_id']
        if not exclude_origin or selected != origin_id:
            return selected
