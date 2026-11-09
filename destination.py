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


def select_destination(candidates, origin_id, rng, constraint=None):
    """
    param: candidates，build_destination_candidates 返回的候选列表
    param: origin_id，当前车辆起点的 edgeID
    param: rng，random.Random 随机数生成器
    param: constraint，可选约束函数，接收起点 ID 和候选路段字典，返回 True 或 False
    return: 随机选中的终点 edgeID
    """
    if constraint is None:
        eligible = candidates
    else:
        eligible = [item for item in candidates if constraint(origin_id, item)]
    if not eligible:
        raise ValueError(f"起点 {origin_id} 没有满足约束的终点候选")
    return rng.choices(eligible, weights=[item["weight"] for item in eligible], k=1)[0]["edge_id"]
