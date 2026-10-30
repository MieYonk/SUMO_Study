import os
from road_bfs import read_edges, build_successors
from Gaussian import calculate_edge_weights
cur_dir=os.path.dirname(os.path.abspath(__file__))
net_path=os.path.join(cur_dir,"luwang\\Beijing_nointernal.net.xml")
edge_map = read_edges(net_path)
successors = build_successors(edge_map)

edge_weights = calculate_edge_weights(edge_map, successors)

print("完成权重计算的路段数量：", len(edge_weights))

# 查看一条路段的计算结果
edge_id = "--160#0"

if edge_id in edge_weights:
    print("当前路段：", edge_id)
    print("权重信息：", edge_weights[edge_id])

zero_count = 0

for info in edge_weights.values():
    if info["weight"] == 0:
        zero_count += 1

print("权重为 0 的路段数量：", zero_count)