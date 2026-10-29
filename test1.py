from road_bfs import read_edges, build_successors, build_road_graph, sort_edges_bfs
import os
"""
    测试脚本，试验当前代码是否能够正确读取路网数据、构建有向图并进行BFS排序。
"""
cur_dir=os.path.dirname(os.path.abspath(__file__))
net_path=os.path.join(cur_dir,"luwang\\Beijing_nointernal.net.xml")
edge_map = read_edges(net_path)
successors = build_successors(edge_map)

graph = build_road_graph(edge_map, successors)
result = sort_edges_bfs(graph)

print("图中的路段数量：", graph.number_of_nodes())
print("图中的后继关系数量：", graph.number_of_edges())
print("完成排序的路段数量：", len(result))

print("前 10 条排序结果：")
for item in result[:10]:
    print(item)