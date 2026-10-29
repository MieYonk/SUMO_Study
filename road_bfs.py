import sumolib
from collections import deque
import networkx as nx
#读取路网数据
def read_edges(net_path):
    net = sumolib.net.readNet(net_path)
    edges = net.getEdges()
    edge_map = {}
    for edge in edges:
        if edge.allows("passenger"):#只筛选能通过客车的路段
            edge_id = edge.getID()
            edge_map[edge_id] = edge
    return edge_map
#获取后继路段
def build_successors(edge_map):
    """
    param: edge_map，路段字典，格式为 {edgeID: 路段对象}
    return: successors，后继路段字典，格式为 {edgeID: 后继edgeID列表}
    """
    successors = {}

    for edge_id, edge in edge_map.items():
        next_ids = []

        # 获取当前路段的后继路段
        for next_edge in edge.getOutgoing():
            next_id = next_edge.getID()
            # 只保留已经读取到的路段
            if next_id in edge_map:
                next_ids.append(next_id)

        # 去重，并按字符串顺序排列
        successors[edge_id] = sorted(set(next_ids))

    return successors
#生成有向图
def build_road_graph(edge_map, successors):
    """
    param: edge_map，筛选后允许 passenger 通行的路段字典
    param: successors，每条路段允许 passenger 通行的后继路段字典
    return: graph，以路段为节点、后继关系为有向边的图
    """
    # 创建有向图
    graph = nx.DiGraph()
    # 先加入全部路段，包括没有后继的路段
    graph.add_nodes_from(edge_map.keys())

    # 再加入路段之间的后继关系
    for edge_id, next_ids in successors.items():
        for next_id in next_ids:
            graph.add_edge(edge_id, next_id)

    return graph
#使用BFS算法为路段排序
def sort_edges_bfs(graph):
    """
    param: graph，由路段和有向后继关系组成的图
    return: result，包含排序号、路段ID和本轮起点的列表
    """
    visited = set()
    result = []

    # 固定起点选择顺序
    all_edge_ids = sorted(graph.nodes)

    for start_id in all_edge_ids:
        # 如果当前路段已经被访问过，则跳过
        if start_id in visited:
            continue

        # 本轮只在尚未编号的路段中搜索
        remaining = set(graph.nodes) - visited
        current_graph = graph.subgraph(remaining)

        # bfs_edges 不会单独返回起点，所以先记录起点
        visited.add(start_id)
        result.append({
            "rank": len(result) + 1,
            "edge_id": start_id,
            "start_edge_id": start_id,
        })

        # 按后继方向进行 BFS；同层后继按 ID 排序
        for parent_id, next_id in nx.bfs_edges(
            current_graph,
            source=start_id,
            sort_neighbors=sorted,
        ):
            visited.add(next_id)
            result.append({
                "rank": len(result) + 1,
                "edge_id": next_id,
                "start_edge_id": start_id,
            })

    return result