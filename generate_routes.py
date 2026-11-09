"""按需求生成出行记录，并用 duarouter 的 A* 算法生成 SUMO 路线。

直接运行本文件，在文件末尾修改路网路径、模拟天数和车辆数量。
中心按现有 Gaussian 模块的路段权重抽样；突发日数量按整个模拟期计算。
生成过程中，无法到达的终点会重新抽取，
重试仍失败时整个任务报错，不会悄悄减少车辆数。
"""
import random
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
import numpy as np
import networkx as nx
from destination import build_destination_candidates, select_destination
from Gaussian import calculate_edge_weights, classify_days, create_patterns
from origins import build_mixture_distribution, sample_origins
from road_bfs import read_edges, build_successors, build_road_graph, sort_edges_bfs


def cars_per_second_avg0(day=14, h1_start=7, h1_end=9, h2_start=17,
                         h2_end=19, total=100000, b1=2, b2=4):
    """
    返回 (仿真秒数, 该秒投放车辆数)。各时段分别向下取整，因此实际
    车辆总数可能略少于 total；这与原函数的累计方式一致。
    """
    max_time = day * 86400 #总时间
    time_high = day * ((h1_end - h1_start) + (h2_end - h2_start)) * 3600 #高峰时间
    time_day = day * (h2_start - h1_end) * 3600 #日间非高峰时间
    time_night = max_time - time_high - time_day #其余时间
    a1 = time_day / time_night #日间非高峰时间占比
    a2 = time_high / time_night #高峰时间占比
    total_day_num = total * a1 * b1 / (1 + a1 * b1 + a2 * b2) #日间非高峰时间车辆数
    total_high_num = total * a2 * b2 / (1 + a1 * b1 + a2 * b2) #高峰时间车辆数
    total_night_num = total - total_day_num - total_high_num #其余时间车辆数
    cars_per_sec_night = total_night_num / time_night #其余时间每秒投放车辆数
    cars_per_sec_high = total_high_num / time_high #高峰时间每秒投放车辆数
    cars_per_sec_day = total_day_num / time_day #日间非高峰时间每秒投放车辆数
    rates = [cars_per_sec_high, cars_per_sec_day, cars_per_sec_high, cars_per_sec_night] #各时段每秒投放车辆数
    run = [0.0, 0.0, 0.0, 0.0]
    elapsed = [0, 0, 0, 0]
    for second in range(max_time):
        t = second % 86400
        if h1_start * 3600 <= t < h1_end * 3600:
            period = 0
        elif h1_end * 3600 <= t < h2_start * 3600:
            period = 1
        elif h2_start * 3600 <= t < h2_end * 3600:
            period = 2
        else:
            period = 3
        old = int(run[period])
        run[period] = elapsed[period] * rates[period] #当前时段已有车辆数
        elapsed[period] += 1 #当前时段已过秒数
        yield second, int(run[period] - old)


def write_trips(path, trips):
    root = ET.Element('routes')
    ET.SubElement(root, 'vType', id='Type1', vClass='passenger', maxSpeed='14')
    for trip in sorted(trips, key=lambda x: (x['depart'], x['id'])):
        ET.SubElement(root, 'trip', id=trip['id'], type='Type1',
                      depart=f"{trip['depart']:.3f}", **{'from': trip['from'], 'to': trip['to']})
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def generate(net_path, trip_path, route_path, day=14, total=100000,
             event_days=1, seed=42, retries=20, threads=1, duarouter='duarouter',
             destination_constraint=None):
    net_path = Path(net_path).resolve()
    trip_path = Path(trip_path).resolve()
    route_path = Path(route_path).resolve()
    if trip_path == route_path or trip_path.exists() or route_path.exists():
        raise ValueError('输出路径必须不同且文件不能已存在')
    if not net_path.is_file():
        raise ValueError(f'路网文件不存在：{net_path}')
    if day < 1 or total < 1 or not 0 <= event_days <= day or retries < 0 or threads < 1:
        raise ValueError('天数和车辆总数须为正，突发日数须在模拟天数范围内，重试次数不能为负')
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    print('读取路网并计算 BFS 排序', flush=True)
    ts0 = time.time()#加载路网开始
    edges = read_edges(str(net_path))
    successors = build_successors(edges)
    graph = build_road_graph(edges, successors)
    ordering = sort_edges_bfs(graph)
    te0 = time.time()#加载路网结束
    print(f'路网加载完成，{len(edges)} 条路段，BFS 排序耗时 {te0 - ts0:.2f} 秒', flush=True)
    ts1 = time.time()#生成模式开始
    patterns = create_patterns(ordering, calculate_edge_weights(edges, successors), seed)
    distributions = {key: build_mixture_distribution(value, ordering) for key, value in patterns.items()}
    te1 = time.time()#生成模式结束
    print(f'生成模式完成，耗时 {te1 - ts1:.2f} 秒', flush=True)
    # 检查每个模式的分布是否有效
    # 无后继的路段无法到达其他终点，从起点分布中剔除并重新归一化。
    for distribution in distributions.values():
        probabilities = distribution['probabilities'].copy()
        for i, edge_id in enumerate(distribution['edge_ids']):
            if not successors[edge_id]:
                probabilities[i] = 0
        if probabilities.sum() <= 0:
            raise ValueError('没有可出发的路段')
        distribution['probabilities'] = probabilities / probabilities.sum()
    days = classify_days(day, event_days, rng)
    day_mode = {day: mode for mode, dates in days.items() for day in dates}
    print('日期模式：', days, flush=True)
    candidates = build_destination_candidates(edges)
    weights = [item['weight'] for item in candidates]
    cumulative = []
    total_weight = 0.0
    for weight in weights:
        total_weight += weight
        cumulative.append(total_weight)

    trips = []
    schedule = [second for second, count in cars_per_second_avg0(day=day, total=total)
                for _ in range(count)]
    print(f'目标投放 {total} 辆，按原累计算法实际投放 {len(schedule)} 辆。', flush=True)
    for date in range(1, day + 1):
        day_departures = [second for second in schedule if (second // 86400) + 1 == date]
        origins = sample_origins(distributions[day_mode[date]], len(day_departures), np_rng)
        for second, origin in zip(day_departures, origins):
            edge_id = origin['edge_id']
            trips.append({'id': str(len(trips)), 'depart': second,
                          'from': edge_id, 'to': select_destination(
                              candidates, edge_id, rng, constraint=destination_constraint,
                              exclude_origin=True, cumulative_weights=cumulative)})

    # 对无路径的车辆保持原起点与出发时间，仅重新选择终点。
    pending = trips[:]
    routed = {}
    reachable_cache = OrderedDict()
    with tempfile.TemporaryDirectory(prefix='sumo_routes_', dir=route_path.parent) as tmp:
        tmp = Path(tmp)
        # 用路段长度作为代价，避免 duarouter 默认按旅行时间求最快路线。
        weights_file = tmp / 'lengths.xml'
        weight_root = ET.Element('meandata')
        interval = ET.SubElement(weight_root, 'interval', begin='0', end=str((day + 1) * 86400))
        for edge_id, edge in edges.items():
            ET.SubElement(interval, 'edge', id=edge_id, traveltime=str(edge.getLength()))
        weight_data = ET.tostring(weight_root, encoding='utf-8', xml_declaration=True)
        del weight_root
        for attempt in range(retries + 1):
            # 外部程序可能清理临时文件；每轮调用前恢复所需的输入文件。
            tmp.mkdir(parents=True, exist_ok=True)
            if not weights_file.is_file():
                weights_file.write_bytes(weight_data)
            batch = tmp / 'batch.trips.xml'
            output = tmp / 'batch.rou.xml'
            write_trips(batch, pending)
            command = [duarouter, '-n', str(net_path), '-r', str(batch), '-o', str(output),
                       '--weight-files', str(weights_file), '--weight-attribute', 'traveltime',
                       '--routing-algorithm', 'astar', '--ignore-errors', '--no-step-log',
                       '--routing-threads', str(threads), '--seed', str(seed)]
            print(f'计算路线，第 {attempt + 1} 轮，待处理 {len(pending)} 辆……', flush=True)
            proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if proc.returncode != 0 or not output.exists():
                raise RuntimeError(f'duarouter 失败（退出码 {proc.returncode}）：\n{proc.stderr[-4000:]}')
            for _, element in ET.iterparse(output, events=('end',)):
                if element.tag == 'vehicle':
                    routed[element.attrib['id']] = ET.tostring(element, encoding='unicode')
                    element.clear()
            pending = [trip for trip in pending if trip['id'] not in routed]
            if not pending:
                break
            if attempt < retries:
                for trip in pending:
                    origin = trip['from']
                    if origin not in reachable_cache:
                        reachable = nx.descendants(graph, origin)
                        options = [item for item in candidates if item['edge_id'] in reachable]
                        reachable_cache[origin] = (options, [item['weight'] for item in options])
                        if len(reachable_cache) > 8:
                            reachable_cache.popitem(last=False)
                    else:
                        reachable_cache.move_to_end(origin)
                    options, option_weights = reachable_cache[origin]
                    if not options:
                        raise RuntimeError(f'起点 {origin} 没有可达的其他候选路段，无法生成路线')
                    trip['to'] = select_destination(
                        options, origin, rng, constraint=destination_constraint,
                        exclude_origin=True)
        if pending:
            raise RuntimeError(f'{len(pending)} 辆车在 {retries} 次重试后仍无路线；例如 {pending[0]}。未生成最终文件。')

        # 最终输出按出发时间排序，保留成功路线对应的 trip。
        trips.sort(key=lambda x: (x['depart'], x['id']))
        trip_tmp = tmp / 'final.trips.xml'
        route_tmp = tmp / 'final.rou.xml'
        write_trips(trip_tmp, trips)
        with route_tmp.open('w', encoding='utf-8', newline='\n') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
            f.write('  <vType id="Type1" vClass="passenger" maxSpeed="14"/>\n')
            for trip in trips:
                f.write('  ' + routed[trip['id']] + '\n')
            f.write('</routes>\n')
        # 独占创建，避免覆盖已有结果。
        for source, target in ((trip_tmp, trip_path), (route_tmp, route_path)):
            with source.open('rb') as src, target.open('xb') as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
    print(f'完成：{len(trips)} 辆；trip={trip_path}；route={route_path}')


if __name__ == '__main__':
    t1=time.time()
    cur_dir = Path(__file__).resolve().parent
    net_path = cur_dir / 'luwang' / 'Beijing_nointernal.net.xml'
    trip_path = cur_dir / 'generated2.trips.xml'
    route_path = cur_dir / 'generated2.rou.xml'
    duarouter_path = 'duarouter'  # 如果未加入 PATH，请填写 duarouter.exe 的完整路径。
    day = 1
    total = 100000  # 整个模拟期的目标车辆总数
    event_days = 1
    seed = 42
    retries = 20
    threads = 1
    generate(net_path, trip_path, route_path, day=day, total=total,
             event_days=event_days, seed=seed, retries=retries,
             threads=threads, duarouter=duarouter_path)
    t2=time.time()
    print(f'总耗时 {t2-t1:.2f} 秒')
    
