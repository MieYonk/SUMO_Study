"""按需求生成出行记录，并用 duarouter 的 A* 算法生成 SUMO 路线。

直接运行本文件，在文件末尾修改路网路径、模拟天数和车辆数量。
中心按路段权重抽样；突发日数量按整个模拟期计算。
失败车辆先按原规则重新抽取起终点，达到指定次数后再筛选可达终点。
重试仍失败时整个任务报错，不会悄悄减少车辆数。
"""
import random
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import networkx as nx
from road_bfs import read_edges, build_successors, build_road_graph, sort_edges_bfs
from origins import build_mixture_distribution, sample_origins,calculate_edge_weights, classify_days, create_patterns
from destination import (build_destination_candidates, build_destination_cumulative_weights,
                         select_destination)

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
             destination_constraint=None, random_retries=3):
    """
    retries 是首次求路之外的总重试次数。
    random_retries 是先重新抽取起点和终点的次数，不包含首次求路；
    后续重试使用可达性预处理。设为 0 时直接使用预处理。
    """
    net_path = Path(net_path).resolve()
    trip_path = Path(trip_path).resolve()
    route_path = Path(route_path).resolve()
    if trip_path == route_path or trip_path.exists() or route_path.exists():
        raise ValueError('输出路径必须不同且文件不能已存在')
    if not net_path.is_file():
        raise ValueError(f'路网文件不存在：{net_path}')
    for name, value in (('retries', retries), ('random_retries', random_retries)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f'{name} 必须是非负整数')
    if random_retries > 0 and random_retries >= retries:
        raise ValueError('random_retries 必须小于总重试次数 retries，至少给预处理留一次求路机会')
    if day < 1 or total < 1 or not 0 <= event_days <= day or threads < 1:
        raise ValueError('天数和车辆总数须为正，突发日数须在模拟天数范围内，线程数须为正')
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    print('读取路网并计算 BFS 排序', flush=True)
    ts0 = time.perf_counter()#加载路网开始
    edges = read_edges(str(net_path))
    successors = build_successors(edges)
    graph = build_road_graph(edges, successors)
    ordering = sort_edges_bfs(graph)
    te0 = time.perf_counter()#加载路网结束
    print(f'路网加载完成，{len(edges)} 条路段，BFS 排序耗时 {te0 - ts0:.2f} 秒', flush=True)
    ts1 = time.perf_counter()#生成模式开始
    patterns = create_patterns(ordering, calculate_edge_weights(edges, successors), seed)
    distributions = {key: build_mixture_distribution(value, ordering) for key, value in patterns.items()}
    te1 = time.perf_counter()#生成模式结束
    print(f'生成模式完成，耗时 {te1 - ts1:.2f} 秒', flush=True)
    tp0 = time.perf_counter()#车辆起终点生成开始
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
    # 默认只排除起点，提前计算一次累计权重供所有车辆复用。
    # 自定义约束仍走原来的逐车筛选方式。
    cumulative_weights = (build_destination_cumulative_weights(candidates)
                          if destination_constraint is None else None)

    def allowed_destination(origin_id, candidate):
        # 排除起点；如果调用方提供其他约束，也一起检查。
        return (candidate['edge_id'] != origin_id and
                (destination_constraint is None or destination_constraint(origin_id, candidate)))

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
                              candidates, edge_id, rng, constraint=allowed_destination,
                              cum_weights=cumulative_weights)})
    print(f'车辆起终点生成完成，耗时 {time.perf_counter() - tp0:.2f} 秒', flush=True)

    # 仅重试尚无路线的车辆：先重抽起终点，之后保留最新起点并筛选可达终点。
    # 两个阶段都保留车辆 ID 和原计划出发时间。
    pending = trips[:]
    routed = {}
    print(f'重试策略：先快速重抽起终点 {random_retries} 次，再使用可达性预处理；'
          f'首次求路之外最多重试 {retries} 次', flush=True)
    tds=time.perf_counter()#生成路线开始
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
            print(f'计算路线，第 {attempt + 1} 轮，待处理 {len(pending)} 辆……', flush=True)
            tw0 = time.perf_counter()
            write_trips(batch, pending)
            print(f'第 {attempt + 1} 轮：临时 trip 写入耗时 {time.perf_counter() - tw0:.2f} 秒', flush=True)
            command = [duarouter, '-n', str(net_path), '-r', str(batch), '-o', str(output),
                       '--weight-files', str(weights_file), '--weight-attribute', 'traveltime',
                       '--routing-algorithm', 'astar', '--ignore-errors', '--no-step-log',
                       '--routing-threads', str(threads), '--seed', str(seed)]
            tr0 = time.perf_counter()
            proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
            print(f'第 {attempt + 1} 轮：duarouter 执行耗时 {time.perf_counter() - tr0:.2f} 秒', flush=True)
            if proc.returncode != 0 or not output.exists():
                raise RuntimeError(f'duarouter 失败（退出码 {proc.returncode}）：\n{proc.stderr[-4000:]}')
            tx0 = time.perf_counter()
            for _, element in ET.iterparse(output, events=('end',)):
                if element.tag == 'vehicle':
                    routed[element.attrib['id']] = ET.tostring(element, encoding='unicode')
                    element.clear()
            pending = [trip for trip in pending if trip['id'] not in routed]
            print(f'第 {attempt + 1} 轮：路线 XML 读取耗时 {time.perf_counter() - tx0:.2f} 秒，'
                  f'剩余 {len(pending)} 辆未得到路线', flush=True)
            if not pending:
                break
            if attempt < retries and attempt < random_retries:
                tq0 = time.perf_counter()
                pending_by_date = {}
                for trip in pending:
                    date = int(trip['depart'] // 86400) + 1
                    pending_by_date.setdefault(date, []).append(trip)
                for date, date_trips in pending_by_date.items():
                    # 出发日期不变，重抽起点仍使用当天原有的工作日/周末/突发分布。
                    origins = sample_origins(distributions[day_mode[date]], len(date_trips), np_rng)
                    for trip, origin in zip(date_trips, origins):
                        edge_id = origin['edge_id']
                        destination_id = select_destination(
                            candidates, edge_id, rng, constraint=allowed_destination,
                            cum_weights=cumulative_weights)
                        trip['from'] = edge_id
                        trip['to'] = destination_id
                print(f'快速重抽起终点第 {attempt + 1}/{random_retries} 次：'
                      f'已为 {len(pending)} 辆失败车辆重新抽样，'
                      f'耗时 {time.perf_counter() - tq0:.2f} 秒，下一轮重新求路', flush=True)
                continue
            if attempt < retries:
                print(f'快速重抽阶段结束，对仍失败的 {len(pending)} 辆车使用可达性预处理', flush=True)
                # 同一起点集中处理，每轮只搜索一次可达路段并准备一次累计权重。
                pending_by_origin = {}
                for trip in pending:
                    pending_by_origin.setdefault(trip['from'], []).append(trip)
                print(f'第 {attempt + 1} 轮：按 {len(pending_by_origin)} 个起点分组重选终点……', flush=True)
                reachable_seconds = 0.0
                sampling_seconds = 0.0
                preprocess_start = time.perf_counter()
                progress_time = preprocess_start
                processed_trips = 0
                for group_index, (origin, origin_trips) in enumerate(pending_by_origin.items(), 1):
                    tg0 = time.perf_counter()
                    reachable = nx.descendants(graph, origin)
                    options = [item for item in candidates if item['edge_id'] in reachable]
                    if not options:
                        raise RuntimeError(f'起点 {origin} 没有可达的其他候选路段，无法生成路线')
                    option_cumulative_weights = (build_destination_cumulative_weights(options)
                                                 if destination_constraint is None else None)
                    reachable_seconds += time.perf_counter() - tg0
                    tg1 = time.perf_counter()
                    for trip in origin_trips:
                        trip['to'] = select_destination(
                            options, origin, rng, constraint=allowed_destination,
                            cum_weights=option_cumulative_weights)
                    sampling_seconds += time.perf_counter() - tg1
                    processed_trips += len(origin_trips)
                    now = time.perf_counter()
                    if now - progress_time >= 5 or group_index == len(pending_by_origin):
                        print(f'预处理进度：{group_index}/{len(pending_by_origin)} 个起点，'
                              f'{processed_trips}/{len(pending)} 辆车，'
                              f'已耗时 {now - preprocess_start:.2f} 秒', flush=True)
                        progress_time = now
                print(f'第 {attempt + 1} 轮：可达性搜索及候选准备耗时 {reachable_seconds:.2f} 秒，'
                      f'终点重选耗时 {sampling_seconds:.2f} 秒', flush=True)
        if pending:
            raise RuntimeError(f'{len(pending)} 辆车在 {retries} 次重试后仍无路线'
                               f'（其中前 {random_retries} 次重抽起终点）；'
                               f'例如 {pending[0]}。未生成最终文件。')
        # 最终输出按出发时间排序，保留成功路线对应的 trip。
        trips.sort(key=lambda x: (x['depart'], x['id']))
        trip_tmp = tmp / 'final.trips.xml'
        route_tmp = tmp / 'final.rou.xml'
        tde=time.perf_counter()#生成路线结束
        print(f'生成路线完成，耗时 {tde - tds:.2f} 秒，成功生成 {len(trips)} 辆路线', flush=True)
        tout0 = time.perf_counter()
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
    print(f'最终文件写出与临时目录清理耗时 {time.perf_counter() - tout0:.2f} 秒', flush=True)
    print(f'完成：{len(trips)} 辆；trip={trip_path}；route={route_path}')


if __name__ == '__main__':
    t1=time.perf_counter()
    cur_dir = Path(__file__).resolve().parent
    net_path = cur_dir / 'luwang' / 'Beijing_nointernal.net.xml'
    trip_path = cur_dir / '路由文件'/'14day_1event_100000cars_hybrid.trips.xml'
    route_path = cur_dir / '路由文件'/'14day_1event_100000cars_hybrid.rou.xml'
    duarouter_path = 'duarouter'  # 如果未加入 PATH，请填写 duarouter.exe 的完整路径。
    day = 14
    total = 100000  # 整个模拟期的目标车辆总数
    event_days = 1
    seed = 42
    retries = 20
    random_retries = 3  # n：失败车辆先重新抽取起点和终点的次数，须小于 retries；0 表示直接预处理。
    threads = 16
    generate(net_path, trip_path, route_path, day=day, total=total,
             event_days=event_days, seed=seed, retries=retries,
             threads=threads, duarouter=duarouter_path, random_retries=random_retries)
    t2=time.perf_counter()
    print(f'总耗时 {t2-t1:.2f} 秒')
    
