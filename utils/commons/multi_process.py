import multiprocessing as mp
from functools import wraps
import tqdm
import glob
import pandas as pd
import os
import json

def run_on_all_parquets(num_processes=32, parquet_pattern=None):
    """
    一个装饰器，用于修饰一个“工作函数”。
    当调用被修饰的函数时，它会自动从指定的 Parquet 路径加载数据，
    并使用多进程池将数据行交给被修饰的函数进行处理。
    """
    assert parquet_pattern is not None, '需要parquetst'

    def decorator(worker_func):
        @wraps(worker_func)
        def wrapper(*args, **kwargs):
            # --- 1. 数据准备阶段 (硬编码在装饰器内部) ---
            print("开始执行标准数据加载流程...")
            parquet_paths = glob.glob(parquet_pattern)
            valid_rows = []
            
            print("开始加载并筛选所有 Parquet 文件...")
            for file_path in tqdm.tqdm(parquet_paths, desc="加载和筛选 Parquet 文件"):
                try:
                    df = pd.read_parquet(file_path)
                    # 使用 to_dict('records') 以获得更好的性能
                    # valid_rows.extend(df.to_dict('records'))
                    for index, row in df.iterrows():
                        valid_rows.append(row)
                except Exception as e:
                    print(f"加载或筛选文件 {file_path} 失败: {e}")
            
            if not valid_rows:
                print("数据准备函数没有返回任何任务，处理结束。")
                return []
            
            total_tasks = len(valid_rows)
            print(f"数据准备完毕，总计 {total_tasks} 个任务需要处理。")

            # --- 2. 多进程处理阶段 ---
            print(f"启动 {num_processes} 个工作进程，使用 '{worker_func.__name__}' 函数进行处理...")
            all_results = []
            
            with mp.Pool(processes=num_processes) as pool:
                # 这里的关键是，将 `worker_func` (即被装饰的原始函数) 作为工作目标
                results_iterator = pool.imap_unordered(worker_func, valid_rows)
                
                for result in tqdm.tqdm(results_iterator, total=total_tasks, desc=f"使用 {worker_func.__name__} 处理"):
                    all_results.append(result)

            print("\n所有任务处理完成！")
            return all_results

        return wrapper
    return decorator

def parallel_process_parquets(worker_func, parquet_paths, num_processes=32, truncate=False):
    """
    一个通用的高阶函数，用于从标准路径加载数据，并使用指定的工作函数并行处理。
    Args:
        worker_func (function): 一个在模块顶层定义的、可被 pickle 的工作函数。
        num_processes (int, optional): 进程数。
    """
    
    # --- 1. 数据准备阶段 ---
    print("开始执行标准数据加载流程...")
    # parquet_paths = glob.glob('/mnt/bn/genai-data2/renyi/entries/interns_env/leike/hq_audio_caption_relevant_token_0908/*_caption.parquet')
    # parquet_paths = glob.glob('/mnt/bn/genai-data2/renyi/entries/interns_env/leike/lizhe_audio_data/hq_tos_9m_data.p*')
    valid_rows = []
    print("开始加载并筛选所有 Parquet 文件...")
    for file_path in tqdm.tqdm(parquet_paths, desc="加载和筛选 Parquet 文件"):
        try:
            df = pd.read_parquet(file_path)
            valid_rows.extend(df.to_dict('records'))
        except Exception as e:
            print(f"加载或筛选文件 {file_path} 失败: {e}")
    if not valid_rows:
        print("没有找到任何有效的需要处理的数据行。")
        return []
    if truncate:
        if isinstance(truncate, list):
            valid_rows = valid_rows[truncate[0]: truncate[1]]
            print(f"截断数据，{truncate =}")
    total_tasks = len(valid_rows)
    print(f"数据准备完毕，总计 {total_tasks} 个任务需要处理。")
    # --- 2. 多进程处理阶段 ---
    print(f"启动 {num_processes} 个工作进程，使用 '{worker_func.__name__}' 函数进行处理...")
    all_results = []
    with mp.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_func, valid_rows)
        for result in tqdm.tqdm(results_iterator, total=total_tasks, desc=f"使用 {worker_func.__name__} 处理"):
            all_results.append(result)
    print("\n所有任务处理完成！")
    return all_results


def parallel_process_json(worker_func, json_paths, num_processes=32):
    """
    一个通用的高阶函数，用于从标准路径加载数据，并使用指定的工作函数并行处理。
    Args:
        worker_func (function): 一个在模块顶层定义的、可被 pickle 的工作函数。
        num_processes (int, optional): 进程数。
    """
    
    # --- 1. 数据准备阶段 ---
    print("开始执行标准数据加载流程...")
    # parquet_paths = glob.glob('/mnt/bn/genai-data2/renyi/entries/interns_env/leike/hq_audio_caption_relevant_token_0908/*_caption.parquet')
    # parquet_paths = glob.glob('/mnt/bn/genai-data2/renyi/entries/interns_env/leike/lizhe_audio_data/hq_tos_9m_data.p*')
    valid_rows = []
    print("开始加载并筛选所有 Parquet 文件...")
    for file_path in tqdm.tqdm(json_paths, desc="加载和筛选 Parquet 文件"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_lst = json.load(f)
            # df = pd.read_parquet(file_path)
            # valid_rows.extend(df.to_dict('records'))
            valid_rows.extend(data_lst)
        except Exception as e:
            print(f"加载或筛选文件 {file_path} 失败: {e}")
    if not valid_rows:
        print("没有找到任何有效的需要处理的数据行。")
        return []
    total_tasks = len(valid_rows)
    print(f"数据准备完毕，总计 {total_tasks} 个任务需要处理。")
    # --- 2. 多进程处理阶段 ---
    print(f"启动 {num_processes} 个工作进程，使用 '{worker_func.__name__}' 函数进行处理...")
    all_results = []
    with mp.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_func, valid_rows)
        for result in tqdm.tqdm(results_iterator, total=total_tasks, desc=f"使用 {worker_func.__name__} 处理"):
            all_results.append(result)
    print("\n所有任务处理完成！")
    return all_results

def parallel_process(worker_func, args_list, num_processes=32):
    """
    一个通用的高阶函数，用于从标准路径加载数据，并使用指定的工作函数并行处理。
    Args:
        worker_func (function): 一个在模块顶层定义的、可被 pickle 的工作函数。
        num_processes (int, optional): 进程数。
    """
    # --- 2. 多进程处理阶段 ---
    num_processes = min(num_processes, len(args_list))
    print(f"启动 {num_processes} 个工作进程，共 {len(args_list)} 任务，使用 '{worker_func.__name__}' 函数进行处理...")
    all_results = []
    with mp.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(worker_func, args_list)
        for result in tqdm.tqdm(results_iterator, desc=f"使用 {worker_func.__name__} 处理"):
            if result is not None:
                all_results.append(result)
    print("\n所有任务处理完成！")
    return all_results

def parallel_process_map(worker_func, args_list, num_processes=32):
    """
    一个通用的高阶函数，用于并行处理数据，使用 pool.map 方法。
    结果会严格按照输入 args 的顺序返回。
    Args:
        worker_func (function): 一个在模块顶层定义的、可被 pickle 的工作函数。
        args (iterable): 传递给 worker_func 的参数列表。
        num_processes (int, optional): 进程数。默认为 32。
    """
    # --- 1. 确定进程数 ---
    # 保持原始逻辑，但通常我们会限制进程数，而不是让它等于参数数量
    # 这里的逻辑可以根据实际需要调整，例如：
    # num_processes = min(num_processes, len(args))
    # 或者直接使用传入的 num_processes
    num_processes = min(num_processes, len(args_list))
    print(f"启动 {num_processes} 个工作进程，使用 '{worker_func.__name__}' 函数进行处理...")
    # --- 2. 多进程处理阶段 ---
    with mp.Pool(processes=num_processes) as pool:
        # pool.map 是一个阻塞操作。它会分发所有任务，等待它们全部完成，
        # 然后一次性返回一个包含所有结果的列表。
        # 结果的顺序与输入 'args' 的顺序完全一致。
        all_results = list(tqdm.tqdm(pool.map(worker_func, args_list), total=len(args_list), desc=f"使用 {worker_func.__name__} 处理"))
    filtered_results = [result for result in all_results if result is not None]
    print("\n所有任务处理完成！")
    return filtered_results

def parallel_process_multiargs(worker_func, args_list, num_processes=32):
    """
    使用 starmap 的多进程函数。 如果返回值是None，会被丢掉
    Args:
        worker_func (function): 工作函数。
        args_list (list of tuples): 参数元组的列表，每个元组对应一次函数调用。
        num_processes (int): 进程数。
    """
    num_processes = min(num_processes, len(args_list))
    print(f"启动 {num_processes} 个工作进程，使用 '{worker_func.__name__}' 函数进行处理...")
    all_results = []
    with mp.Pool(processes=num_processes) as pool:
        # 使用 starmap，它会自动解包元组
        # 注意：starmap 没有 imap_unordered 的对应版本，但可以通过 starmap_async 模拟
        # 这里为了简单，直接用 starmap，它会按顺序返回结果
        results_iterator = pool.starmap(worker_func, args_list)
        # tqdm 可以直接包装 starmap 的结果
        for result in tqdm.tqdm(results_iterator, total=len(args_list), desc=f"使用 {worker_func.__name__} 处理"):
            if result is not None:
                all_results.append(result)
    print("\n所有任务处理完成！")
    return all_results