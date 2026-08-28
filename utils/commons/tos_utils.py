import io
import logging
import os
import pathlib
import random
import sys
import threading
import time
import traceback
import zipfile
from functools import partial
from multiprocessing import Pool
from queue import Queue

import bytedtos
import tenacity
from tqdm import tqdm

logger = logging.getLogger(__name__)

bucket = 'sa-ag-sg-research-sg'
ak = '*'
psm = 'toutiao.tos.tosapi'
cluster = 'default'
idc = 'my'
client = None
min_part_size = 16 * 1024 * 1024  # 50MB


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=16),
    after=tenacity.after_log(logger, logging.WARN),
    stop=tenacity.stop_after_attempt(5),
)
def list_prefix(*arg, **args):
    client = get_tos_client()
    try:
        resp = client.list_prefix(*arg, **args)
        return resp
    except:
        traceback.print_exc()
        raise


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
    after=tenacity.after_log(logger, logging.WARN),
    stop=tenacity.stop_after_attempt(5),
)
def upload_part(key: str, upload_id: str, part_number: str, data: bytes):
    tos_client = get_tos_client()
    try:
        return tos_client.upload_part(key, upload_id, part_number, data)
    except:
        traceback.print_exc()
        raise


@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
    after=tenacity.after_log(logger, logging.WARN),
    stop=tenacity.stop_after_attempt(5),
)
def put_object(k, v):
    tos_client = get_tos_client()
    total_size = len(v)
    if total_size < min_part_size:
        try:
            k = k.replace('//', '/')
            resp = tos_client.put_object(k, v)
            return resp
        except:
            traceback.print_exc()
            raise
    else:
        init_resp = tos_client.init_upload(k)
        upload_id = init_resp.upload_id
        offset = 0
        part_number = 1
        part_list = []

        while offset < total_size:
            if total_size - offset < 2 * min_part_size:
                upload_size = total_size - offset
            else:
                upload_size = min(min_part_size, total_size - offset)
            resp = upload_part(k, upload_id, part_number, v[offset: offset + upload_size])
            part_list.append(resp.part_number)
            offset += upload_size
            part_number += 1

        comp_resp = tos_client.complete_upload(k, upload_id, part_list)
        return comp_resp


# @tenacity.retry(
#     wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
#     after=tenacity.after_log(logger, logging.WARN),
#     stop=tenacity.stop_after_attempt(5),
#     )
# def get_object(k):
#     client = get_tos_client()
#     try:
#         k = k.replace('//', '/')
#         return client.get_object(k).data
#     except bytedtos.TosException as e:
#         traceback.print_exc()
#         if e.code == 404:
#             print(f"| tos file {k} not found.")
#             return None
#         else:
#             raise e
#     except:
#         traceback.print_exc()
#         raise

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
    after=tenacity.after_log(logger, logging.WARN),
    stop=tenacity.stop_after_attempt(5),
)
def get_object(k):
    client = get_tos_client()
    k = k.replace('//', '/')
    try:
        return client.get_object(k).data
        # pos = 0
        # chunk_size = 10 * 1024 * 1024
        # bytes = b''
        # while True:
        #     resp = client.get_object_range(k, start=pos, end=pos + chunk_size - 1)
        #     bytes += resp.data
        #     if len(resp.data) != chunk_size:
        #         break
        #     pos += chunk_size
        # return bytes
    except bytedtos.TosException as e:
        traceback.print_exc()
        if e.code == 404:
            print(f"| tos file {k} not found.")
            return None
        else:
            raise e
    except:
        traceback.print_exc()
        raise


def check_tos_file_exists(key, use_listprefix=True):
    if use_listprefix:
        client = get_tos_client()
        try:
            resp = client.list_prefix(key, '/', '', 10)
            data = resp.json["payload"]
            objects = data["objects"]  # (单层)目录列表, 对于子目录，用户需要递归列举
            return objects is not None
        except:
            traceback.print_exc()
            return False
    try:
        get_object(key)
        return True
    except bytedtos.TosException as e:
        if e.code == 404:
            return False
        else:
            raise e


def get_tos_client():
    import bytedtos
    global client
    if client is None:
        client = bytedtos.Client(
            bucket, ak, service=psm, cluster=cluster, idc=idc,
            connect_timeout=30, timeout=60)
    return client


def ls_tos(prefix):
    if len(prefix) == 0:
        prefix = '/'
    else:
        if prefix[-1] != '/':
            prefix = prefix + '/'
    delimiter = '/'  # Default delimiter
    max_keys = 1000  # Maximum number of files to list

    # Set the parameters for listing files
    files = []
    dirs = []
    start_after = ''
    while True:
        resp = list_prefix(prefix, delimiter, start_after, max_keys)
        try:
            data = resp.json["payload"]
            commonPrefix = data["commonPrefix"]  # (单层)目录列表, 对于子目录，用户需要递归列举
            objects = data["objects"]  # (单层)目录列表, 对于子目录，用户需要递归列举
            start_after = data["startAfter"]
            if commonPrefix is not None:
                dirs += commonPrefix
            elif objects is not None:
                files += [o['key'] for o in objects]
            else:
                break
        except:
            traceback.print_exc()
            print("| resp: ", resp)
            break
        if start_after == '':
            break
    return files, dirs


def glob_tos(pattern, excludes=None):
    pattern_dir = os.path.dirname(pattern)
    paths, dirs = ls_tos(pattern_dir)
    paths = [p.rstrip('/') for p in paths]
    dirs = [p.rstrip('/') for p in dirs]
    if excludes is not None:
        paths = [x for x in paths if all([e not in x for e in excludes])]
        dirs = [x for x in dirs if all([e not in x for e in excludes])]
    paths = [x for x in paths if pathlib.Path(x).match(pattern)]
    dirs = [x for x in dirs if pathlib.Path(x).match(pattern)]
    return paths, dirs


def threaded_os_walk_worker_tos(queue, queue_output):
    while True:
        dir_path = queue.get()
        if dir_path is None:
            break
        if '*' in dir_path:
            files, dirs = multiprocess_glob_tos(dir_path)
        else:
            files, dirs = ls_tos(dir_path)
        for item in dirs:
            queue.put(item)
        for item in files:
            queue_output.put(item)
        if random.random() < 0.01:
            print(f"qsize: {queue.qsize()}, {dir_path}, {queue_output.qsize()}")
        queue.task_done()


def os_walk_tos(root_dir, num_threads=64):
    queue = Queue()
    queue_output = Queue()
    threads = []

    # 创建worker线程
    for _ in range(num_threads):
        t = threading.Thread(target=threaded_os_walk_worker_tos, args=(queue, queue_output))
        t.start()
        threads.append(t)

    # 将根目录添加到队列中
    queue.put(root_dir)

    while True:
        # Try to yield results as they become available
        try:
            yield queue_output.get(timeout=0.1)
        except:
            time.sleep(1)
            if queue_output.empty() and queue.empty():
                with queue.all_tasks_done:
                    if queue.unfinished_tasks == 0:
                        break
    # 等待队列处理完所有任务
    queue.join()

    # 终止线程
    for _ in range(num_threads):
        queue.put(None)
    for t in threads:
        t.join()

    while not queue_output.empty():
        yield queue_output.get()


def multiprocess_glob_tos(pattern, num_workers=32, excludes=None):
    split_pattern = pattern.split("/")
    recursive_depth = 0  # number of recursive depth
    for split in split_pattern:
        if '*' in split:
            recursive_depth += 1
    if recursive_depth == 0 or (recursive_depth == 1 and '*' in split_pattern[-1]):
        paths, dirs = glob_tos(pattern, excludes)
        return paths, dirs
    else:
        _, dirs = multiprocess_glob_tos('/'.join(split_pattern[:-1]), excludes=excludes)
        args = [f'{d}/{split_pattern[-1]}' for d in dirs]
        if len(args) == 1:
            paths_all, dirs_all = glob_tos(args[0])
        else:
            paths_all = []
            dirs_all = []
            p = Pool(num_workers)
            for paths, dirs in tqdm(p.imap_unordered(
                    partial(glob_tos, excludes=excludes), args),
                    total=len(args), desc=f"globing {pattern}"):
                paths_all += paths
                dirs_all += dirs
        return paths_all, dirs_all

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
    after=tenacity.after_log(logger, logging.WARN),
    stop=tenacity.stop_after_attempt(5)
)
def unzip(zip_path, output_path):
    data_binary = get_object(zip_path)
    if data_binary is None:
        return False
    compressed_bytes_io = io.BytesIO(data_binary)
    file_count = 0
    with zipfile.ZipFile(compressed_bytes_io, mode="r", compression=zipfile.ZIP_DEFLATED) as myzip:
        for file_name in myzip.namelist():
            file_count += 1
            file_path = os.path.join(output_path, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file:
                data = myzip.read(file_name)
                file.write(data)
    return True


def zip_and_upload(basedir, paths, output_tos_path):
    compressed_binary_io = io.BytesIO()
    with zipfile.ZipFile(
            compressed_binary_io,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=0,
    ) as zip:
        for abs_path in paths:
            rel_path = os.path.relpath(abs_path, basedir)
            with open(os.path.realpath(os.path.join(basedir, rel_path)), "rb") as f:
                data = f.read()
                zip.writestr(rel_path, data)
    value = compressed_binary_io.getvalue()
    put_object(output_tos_path, value)


if __name__ == '__main__':
    # files = multiprocess_glob_tos('files_storage/rf123_full/Rf123-2/*/dash*.mp4')
    # print(files)
    # */*/*.mp4/results_v1.zip
    unzip('test/1.zip', 'tmp_test')
    # print(multiprocess_glob_tos(sys.argv[1]))
    # print(ls_tos('files_storage/megaavatar/processed_body_v1/rf123/Rf123/0'))
