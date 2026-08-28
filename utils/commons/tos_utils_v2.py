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
from multiprocessing import dummy
from queue import Queue
import bytedtos
import tenacity
from tqdm import tqdm
import bytedtos
import requests

logger = logging.getLogger(__name__)
min_part_size = 100 * 1024 * 1024  # 50MB

_tos_retry_log_lock = threading.Lock()
_tos_retry_log_last_ts = {}
_tos_retry_log_suppressed = {}


def _make_before_sleep_log_throttled(logger, level=logging.WARN, min_interval_s=60.0):
    def _hook(retry_state):
        if os.environ.get("TOS_TENACITY_RETRY_LOG_DISABLE", "").lower() in ("1", "true", "yes"):
            return

        try:
            interval_s = float(os.environ.get("TOS_TENACITY_RETRY_LOG_INTERVAL_S", str(min_interval_s)))
        except Exception:
            interval_s = float(min_interval_s)

        fn = getattr(retry_state, "fn", None)
        fn_name = None
        if fn is not None:
            fn_name = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', str(fn)))}"
        else:
            fn_name = "unknown"

        now = time.time()
        with _tos_retry_log_lock:
            last = float(_tos_retry_log_last_ts.get(fn_name, 0.0))
            if interval_s > 0 and (now - last) < interval_s:
                _tos_retry_log_suppressed[fn_name] = int(_tos_retry_log_suppressed.get(fn_name, 0)) + 1
                return
            suppressed = int(_tos_retry_log_suppressed.pop(fn_name, 0))
            _tos_retry_log_last_ts[fn_name] = now

        sleep_s = None
        next_action = getattr(retry_state, "next_action", None)
        if next_action is not None:
            sleep_s = getattr(next_action, "sleep", None)

        exc = None
        outcome = getattr(retry_state, "outcome", None)
        if outcome is not None:
            try:
                exc = outcome.exception()
            except Exception:
                exc = None

        exc_summary = repr(exc)
        if exc_summary is None:
            exc_summary = "None"
        if len(exc_summary) > 500:
            exc_summary = exc_summary[:500] + "...(truncated)"

        extra = f" (suppressed {suppressed} retries in last ~{int(interval_s)}s)" if suppressed > 0 else ""
        if sleep_s is not None:
            logger.log(level, f"Retrying {fn_name} in {sleep_s} seconds as it raised {exc_summary}.{extra}")
        else:
            logger.log(level, f"Retrying {fn_name} as it raised {exc_summary}.{extra}")

    return _hook


class TosClient:
    def __init__(self, bucket='sa-ag-sg-research-sg'):
        super().__init__()
        client = None
        if os.environ.get('RUNTIME_IDC_NAME') in ['maliva', 'my2']:
            idc = 'my'
        else:
            idc = 'sg1'
        if bucket == 'sa-ag-sg-research-sg':
            ak = '*'
            psm = 'toutiao.tos.tosapi'
            cluster = 'default'
            self.url_prefix = 'https://tosv-sg.tiktok-row.org/obj/sa-ag-sg-research-sg'
        if bucket == 'videoclip-embeddings-512d-sg':
            ak = '*'
            psm = 'toutiao.tos.tosapi'
            cluster = 'default'
        if bucket == 'humanaigc-ads':
            tos_psm = "toutiao.tos.tosapi"
            tos_cluster = "default"
            ak = "*"
            sk = "*"
            bucket_name = "humanaigc-ads"
            tos_idc = "default"
            endpoint = 'tos-cn-north.byted.org'
            client = bytedtos.Client(bucket_name, bytedtos.StaticCredentials(ak, sk),
                                     service=tos_psm,
                                     cluster=tos_cluster,
                                     endpoint=endpoint,
                                     idc=tos_idc)
            self.url_prefix = 'https://tosv.byted.org/obj/humanaigc-ads'
        if bucket == 'humanaigc-ads-data':
            tos_psm = "toutiao.tos.tosapi"
            tos_cluster = "default"
            ak = "*"
            sk = "*"
            bucket_name = "humanaigc-ads-data"
            tos_idc = "default"
            endpoint = 'tos-cn-north.byted.org'
            client = bytedtos.Client(bucket_name, bytedtos.StaticCredentials(ak, sk),
                                     service=tos_psm,
                                     cluster=tos_cluster,
                                     endpoint=endpoint,
                                     idc=tos_idc)
            self.url_prefix = '*'
        if client is None:
            client = bytedtos.Client(
                bucket, ak, service=psm, cluster=cluster, idc=idc,
                connect_timeout=600, timeout=600)
        self.client = client

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=16),
        after=tenacity.after_log(logger, logging.WARN))
    def _list_prefix(self, *arg, **args):
        client = self.client
        try:
            resp = client.list_prefix(*arg, **args)
            return resp
        except:
            traceback.print_exc()
            raise

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=1),
        after=tenacity.after_log(logger, logging.WARN))
    def upload_part(self, key: str, upload_id: str, part_number: str, data: bytes):
        tos_client = self.client
        try:
            return tos_client.upload_part(key, upload_id, part_number, data)
        except:
            traceback.print_exc()
            raise

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=1),
        after=tenacity.after_log(logger, logging.WARN))
    def put_object(self, k, v):
        tos_client = self.client
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
                resp = self.upload_part(k, upload_id, part_number, v[offset: offset + upload_size])
                part_list.append(resp.part_number)
                offset += upload_size
                part_number += 1

            comp_resp = tos_client.complete_upload(k, upload_id, part_list)
            return comp_resp

    def put_object_return_url(self, k, v):
        self.put_object(k, v)
        url = f"{self.url_prefix}/{k}"
        return url

    # @tenacity.retry(
    #     wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
    #     after=tenacity.after_log(logger, logging.WARN))
    # @tenacity.retry(
    #     wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
    #     before_sleep=tenacity.before_sleep_log(logger, logging.WARN))
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        before_sleep=_make_before_sleep_log_throttled(logger, logging.WARN, min_interval_s=60.0))
    def get_object(self, k, verbose=True):
        client = self.client
        k = k.replace('//', '/')
        try:
            resp = client.get_object(k)
            return resp.data
        except bytedtos.TosException as e:
            if e.code == 404:
                if verbose:
                    print(f"| tos file {k} not found.")
                return None
            else:
                if verbose:
                    traceback.print_exc()
                raise e
        except:
            if verbose:
                traceback.print_exc()
            raise
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def get_object_range(self, k, start, end):
        client = self.client
        k = k.replace('//', '/')
        try:
            resp = client.get_object_range(k, start=start, end=end)
            return resp.data
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

    def check_tos_file_exists(self, key, use_head=True, use_listprefix=True):
        if use_head:
            try:
                resp = self.client.head_object(key)
                if int(resp.headers['Content-Length']) > 0:
                    return True
                return False
            except bytedtos.errors.TosException as e:
                if e.code == 429:
                    time.sleep(1)
                    return self.check_tos_file_exists(key, use_head, use_listprefix)
                return False
            except:
                traceback.print_exc()
                return False
        elif use_listprefix:
            try:
                resp = self._list_prefix(key, '/', '', 10)
                data = resp.json["payload"]
                objects = data["objects"]  # (单层)目录列表, 对于子目录，用户需要递归列举
                return objects is not None
            except:
                traceback.print_exc()
                return False
        try:
            res = self.get_object(key)
            if res is None:
                return False
            else:
                return True
        except bytedtos.TosException as e:
            if e.code == 404:
                return False
            else:
                raise e
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def get_file_size(self, key):
        try:
            resp = self.client.head_object(key)
            return int(resp.headers['Content-Length'])
        except bytedtos.errors.TosException:
            return None
        except:
            traceback.print_exc()
            return None

    def ls_tos(self, prefix):
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
            resp = self._list_prefix(prefix, delimiter, start_after, max_keys)
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

    def glob_tos(self, pattern, excludes=None):
        pattern_dir = os.path.dirname(pattern)
        paths, dirs = self.ls_tos(pattern_dir)
        paths = [p.rstrip('/') for p in paths]
        dirs = [p.rstrip('/') for p in dirs]
        if excludes is not None:
            paths = [x for x in paths if all([e not in x for e in excludes])]
            dirs = [x for x in dirs if all([e not in x for e in excludes])]
        paths = [x for x in paths if pathlib.Path(x).match(pattern)]
        dirs = [x for x in dirs if pathlib.Path(x).match(pattern)]
        return paths, dirs

    def threaded_os_walk_worker_tos(self, queue, queue_output):
        while True:
            dir_path = queue.get()
            if dir_path is None:
                break
            if '*' in dir_path:
                files, dirs = self.multiprocess_glob_tos(dir_path)
            else:
                files, dirs = self.ls_tos(dir_path)
            for item in dirs:
                queue.put(item)
            for item in files:
                queue_output.put(item)
            if random.random() < 0.01:
                print(f"qsize: {queue.qsize()}, {dir_path}, {queue_output.qsize()}")
            queue.task_done()

    def os_walk_tos(self, root_dir, num_threads=64):
        queue = Queue()
        queue_output = Queue()
        threads = []

        # 创建worker线程
        for _ in range(num_threads):
            t = threading.Thread(target=self.threaded_os_walk_worker_tos, args=(queue, queue_output))
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

    def multiprocess_glob_tos(self, pattern, num_workers=32, excludes=None):
        split_pattern = pattern.split("/")
        recursive_depth = 0  # number of recursive depth
        for split in split_pattern:
            if '*' in split:
                recursive_depth += 1
        if recursive_depth == 0 or (recursive_depth == 1 and '*' in split_pattern[-1]):
            paths, dirs = self.glob_tos(pattern, excludes)
            return paths, dirs
        else:
            _, dirs = self.multiprocess_glob_tos('/'.join(split_pattern[:-1]), excludes=excludes)
            args = [f'{d}/{split_pattern[-1]}' for d in dirs]
            if len(args) == 1:
                paths_all, dirs_all = self.glob_tos(args[0])
            else:
                paths_all = []
                dirs_all = []
                p = dummy.Pool(num_workers)
                for paths, dirs in tqdm(p.imap_unordered(
                        partial(self.glob_tos, excludes=excludes), args),
                        total=len(args), desc=f"globing {pattern}"):
                    paths_all += paths
                    dirs_all += dirs
            return paths_all, dirs_all

    def unzip(self, zip_path, output_path):
        data_binary = self.get_object(zip_path)
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

    def zip_and_upload(self, basedir, paths, output_tos_path):
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
        self.put_object(output_tos_path, value)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def delete(self, k):
        self.client.delete_object(k)
        
        
def probe_m4a_duration_via_http(url, max_fetch_mb=8):
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def _get_bytes(url, start=None, end=None, timeout=10):
        # 若 start 和 end 都为 None，取整文件，不要这么做
        headers = {}
        if start is None and end is not None:
            headers["Range"] = f"bytes=-{end}"  # 最后 end 字节
        elif start is not None and end is None:
            headers["Range"] = f"bytes={start}-"  # 从 start 到末尾
        elif start is not None and end is not None:
            headers["Range"] = f"bytes={start}-{end}"
        # 若两者都 None，headers 不加 Range（不推荐）
        r = requests.get(url, headers=headers, timeout=timeout)
        # 期望 206，部分内容
        if r.status_code not in (200, 206):
            r.raise_for_status()
        return r.content, r.status_code, r.headers

    def _parse_duration_from_blob(blob):
        
        def _read_u32(b, o): return int.from_bytes(b[o:o+4], "big")
        def _read_u64(b, o): return int.from_bytes(b[o:o+8], "big")

        # 在二进制块里找 mvhd，并解析 version/timescale/duration
        idx = blob.find(b"mvhd")
        if idx == -1:
            return None
        box_start = idx - 4  # mvhd 前 4 字节是 box size
        if box_start < 0:
            return None
        size32 = _read_u32(blob, box_start)
        # box 头长度：8（常规）或 16（扩展大小 size==1）
        header_len = 16 if size32 == 1 else 8
        # 确保数据足够
        if box_start + header_len + 4 + 24 > len(blob):
            # 不一定失败（v1 需要更多字节），但先做基本检查
            pass
        ver = blob[box_start + header_len]
        # version 后紧跟 3 字节 flags
        if ver == 0:
            ts_off = box_start + header_len + 4 + 8   # timescale offset
            du_off = ts_off + 4
            if du_off + 4 > len(blob):
                return None
            timescale = _read_u32(blob, ts_off)
            duration  = _read_u32(blob, du_off)
        elif ver == 1:
            ts_off = box_start + header_len + 4 + 16
            du_off = ts_off + 4
            if du_off + 8 > len(blob):
                return None
            timescale = _read_u32(blob, ts_off)
            duration  = _read_u64(blob, du_off)
        else:
            return None
        if not timescale:
            return None
        return duration / timescale
    
    # 先尝试抓“头”，再尝试抓“尾”，逐步扩大窗口
    steps = [256*1024, 512*1024, 1*1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024]
    steps = [s for s in steps if s <= max_fetch_mb*1024*1024]
    # 1) 试图从文件头拿到（faststart 情况）
    for sz in steps:
        blob, code, _ = _get_bytes(url, start=0, end=sz-1)
        dur = _parse_duration_from_blob(blob)
        if dur is not None:
            return dur
    # 2) 试图从文件尾拿到（moov 在末尾的常见情况）
    for sz in steps:
        blob, code, _ = _get_bytes(url, start=None, end=sz)  # suffix-range: last sz bytes
        dur = _parse_duration_from_blob(blob)
        if dur is not None:
            return dur
    raise RuntimeError("无法在限定的字节范围内找到 mvhd，尝试增大 max_fetch_mb 或文件可能异常。")


if __name__ == '__main__':
    client = TosClient(bucket='sa-ag-sg-research-sg')
    print(client.check_tos_file_exists('files_storage/megaavatar/raw_clip'))
    client.zip_and_upload('tmp', ['tmp/video.mp4', 'tmp/test_mask.jpg'], 'test/1.zip')
    client.unzip('test/1.zip', 'tmp_test')
    client.put_object('test/1.txt', '1234')
    print(len(client.get_object('test/1.zip')))
    print(client.multiprocess_glob_tos(
        'files_storage/megaavatar/processed_body_241126v1/bilibili230807/10094436/*.mp4/lats_*.zip'))
