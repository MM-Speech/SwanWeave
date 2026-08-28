import io
import logging
import os
import pathlib
import random
import threading
import time
import traceback
import zipfile
from functools import partial
from multiprocessing import dummy
from queue import Queue
import tenacity
from tqdm import tqdm
from pyarrow import fs

from utils.commons.io import print_once

logger = logging.getLogger(__name__)


class HDFSClient:
    def __init__(self, base_dir='/home/byte_ads_videogen/user/renyi/megaavatar', namespace=None):
        super().__init__()
        if namespace is None:
            cluster = os.environ.get('CLUSTER')
            if cluster is not None and cluster.lower() in ['lq', 'hl', 'sg', 'va']:
                if cluster.lower() in ['lq', 'hl']:
                    namespace = 'haruna'
                elif cluster.lower() == 'sg':
                    namespace = 'harunasg'
                elif cluster.lower() == 'va':
                    namespace = 'harunava'
                else:
                    namespace = 'harunasg'
            else:
                namespace = 'harunasg'
        self.client = fs.HadoopFileSystem(namespace)
        self.namespace = namespace
        self.base_dir = f"hdfs://{namespace}{base_dir}"

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=16),
        after=tenacity.after_log(logger, logging.WARN))
    def _list_prefix(self, d, **args):
        d = self.to_abs_path(d)
        selector = fs.FileSelector(d)  # Avoid deep recursion
        files = self.client.get_file_info(selector)
        return files

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def put_object(self, k, v):
        k = self.to_abs_path(k)
        # print("| put file to hdfs path: ", f'hdfs://{self.namespace}{k}')
        self.client.create_dir(os.path.dirname(k))
        with self.client.open_output_stream(k) as hdfs_file:
            if isinstance(v, str):
                v = v.encode('utf-8')
            hdfs_file.write(v)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def get_object(self, k):
        k = self.to_abs_path(k)
        try:
            with self.client.open_input_stream(k) as hdfs_file:
                data = hdfs_file.read()
            return data
        except:
            traceback.print_exc()
            return None

    def check_file_exists(self, k):
        k = self.to_abs_path(k)
        file_info = self.client.get_file_info(k)
        return file_info.is_file

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=8),
        after=tenacity.after_log(logger, logging.WARN))
    def delete(self, k):
        k = self.to_abs_path(k)
        try:
            self.client.delete_file(k)
        except Exception as e:
            logger.warning(f"Failed to delete {k}: {e}")

    def ls_tos(self, prefix):
        prefix = self.to_abs_path(prefix)

        if len(prefix) == 0:
            prefix = '/'
        else:
            if prefix[-1] != '/':
                prefix = prefix + '/'

        files = []
        dirs = []

        file_infos = self._list_prefix(prefix)
        for file_info in file_infos:
            if file_info.type == fs.FileType.File:
                files.append(self.trim_port(file_info.path))
            elif file_info.type == fs.FileType.Directory:
                dirs.append(self.trim_port(file_info.path))

        return files, dirs

    def unzip(self, zip_path, output_path):
        zip_path = self.to_abs_path(zip_path)
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

    def zip_and_upload(self, basedir, paths, output_hdfs_path):
        output_hdfs_path = self.to_abs_path(output_hdfs_path)
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
        self.put_object(output_hdfs_path, value)

    def glob(self, pattern, excludes=None):
        pattern = self.to_abs_path(pattern)
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
                files, dirs = self.multiprocess_glob(dir_path)
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
        root_dir = self.to_abs_path(root_dir)

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

    def multiprocess_glob(self, pattern, num_workers=32, excludes=None):
        pattern = self.to_abs_path(pattern)
        split_pattern = pattern.split("/")
        recursive_depth = 0  # number of recursive depth
        for split in split_pattern:
            if '*' in split:
                recursive_depth += 1
        if recursive_depth == 0 or (recursive_depth == 1 and '*' in split_pattern[-1]):
            paths, dirs = self.glob(pattern, excludes)
            return paths, dirs
        else:
            _, dirs = self.multiprocess_glob('/'.join(split_pattern[:-1]), excludes=excludes)
            args = [f'{d}/{split_pattern[-1]}' for d in dirs]
            if len(args) == 1:
                paths_all, dirs_all = self.glob(args[0])
            else:
                paths_all = []
                dirs_all = []
                p = dummy.Pool(num_workers)
                for paths, dirs in tqdm(p.imap_unordered(
                        partial(self.glob, excludes=excludes), args),
                        total=len(args), desc=f"globing {pattern}"):
                    paths_all += paths
                    dirs_all += dirs
            return paths_all, dirs_all

    def to_abs_path(self, path):
        if path.startswith('hdfs://'):
            path = path[len(self.namespace) + len('hdfs://'):]
        if path[0] != '/':
            return self.base_dir + '/' + path
        else:
            return path

    def trim_port(self, p):
        if p[0] == ':':
            p = '/' + '/'.join(p.split('/')[1:])
        return p


if __name__ == '__main__':
    client = HDFSClient()
    print(client.check_file_exists('hdfs://harunasg/home/byte_ads_videogen/user/renyi/megaavatar/raw_clip'))
    client.zip_and_upload('tmp', ['tmp/1.mp4', 'tmp/1.png'], 'test/1.zip')
    client.unzip('test/1.zip', 'tmp_test')
    client.put_object('test/1.txt', '1234')
    print(len(client.get_object('test/1.zip')))
    print(client.multiprocess_glob('test/1.*'))
