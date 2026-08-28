import os
import pathlib
import subprocess
import glob
import multiprocessing
from multiprocessing import dummy
from tqdm import tqdm


def link_file(from_file, to_file):
    subprocess.check_call(
        f'ln -s "`realpath --relative-to="{os.path.dirname(to_file)}" "{from_file}"`" "{to_file}"', shell=True)


def move_file(from_file, to_file):
    subprocess.check_call(f'mv "{from_file}" "{to_file}"', shell=True)


def copy_file(from_file, to_file):
    subprocess.check_call(f'cp -r "{from_file}" "{to_file}"', shell=True)


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


def multiprocess_glob(pattern, num_workers=None):
    split_pattern = pattern.split("/")
    recursive_depth = 0  # number of recursive depth
    for split in split_pattern:
        if '*' in split:
            recursive_depth += 1

    if recursive_depth <= 1:
        return glob.glob(pattern)

    dirs = multiprocess_glob('/'.join(split_pattern[:-1]), num_workers=num_workers)
    args = [f'{d}/{split_pattern[-1]}' for d in dirs]

    if '*' not in split_pattern[-1]:
        return args

    if len(args) == 0:
        return []

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = max(1, min(int(num_workers), len(args)))

    ret = []
    with dummy.Pool(num_workers) as p:
        for res in tqdm(p.imap_unordered(glob.glob, args), total=len(args), desc=f"globing {pattern}"):
            ret.extend(res)
    return ret


def _walk_one_root(root):
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        results.append((dirpath, dirnames, filenames))
    return results

def multiprocess_walk(root, num_workers=None, include_root=True):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    subdirs = []
    with os.scandir(root) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                subdirs.append(entry.path)

    if include_root:
        root_files = []
        dirnames = []
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    dirnames.append(entry.name)
                elif entry.is_file(follow_symlinks=False):
                    root_files.append(entry.name)
        yield (root, dirnames, root_files)

    with dummy.Pool(num_workers) as p:
        for sub_res in tqdm(
            p.imap_unordered(_walk_one_root, subdirs),
            total=len(subdirs),
            desc=f"walking {root}",
        ):
            for item in sub_res:
                yield item


def glob_hdfs(pattern):
    split_path = pattern.split("/")
    assert sum([1 for s in pattern if s == '*']) == 1, pattern
    assert '*' in split_path[-1]
    pattern_dir = os.path.dirname(pattern)
    paths = subprocess.check_output(
        f'hdfs dfs -ls {pattern_dir}', shell=True).decode().strip().split("\n")[1:]
    paths = [x.split(" ")[-1] for x in paths]
    paths = [x for x in paths if pathlib.Path(x).match(pattern)]
    return paths


def multiprocess_glob_hdfs(pattern, num_workers=None):
    split_pattern = pattern.split("/")
    recursive_depth = 0  # number of recursive depth
    for split in split_pattern:
        if '*' in split:
            recursive_depth += 1
    if recursive_depth == 1 and '*' in split_pattern[-1]:
        return glob_hdfs(pattern)
    else:
        dirs = multiprocess_glob_hdfs('/'.join(split_pattern[:-1]))
        ret = []
        args = [f'{d}/{split_pattern[-1]}' for d in dirs]
        if '*' not in split_pattern[-1]:
            return args
        p = dummy.Pool()
        for res in tqdm(p.imap_unordered(glob_hdfs, args), total=len(args), desc=f"globing {pattern}"):
            ret += res
        return ret


def kill_void():
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(devices) > 0:
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
        for d in devices:
            os.system(f'pkill -f "voidgpu{d}" &')
    # os.system(f'pkill -f user/void.py &')
            os.system(f'pkill -f gpu_mm{d}')


def handle_exacption(err, name='', verbose=True):
    import sys, traceback
    _, exc_value, exc_tb = sys.exc_info()
    tb = traceback.extract_tb(exc_tb)[-1]
    name = f" {name}" if name is not None and name != '' else ''
    if verbose:
        print(f'skip{name}, {err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
    return


def load_env_local():
    if os.path.isfile('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
