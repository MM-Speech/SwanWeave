import os
import struct
import numpy as np
import traceback
import mmap
import json
from pathlib import Path
from typing import List, Tuple, Literal, Dict, Union, Any, Optional


ParserName = Literal['orjson', 'simdjson', 'json']
OnErrorMode = Literal['raise', 'none']


def _get_parser_func(parser: ParserName):
    if parser == 'orjson':
        import orjson
        return orjson.loads
    elif parser == 'json':
        import json
        return json.loads
    elif parser == 'simdjson':
        import simdjson
        simd_parser = simdjson.Parser()
        return simd_parser.parse
    else:
        raise ValueError(f"unknown parser: {parser}")


def _build_error_info(
    exc: Exception,
    line_no: Optional[int] = None,
    byte_start: Optional[int] = None,
    raw_line: Optional[bytes] = None,
    parser: Optional[str] = None,
) -> Dict[str, Any]:
    info = {
        'error_type': type(exc).__name__,
        'error_message': str(exc),
    }
    if parser is not None:
        info['parser'] = parser
    if line_no is not None:
        info['line_no'] = line_no
    if byte_start is not None:
        info['byte_start'] = byte_start
    if raw_line is not None:
        preview = raw_line[:200]
        info['raw_preview'] = preview.decode('utf-8', errors='replace')
        info['raw_length'] = len(raw_line)
    return info


def _parse_json_line(
    line: bytes,
    parser: ParserName,
    on_error: OnErrorMode = 'raise',
    error_value: Any = None,
    line_no: Optional[int] = None,
    byte_start: Optional[int] = None,
    return_error: bool = False,
):
    line = line.rstrip(b'\r\n')
    if line.strip() == b'':
        if return_error:
            return None, None
        return None

    parser_func = _get_parser_func(parser)
    try:
        result = parser_func(line)
        if return_error:
            return result, None
        return result
    except Exception as exc:
        if on_error == 'raise':
            raise
        elif on_error == 'none':
            err = _build_error_info(
                exc,
                line_no=line_no,
                byte_start=byte_start,
                raw_line=line,
                parser=parser,
            )
            if return_error:
                return error_value, err
            return error_value
        else:
            raise ValueError(f"unknown on_error: {on_error}")


def save_dicts_to_jsonl(items: List[Dict], jsonl_path: Union[str, Path]):
    with open(jsonl_path, 'w') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_samples_from_jsonl(
    jsonl_path,
    parser: ParserName = 'json',
    on_error: OnErrorMode = 'raise',
    error_value: Any = None,
    return_errors: bool = False,
):
    lines = []
    errors = []
    with open(jsonl_path, 'rb') as f:
        for line_no, line in enumerate(f, start=0):
            parsed = _parse_json_line(
                line,
                parser=parser,
                on_error=on_error,
                error_value=error_value,
                line_no=line_no,
                return_error=return_errors,
            )
            if return_errors:
                value, err = parsed
                if line.strip() != b'':
                    lines.append(value)
                if err is not None:
                    errors.append(err)
            else:
                if line.strip() != b'':
                    lines.append(parsed)
    if return_errors:
        return lines, errors
    return lines


def build_jsonl_index(jsonl_path, idx_path=None, use_tqdm=True):
    """
    为 JSONL 文件创建行索引，并显示处理进度的tqdm进度条。
    进度条基于文件处理的字节数。
    """
    if idx_path is None:
        idx_path = jsonl_path + '.idx'

    total_size = os.path.getsize(jsonl_path)

    with open(jsonl_path, 'rb') as f, open(idx_path, 'wb') as idx:
        pos = 0
        if use_tqdm:
            from tqdm import tqdm
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=f"Building index for {os.path.basename(jsonl_path)}"
            ) as pbar:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    idx.write(struct.pack('<Q', pos))
                    line_length = len(line)
                    pos += line_length
                    pbar.update(line_length)
        else:
            while True:
                line = f.readline()
                if not line:
                    break
                idx.write(struct.pack('<Q', pos))
                line_length = len(line)
                pos += line_length

    return idx_path


def get_jsonl_line_by_number(
    jsonl_path,
    idx_path,
    n,
    use_mmap=True,
    parser: ParserName = 'orjson',
    verbose=True,
    on_error: OnErrorMode = 'raise',
    error_value: Any = None,
    return_error: bool = False,
):
    try:
        offsets = np.memmap(idx_path, dtype=np.uint64, mode='r')
        if n < 0 or n >= len(offsets):
            raise IndexError("line number out of range")
        start_off = int(offsets[n])
        end_off = int(offsets[n + 1]) if (n + 1) < len(offsets) else os.path.getsize(jsonl_path)
        length = end_off - start_off
        if length <= 0:
            raise IndexError("line length <= 0")

        if use_mmap:
            with open(jsonl_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
                data = mm[start_off:end_off]
                mm.close()
        else:
            with open(jsonl_path, 'rb') as f:
                data = os.pread(f.fileno(), length, start_off)

        return _parse_json_line(
            data,
            parser=parser,
            on_error=on_error,
            error_value=error_value,
            line_no=n,
            byte_start=start_off,
            return_error=return_error,
        )
    except Exception:
        if verbose:
            print(f"{jsonl_path = } {idx_path = } {n = }")
            traceback.print_exc()
        if return_error:
            return error_value, {
                'error_type': 'OuterReadError',
                'error_message': f'failed to read line {n}',
                'line_no': n,
                'jsonl_path': str(jsonl_path),
                'idx_path': str(idx_path),
            }
        return error_value


def count_jsonl_n_lines(idx_path):
    size = os.path.getsize(idx_path)
    if size % 8 != 0:
        raise ValueError("idx 文件大小不是 8 的整数倍，可能不是全量偏移索引或文件损坏")
    return size // 8


class JsonlChunkReader:

    def __init__(
        self,
        jsonl_path: str,
        idx_path: str = None,
        mmap_jsonl: bool = True,
        mmap_idx: bool = True,
        parser: ParserName = 'orjson',
    ):
        self.jsonl_path = jsonl_path
        self.idx_path = idx_path if idx_path is not None else jsonl_path + '.idx'
        if not os.path.isfile(self.idx_path):
            self.idx_path = build_jsonl_index(jsonl_path, self.idx_path, use_tqdm=False)
        self.file_size = os.path.getsize(jsonl_path)
        self.n_lines = count_jsonl_n_lines(self.idx_path)

        self.offsets = np.memmap(self.idx_path, dtype=np.uint64, mode='r') if mmap_idx else None

        self._f = open(jsonl_path, 'rb')
        self._mm = mmap.mmap(self._f.fileno(), length=0, access=mmap.ACCESS_READ) if mmap_jsonl else None

        self.parser_name = parser

    def __len__(self):
        return self.n_lines

    def close(self):
        try:
            if self._mm is not None:
                self._mm.close()
            if self._f is not None:
                self._f.close()
        except Exception:
            pass

    def _get_offsets(self, start: int, end: int, strict=False) -> Tuple[int, int]:
        if self.offsets is None:
            offsets = np.memmap(self.idx_path, dtype=np.uint64, mode='r')
        else:
            offsets = self.offsets

        if not strict and end >= len(offsets):
            end = len(offsets) - 1

        if start < 0 or end < start or end >= len(offsets):
            raise IndexError("line range out of range")

        start_off = int(offsets[start])
        if end + 1 < len(offsets):
            end_off = int(offsets[end + 1])
        else:
            end_off = self.file_size
        return start_off, end_off

    def _read_chunk_bytes(self, start_off: int, end_off: int) -> bytes:
        length = end_off - start_off
        if length <= 0:
            return b''

        if self._mm is not None:
            return self._mm[start_off:end_off]
        return os.pread(self._f.fileno(), length, start_off)

    def _parse_chunk_lines(
        self,
        chunk: bytes,
        start_line_no: int,
        start_off: Optional[int] = None,
        on_error: OnErrorMode = 'raise',
        error_value: Any = None,
        return_errors: bool = False,
    ):
        res = []
        errors = []
        cur_off = start_off

        for idx, line in enumerate(chunk.splitlines(keepends=True)):
            raw_line = line.rstrip(b'\r\n')
            line_no = start_line_no + idx
            byte_start = cur_off
            if cur_off is not None:
                cur_off += len(line)

            if raw_line.strip() == b'':
                continue

            parsed = _parse_json_line(
                raw_line,
                parser=self.parser_name,
                on_error=on_error,
                error_value=error_value,
                line_no=line_no,
                byte_start=byte_start,
                return_error=return_errors,
            )

            if return_errors:
                value, err = parsed
                res.append(value)
                if err is not None:
                    errors.append(err)
            else:
                res.append(parsed)

        if return_errors:
            return res, errors
        return res

    def read_range(
        self,
        start: int,
        end: int,
        strict: bool = False,
        on_error: OnErrorMode = 'raise',
        error_value: Any = None,
        return_errors: bool = False,
    ):
        """
        读取 [start, end] 连续行，返回已解析的对象列表。

        参数：
            strict:
                False: 当 end 超过文件末尾时自动截断（保持原行为）
                True : 越界直接报错
            on_error:
                'raise': 默认行为，遇到坏行直接抛异常（前向兼容）
                'none' : 坏行用 error_value 代替继续返回
            error_value:
                on_error='none' 时用于替代坏行的返回值，默认 None
            return_errors:
                True 时额外返回错误信息列表：(results, errors)
        """
        start_off, end_off = self._get_offsets(start, end, strict)
        chunk = self._read_chunk_bytes(start_off, end_off)
        if chunk == b'':
            if return_errors:
                return [], []
            return []

        if self.parser_name == 'simdjson' and on_error == 'raise' and not return_errors:
            import simdjson
            return list(simdjson.Parser().parse_many(chunk))

        return self._parse_chunk_lines(
            chunk,
            start_line_no=start,
            start_off=start_off,
            on_error=on_error,
            error_value=error_value,
            return_errors=return_errors,
        )

    def read_one(
        self,
        n: int,
        on_error: OnErrorMode = 'raise',
        error_value: Any = None,
        return_error: bool = False,
    ):
        """
        单行读取（通过两个偏移计算长度 + mmap 切片/posix pread），避免 readline 的逐字节扫描。
        """
        start_off, end_off = self._get_offsets(n, n)
        length = end_off - start_off
        if length <= 0:
            raise IndexError("line length <= 0")

        if self._mm is not None:
            data = self._mm[start_off:end_off]
        else:
            data = os.pread(self._f.fileno(), length, start_off)

        return _parse_json_line(
            data,
            parser=self.parser_name,
            on_error=on_error,
            error_value=error_value,
            line_no=n,
            byte_start=start_off,
            return_error=return_error,
        )


def get_jsonl_lines_by_range(
    jsonl_path,
    idx_path,
    start,
    end,
    use_mmap=True,
    parser: ParserName = 'orjson',
    on_error: OnErrorMode = 'raise',
    error_value: Any = None,
    return_errors: bool = False,
):
    file_size = os.path.getsize(jsonl_path)
    offsets = np.memmap(idx_path, dtype=np.uint64, mode='r')
    if start < 0 or end < start or end >= len(offsets):
        raise IndexError("line range out of range")

    start_off = int(offsets[start])
    end_off = int(offsets[end + 1]) if (end + 1) < len(offsets) else file_size
    length = end_off - start_off
    if length <= 0:
        if return_errors:
            return [], []
        return []

    if use_mmap:
        with open(jsonl_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            chunk = mm[start_off:end_off]
            mm.close()
    else:
        with open(jsonl_path, 'rb') as f:
            chunk = os.pread(f.fileno(), length, start_off)

    if parser == 'simdjson' and on_error == 'raise' and not return_errors:
        import simdjson
        return list(simdjson.Parser().parse_many(chunk))

    res = []
    errors = []
    cur_off = start_off

    for idx, line in enumerate(chunk.splitlines(keepends=True)):
        raw_line = line.rstrip(b'\r\n')
        line_no = start + idx
        byte_start = cur_off
        cur_off += len(line)

        if raw_line.strip() == b'':
            continue

        parsed = _parse_json_line(
            raw_line,
            parser=parser,
            on_error=on_error,
            error_value=error_value,
            line_no=line_no,
            byte_start=byte_start,
            return_error=return_errors,
        )

        if return_errors:
            value, err = parsed
            res.append(value)
            if err is not None:
                errors.append(err)
        else:
            res.append(parsed)

    if return_errors:
        return res, errors
    return res