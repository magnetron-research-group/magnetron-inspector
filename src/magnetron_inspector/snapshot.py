from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

from magnetron import Tensor
from magnetron._magnetron_bindings import SnapshotStreamReader, dtype as mag_dtype

DTYPE_BY_ID: dict[int, Any] = {dt.ordinal: dt for dt in mag_dtype.all}
DTYPE_BY_NAME: dict[str, Any] = {dt.name: dt for dt in mag_dtype.all}

# Same-width unsigned dtype, used to reinterpret elements as raw bits
UINT_FOR_SIZE: dict[int, Any] = {1: mag_dtype.uint8, 2: mag_dtype.uint16, 4: mag_dtype.uint32, 8: mag_dtype.uint64}


def is_float(dtype_name: str) -> bool:
    return dtype_name in ('float32', 'float16', 'bfloat16', 'float8_e4m3fn')


def human_bytes(n: int) -> str:
    v = float(n)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if v < 1024.0 or unit == 'TiB':
            return f'{int(v)} {unit}' if unit == 'B' else f'{v:.2f} {unit}'
        v /= 1024.0
    return f'{n} B'


def human_count(n: int) -> str:
    for div, suffix in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if n >= div:
            return f'{n / div:.2f}{suffix}'
    return str(n)


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    dtype_id: int
    offset: int
    nbytes: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def shape_str(self) -> str:
        return '×'.join(str(d) for d in self.shape) if self.shape else 'scalar'


class Snapshot:
    def __init__(self, path: str) -> None:
        self.path: str = os.path.abspath(path)
        self.file_size: int = os.path.getsize(self.path)
        self._reader: SnapshotStreamReader = SnapshotStreamReader(self.path)
        self.metadata_raw: str = self._reader.metadata
        self.manifest: dict[str, Any] = json.loads(self.metadata_raw)
        self.format_version: tuple[int, ...] = tuple(self._reader.format_version)
        self.blob_numbytes: int = self._reader.blob_numbytes
        self.tensors: dict[str, TensorInfo] = {
            k: TensorInfo(k, tuple(v['shape']), v['dtype'], v['dtype_id'], v['offset'], v['nbytes'])
            for k, v in self.manifest.get('tensor_map', {}).items()
        }
        self.validation_error: str | None = self._validate()
        self._borrowed: dict[str, Any] = {}

    def _validate(self) -> str | None:
        try:
            from magnetron.snapshot import _FileManifest

            _FileManifest.deserialize(self.metadata_raw).validate(self.blob_numbytes)
        except Exception as e:
            return str(e)
        return None

    @property
    def usr_metadata(self) -> dict[str, Any]:
        return self.manifest.get('usr_metadata', {})

    @property
    def total_params(self) -> int:
        return sum(t.numel for t in self.tensors.values())

    def tensor(self, name: str) -> Tensor:
        """Zero copy, read only magnetron tensor borrowed from the mmapped data section."""
        info = self.tensors[name]
        if info.nbytes == 0:
            return Tensor.empty(*info.shape, dtype=DTYPE_BY_ID[info.dtype_id])
        t = self._reader.tensor(info.offset, info.nbytes, info.dtype_id, list(info.shape))
        self._borrowed[name] = t  # Keep borrowed tensors alive until the reader closes
        return t

    def close(self) -> None:
        self._borrowed.clear()
        self._reader.close()
