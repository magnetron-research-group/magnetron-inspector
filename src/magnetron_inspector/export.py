from __future__ import annotations

import csv
import json
from collections.abc import Callable
from typing import Any

from magnetron import Tensor

from .layout import TENSOR_BLOB_ALIGN, Layout
from .snapshot import Snapshot

Progress = Callable[[int, int], bool]  # (done, total) -> keep going
_CHUNK_CELLS = 1 << 18


def _cell(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return repr(v)  # Shortest round-trip form, nan/inf stay readable
    return v


def as_matrix(t: Tensor) -> Tensor:
    """Rows x last dim view: scalars become 1x1, 1D a single column, N-d keeps its last dim as columns."""
    if t.rank == 0:
        return t.reshape(1, 1)
    if t.rank == 1:
        return t.reshape(-1, 1)
    return t.reshape(-1, t.shape[-1])


def write_tensor_csv(path: str, t: Tensor, progress: Progress | None = None, header: list[str] | None = None) -> bool:
    """Stream a tensor to CSV in row blocks. Returns False when cancelled, the partial file is left as is."""
    m = as_matrix(t.contiguous() if not t.is_contiguous else t)
    rows, cols = m.shape
    step = max(1, _CHUNK_CELLS // max(1, cols))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r0 in range(0, rows, step):
            block = m.narrow(0, r0, min(step, rows - r0)).tolist()
            w.writerows([_cell(v) for v in row] for row in block)
            if progress is not None and not progress(min(rows, r0 + step), rows):
                return False
    return True


def write_metadata_json(path: str, snap: Snapshot, raw: bool = False) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        if raw:
            f.write(snap.metadata_raw)  # The document exactly as stored in the file
        else:
            json.dump(snap.manifest, f, indent=2, ensure_ascii=False)
            f.write('\n')


def write_json_value(path: str, value: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        if isinstance(value, str):
            f.write(value)
        else:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write('\n')


def write_manifest_csv(path: str, snap: Snapshot, layout: Layout | None) -> None:
    blob = layout.header.blob_offset if layout else None
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['name', 'shape', 'dtype', 'numel', 'nbytes', 'data_offset', 'file_offset', 'aligned'])
        for t in snap.tensors.values():
            w.writerow(
                [
                    t.name,
                    'x'.join(map(str, t.shape)),
                    t.dtype,
                    t.numel,
                    t.nbytes,
                    t.offset,
                    '' if blob is None else blob + t.offset,
                    int(t.offset % TENSOR_BLOB_ALIGN == 0),
                ]
            )


def write_layout_csv(path: str, layout: Layout) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'name', 'dtype', 'start', 'end', 'size', 'issues'])
        for r in layout.regions:
            w.writerow([r.kind, r.name, r.dtype, r.start, r.end, r.size, '; '.join(r.issues)])
