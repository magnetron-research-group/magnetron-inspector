from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any

from magnetron import Tensor
from magnetron._magnetron_bindings import dtype as mag_dtype
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QStandardItem, QStandardItemModel

from . import theme
from .snapshot import UINT_FOR_SIZE, Snapshot, human_bytes, is_float

_ROOT = QModelIndex()

FULL_NAME_ROLE = Qt.ItemDataRole.UserRole + 1
TENSOR_ROLE = Qt.ItemDataRole.UserRole + 2
SORT_ROLE = Qt.ItemDataRole.UserRole + 3

_BLOCK_CELLS = 1 << 18
_MAX_BLOCKS = 48
_HEAT_LEVELS = 48


def _natural(s: str) -> str:
    return re.sub(r'\d+', lambda m: m.group().zfill(10), s)


def _heat_palette() -> tuple[list[QColor], list[QColor]]:
    base = QColor(theme.SURFACE)
    neg, pos = QColor(theme.HEAT_NEG), QColor(theme.HEAT_POS)

    def mix(c: QColor, t: float) -> QColor:
        return QColor(
            round(base.red() + (c.red() - base.red()) * t),
            round(base.green() + (c.green() - base.green()) * t),
            round(base.blue() + (c.blue() - base.blue()) * t),
        )

    levels = [(i / (_HEAT_LEVELS - 1)) * 0.62 for i in range(_HEAT_LEVELS)]
    return [mix(neg, t) for t in levels], [mix(pos, t) for t in levels]


class TensorTableModel(QAbstractTableModel):
    """2D window over an N-d tensor: the last two dims are rows and columns, leading dims are fixed by slice indices."""

    FORMATS = ('Auto', 'Fixed', 'Scientific', 'Hex bits')

    def __init__(self) -> None:
        super().__init__()
        self._tensor: Tensor | None = None
        self._view: Tensor | None = None
        self._rows = self._cols = 0
        self._dtype: str = 'float32'
        self._slice: tuple[int, ...] = ()
        self._blocks: OrderedDict[int, list[list[Any]]] = OrderedDict()
        self._bit_blocks: OrderedDict[int, list[list[int]]] = OrderedDict()
        self._block_rows: int = 1
        self.fmt: str = 'Auto'
        self.precision: int = 5
        self.heatmap: bool = True
        self._absmax: float = 0.0
        self._neg_pal, self._pos_pal = _heat_palette()
        self._mono = theme.mono_font(10)
        self._nan_color = QColor(theme.NAN)
        self._zero_color = QColor(theme.TEXT_FAINT)
        self._align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

    # Setup

    def set_tensor(self, tensor: Tensor | None, dtype_name: str) -> None:
        self.beginResetModel()
        self._tensor = tensor
        self._dtype = dtype_name
        self._absmax = 0.0
        self._slice = (0,) * max(0, (tensor.rank if tensor is not None else 0) - 2)
        self._rebuild_view()
        self.endResetModel()

    def set_slice(self, idx: tuple[int, ...]) -> None:
        self.beginResetModel()
        self._slice = idx
        self._rebuild_view()
        self.endResetModel()

    def set_absmax(self, v: float) -> None:
        self._absmax = v
        if self.heatmap:
            self._refresh()

    def set_format(self, fmt: str, precision: int) -> None:
        self.fmt, self.precision = fmt, precision
        self._refresh()

    def set_heatmap(self, on: bool) -> None:
        self.heatmap = on
        self._refresh()

    def _refresh(self) -> None:
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def _rebuild_view(self) -> None:
        self._blocks.clear()
        self._bit_blocks.clear()
        t = self._tensor
        if t is None or t.numel == 0:
            self._view, self._rows, self._cols = None, 0, 0
            return
        if t.rank == 0:
            v = t.reshape(1, 1)
        elif t.rank == 1:
            v = t.reshape(-1, 1)
        else:
            v = t
            for i in self._slice:
                v = v.select(0, i)
        self._view = v
        self._rows, self._cols = v.shape
        self._block_rows = max(1, _BLOCK_CELLS // max(1, self._cols))

    @property
    def dtype_name(self) -> str:
        return self._dtype

    @property
    def view_tensor(self) -> Tensor | None:
        """The current 2D slice the grid shows."""
        return self._view

    # Access, decoded in row blocks so scrolling costs one tolist() per block

    def _fetch(self, cache: OrderedDict[int, list[list[Any]]], b: int, bits: bool) -> list[list[Any]]:
        blk = cache.get(b)
        if blk is not None:
            cache.move_to_end(b)
            return blk
        assert self._view is not None
        r0 = b * self._block_rows
        part = self._view.narrow(0, r0, min(self._block_rows, self._rows - r0))
        if bits:
            part = part.reinterpret_view(UINT_FOR_SIZE[part.dtype.size], *part.shape)
        blk = part.tolist()
        cache[b] = blk
        if len(cache) > _MAX_BLOCKS:
            cache.popitem(last=False)
        return blk

    def value(self, row: int, col: int) -> Any:
        b, r = divmod(row, self._block_rows)
        return self._fetch(self._blocks, b, False)[r][col]

    def raw_bits(self, row: int, col: int) -> str:
        b, r = divmod(row, self._block_rows)
        bits = self._fetch(self._bit_blocks, b, True)[r][col]
        assert self._view is not None
        return f'0x{bits:0{self._view.dtype.size * 2}X}'

    def format_value(self, v: Any, row: int, col: int) -> str:
        if self.fmt == 'Hex bits':
            return self.raw_bits(row, col)
        if self._dtype == 'boolean':
            return 'true' if v else 'false'
        if not is_float(self._dtype):
            return str(int(v))
        f = float(v)
        if math.isnan(f):
            return 'NaN'
        if math.isinf(f):
            return '+Inf' if f > 0 else '-Inf'
        p = self.precision
        if self.fmt == 'Fixed':
            return f'{f:.{p}f}'
        if self.fmt == 'Scientific':
            return f'{f:.{p}e}'
        return f'{f:.{p}g}'

    # Qt model

    def rowCount(self, parent: QModelIndex = _ROOT) -> int:
        return 0 if parent.isValid() else self._rows

    def columnCount(self, parent: QModelIndex = _ROOT) -> int:
        return 0 if parent.isValid() else self._cols

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            r, c = index.row(), index.column()
            return self.format_value(self.value(r, c), r, c)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return self._align
        if role == Qt.ItemDataRole.FontRole:
            return self._mono
        if role == Qt.ItemDataRole.ForegroundRole:
            v = self.value(index.row(), index.column())
            if is_float(self._dtype) and not math.isfinite(float(v)):
                return self._nan_color
            if v == 0:
                return self._zero_color
            return None
        if role == Qt.ItemDataRole.BackgroundRole and self.heatmap and self._absmax > 0 and self._dtype != 'boolean':
            f = float(self.value(index.row(), index.column()))
            if not math.isfinite(f):
                return None
            t = min(1.0, math.sqrt(abs(f) / self._absmax))
            lvl = round(t * (_HEAT_LEVELS - 1))
            return (self._neg_pal if f < 0 else self._pos_pal)[lvl]
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            return str(section)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None


class StatsSignals(QObject):
    done = pyqtSignal(int, dict)


class StatsJob(QRunnable):
    """Computes summary stats and a histogram with magnetron ops, in chunks, off the UI thread."""

    CHUNK = 1 << 22  # Elements cast to float32 at a time, bounds peak memory
    SUM_ROW = 4096  # Partial sums stay short so float32 accumulation stays exact enough, fsum combines them
    BINS = 96

    def __init__(self, token: int, tensor: Tensor, dtype_name: str) -> None:
        super().__init__()
        self.token = token
        self.flat = tensor.flatten() if tensor.rank else tensor.reshape(1)
        self.dtype_name = dtype_name
        self.cancelled = False
        self.signals = StatsSignals()

    def run(self) -> None:
        try:
            res = self._compute()
        except Exception as e:
            res = {'error': str(e)}
        if not self.cancelled:
            self.signals.done.emit(self.token, res)

    def _chunks(self) -> Any:
        n = self.flat.numel
        for i in range(0, n, self.CHUNK):
            if self.cancelled:
                return
            yield self.flat.narrow(0, i, min(self.CHUNK, n - i)).cast(mag_dtype.float32)

    @classmethod
    def _sum(cls, x: Tensor) -> float:
        n = x.numel
        main = n - n % cls.SUM_ROW
        parts = x.narrow(0, 0, main).reshape(-1, cls.SUM_ROW).sum(1).tolist() if main else []
        if main < n:
            parts.append(x.narrow(0, main, n - main).sum().item())
        return math.fsum(parts)

    @staticmethod
    def _count(mask: Tensor) -> int:
        return int(mask.cast(mag_dtype.int64).sum().item())

    def _compute(self) -> dict[str, Any]:
        n = self.flat.numel
        if n == 0:
            return {'numel': 0}
        lo, hi = math.inf, -math.inf
        total = 0.0
        nan = inf = zeros = finite = 0
        for c in self._chunks():
            fin = (c == c) & (c.abs() != math.inf)
            nf = self._count(fin)
            nn = self._count(c != c)
            nan += nn
            inf += c.numel - nf - nn
            zeros += self._count(c == 0.0)
            if nf:
                cf = c if nf == c.numel else Tensor.where(fin, c, 0.0)
                lo = min(lo, (c if nf == c.numel else Tensor.where(fin, c, math.inf)).amin().item())
                hi = max(hi, (c if nf == c.numel else Tensor.where(fin, c, -math.inf)).amax().item())
                total += self._sum(cf)
                finite += nf
        if self.cancelled:
            return {}
        res: dict[str, Any] = {'numel': n, 'nan': nan, 'inf': inf, 'zeros': zeros, 'finite': finite}
        if not finite:
            return res
        mean = total / finite
        edges_hi = hi if hi > lo else lo + 1.0
        scale = self.BINS / (edges_hi - lo)
        sq = 0.0
        hist = [0] * self.BINS
        for c in self._chunks():
            fin = (c == c) & (c.abs() != math.inf)
            all_finite = self._count(fin) == c.numel
            d = (c if all_finite else Tensor.where(fin, c, mean)) - mean
            sq += self._sum(d * d)
            # Non-finite values go to -inf, which always clamps into bin 0
            hc = c if all_finite else Tensor.where(fin, c, -math.inf)
            bins = ((hc - lo) * scale).floor().clamp(0, self.BINS - 1).cast(mag_dtype.int64)
            for i, k in enumerate(bins.bincount(minlength=self.BINS).tolist()):
                hist[i] += int(k)
        if self.cancelled:
            return {}
        hist[0] -= nan + inf
        res.update(min=lo, max=hi, mean=mean, std=math.sqrt(sq / finite), absmax=max(abs(lo), abs(hi)))
        res['hist'] = hist
        res['hist_range'] = (lo, edges_hi)
        return res


class TensorTreeModel(QStandardItemModel):
    def __init__(self) -> None:
        super().__init__()
        self.setHorizontalHeaderLabels(['Name', 'Shape', 'DType', 'Size'])

    def populate(self, snap: Snapshot | None) -> None:
        self.removeRows(0, self.rowCount())
        if snap is None:
            return
        mono = theme.mono_font(9)
        dim = QColor(theme.TEXT_DIM)
        group_color = QColor(theme.ACCENT)
        groups: dict[str, list[QStandardItem]] = {}
        totals: dict[str, list[int]] = {}

        def group_row(path: str) -> list[QStandardItem]:
            if path in groups:
                return groups[path]
            parent_path, _, seg = path.rpartition('.')
            items = [QStandardItem(seg), QStandardItem(), QStandardItem(), QStandardItem()]
            items[0].setData(path, FULL_NAME_ROLE)
            items[0].setData(_natural(seg), SORT_ROLE)
            items[0].setForeground(group_color)
            for it in items:
                it.setEditable(False)
            (group_row(parent_path)[0] if parent_path else self.invisibleRootItem()).appendRow(items)
            groups[path] = items
            totals[path] = [0, 0]
            return items

        for name, info in snap.tensors.items():
            parent_path, _, seg = name.rpartition('.')
            items = [QStandardItem(seg), QStandardItem(info.shape_str), QStandardItem(info.dtype), QStandardItem(human_bytes(info.nbytes))]
            items[0].setData(name, FULL_NAME_ROLE)
            items[0].setData(name, TENSOR_ROLE)
            items[0].setData(_natural(seg), SORT_ROLE)
            items[0].setToolTip(name)
            items[1].setData(info.numel, SORT_ROLE)
            items[2].setData(info.dtype, SORT_ROLE)
            items[3].setData(info.nbytes, SORT_ROLE)
            for it in items:
                it.setEditable(False)
            for it in items[1:]:
                it.setFont(mono)
                it.setForeground(dim)
            (group_row(parent_path)[0] if parent_path else self.invisibleRootItem()).appendRow(items)
            p = parent_path
            while p:
                totals[p][0] += 1
                totals[p][1] += info.nbytes
                p = p.rpartition('.')[0]

        for path, items in groups.items():
            count, nbytes = totals[path]
            items[1].setText(f'{count} tensors')
            items[1].setData(count, SORT_ROLE)
            items[3].setText(human_bytes(nbytes))
            items[3].setData(nbytes, SORT_ROLE)
            items[2].setData('', SORT_ROLE)
            for it in items[1:]:
                it.setForeground(QColor(theme.TEXT_FAINT))
                it.setFont(mono)


class TensorFilterProxy(QSortFilterProxyModel):
    def __init__(self) -> None:
        super().__init__()
        self.setRecursiveFilteringEnabled(True)
        self.setAutoAcceptChildRows(True)
        self.setFilterRole(FULL_NAME_ROLE)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setSortRole(SORT_ROLE)
