from __future__ import annotations

import bisect
import math
from typing import Any

from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .layout import FILE_BLOB_ALIGN, HEADER_SIZE, TENSOR_BLOB_ALIGN, Layout, Region
from .snapshot import human_bytes
from .widgets import Panel, StatCard, clear_layout, label

_KIND_COLORS = {'header': '#1b1d22', 'metadata': '#8a6d3b', 'padding': '#d9d7cf', 'gap': theme.ERROR, 'trailing': theme.WARN}
_ROW_H = 6
_ROW_GAP = 1
_BYTES_PER_ROW = (4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20)


_MAX_MAP_HEIGHT = 4_000_000  # Stay well below Qt's widget height limit


def group_key(r: Region, color_by: str) -> str:
    if color_by == 'dtype':
        return r.dtype
    parts = r.name.split('.')
    if color_by == 'module':
        return parts[0]
    # Layer: the prefix through the first numeric segment, e.g. decoder.layers.3
    for i, seg in enumerate(parts):
        if seg.isdigit():
            return '.'.join(parts[: i + 1])
    return '.'.join(parts[:-1]) or parts[0]


def _region_color(r: Region, color_by: str, palette: dict[str, QColor], index: int) -> QColor:
    if r.kind != 'tensor':
        return QColor(_KIND_COLORS[r.kind])
    key = group_key(r, color_by)
    c = QColor(palette.get(key, QColor(theme.ACCENT)))
    # Alternate lightness so neighbouring tensors of the same color stay distinguishable
    return c.lighter(118) if index % 2 else c


class ByteMap(QWidget):
    """The file wrapped into rows of a fixed byte count, every region painted where its bytes live."""

    hovered = pyqtSignal(int)  # Region index or -1
    activated = pyqtSignal(int)
    opened = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self._layout: Layout | None = None
        self._starts: list[int] = []
        self._colors: list[QColor] = []
        self.bytes_per_row = 64 << 10
        self.color_by = 'dtype'
        self._hover = -1
        self._selected = -1
        self._label_w = 84

    def set_layout(self, layout: Layout | None) -> None:
        self._layout = layout
        self._starts = [r.start for r in layout.regions] if layout else []
        self._hover = self._selected = -1
        self._recolor()
        self._resize()

    def set_bytes_per_row(self, n: int) -> None:
        self.bytes_per_row = n
        self._resize()

    def set_color_by(self, mode: str) -> None:
        self.color_by = mode
        self._recolor()
        self.update()

    def palette_keys(self) -> dict[str, QColor]:
        if self._layout is None:
            return {}
        keys: list[str] = []
        for r in self._layout.regions:
            if r.kind == 'tensor':
                k = group_key(r, self.color_by)
                if k not in keys:
                    keys.append(k)
        return {k: QColor(theme.CATEGORICAL[i % len(theme.CATEGORICAL)]) for i, k in enumerate(keys)}

    def _recolor(self) -> None:
        pal = self.palette_keys()
        self._colors = [_region_color(r, self.color_by, pal, i) for i, r in enumerate(self._layout.regions)] if self._layout else []

    def _rows(self) -> int:
        return math.ceil(self._layout.file_size / self.bytes_per_row) if self._layout else 0

    def _resize(self) -> None:
        self.setMinimumHeight(max(1, self._rows() * (_ROW_H + _ROW_GAP) + 8))
        self.updateGeometry()
        self.update()

    def select(self, idx: int) -> None:
        self._selected = idx
        self.update()

    def region_rect_top(self, idx: int) -> int:
        if self._layout is None or idx < 0:
            return 0
        return (self._layout.regions[idx].start // self.bytes_per_row) * (_ROW_H + _ROW_GAP)

    def _bar_width(self) -> int:
        return max(1, self.width() - self._label_w - 8)

    def offset_at(self, pos: QPoint) -> int | None:
        if self._layout is None or pos.x() < self._label_w:
            return None
        row = pos.y() // (_ROW_H + _ROW_GAP)
        frac = (pos.x() - self._label_w) / self._bar_width()
        if not 0 <= frac < 1:
            return None
        off = row * self.bytes_per_row + int(frac * self.bytes_per_row)
        return off if 0 <= off < self._layout.file_size else None

    def region_at(self, off: int) -> int:
        i = bisect.bisect_right(self._starts, off) - 1
        if self._layout is None or i < 0:
            return -1
        return i if off < self._layout.regions[i].end else -1

    def _spans(self, r: Region) -> list[tuple[int, float, float]]:
        """(row, x0, x1) pieces of a region, x in bar pixels."""
        out = []
        bpr, w = self.bytes_per_row, self._bar_width()
        start, end = r.start, max(r.end, r.start + 1)
        row = start // bpr
        while row * bpr < end:
            a = max(start, row * bpr) - row * bpr
            b = min(end, (row + 1) * bpr) - row * bpr
            out.append((row, a / bpr * w, b / bpr * w))
            row += 1
        return out

    def paintEvent(self, e: Any) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.SURFACE))
        if self._layout is None:
            return
        clip: QRect = e.rect()
        pitch = _ROW_H + _ROW_GAP
        r0 = max(0, clip.top() // pitch)
        r1 = min(self._rows() - 1, clip.bottom() // pitch + 1)
        bpr, x0 = self.bytes_per_row, self._label_w
        regions = self._layout.regions
        first = max(0, bisect.bisect_right(self._starts, r0 * bpr) - 1)
        p.setPen(Qt.PenStyle.NoPen)
        i = first
        while i < len(regions) and regions[i].start < (r1 + 1) * bpr:
            r = regions[i]
            color = self._colors[i]
            for row, a, b in self._spans(r):
                if row < r0 or row > r1:
                    continue
                rect = QRectF(x0 + a, row * pitch, max(1.0, b - a), _ROW_H)
                p.fillRect(rect, color)
                if r.kind in ('padding', 'gap'):
                    p.fillRect(rect, QBrush(QColor(theme.TEXT_FAINT), Qt.BrushStyle.BDiagPattern))
            i += 1
        # Row offset labels every few rows
        p.setFont(theme.mono_font(7))
        p.setPen(QColor(theme.TEXT_FAINT))
        every = max(1, 18 // pitch + 1)
        for row in range(r0 - r0 % every, r1 + 1, every):
            p.drawText(QRectF(0, row * pitch - 4, x0 - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'0x{row * bpr:X}')
        for idx, pen in ((self._hover, QPen(QColor(theme.INK), 1)), (self._selected, QPen(QColor(theme.ACCENT), 2))):
            if 0 <= idx < len(regions):
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                for row, a, b in self._spans(regions[idx]):
                    if r0 <= row <= r1:
                        p.drawRect(QRectF(x0 + a, row * pitch - 0.5, max(1.0, b - a), _ROW_H + 1))

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        off = self.offset_at(e.position().toPoint())
        idx = self.region_at(off) if off is not None else -1
        if idx != self._hover:
            self._hover = idx
            self.update()
            self.hovered.emit(idx)
        if idx >= 0 and off is not None and self._layout is not None:
            r = self._layout.regions[idx]
            tip = f'<b>{r.name}</b><br>{r.kind} · {human_bytes(r.size)}<br>0x{r.start:X} – 0x{r.end:X}<br>byte 0x{off:X} (+{off - r.start:,})'
            if r.issues:
                tip += '<br><span style="color:#c53030">' + '<br>'.join(r.issues) + '</span>'
            QToolTip.showText(e.globalPosition().toPoint(), tip, self)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        off = self.offset_at(e.position().toPoint())
        idx = self.region_at(off) if off is not None else -1
        if idx >= 0:
            self.select(idx)
            self.activated.emit(idx)

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        off = self.offset_at(e.position().toPoint())
        idx = self.region_at(off) if off is not None else -1
        if idx >= 0:
            self.opened.emit(idx)

    def leaveEvent(self, _e: Any) -> None:
        self._hover = -1
        self.update()


class LayoutPage(QWidget):
    tensor_requested = pyqtSignal(str)

    COLUMNS = ('Kind', 'Name', 'Start', 'End', 'Size', 'Align', 'Issues')

    def __init__(self) -> None:
        super().__init__()
        self._layout: Layout | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(10)

        self.cards = QGridLayout()
        self.cards.setSpacing(10)
        lay.addLayout(self.cards)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(label('BYTE MAP', 'SectionTitle'))
        bar.addSpacing(8)
        self.bpr = QComboBox()
        for n in _BYTES_PER_ROW:
            self.bpr.addItem(f'{human_bytes(n)} / row', n)
        self.bpr.setToolTip('Bytes per row. At 4 KiB every row is one page, so aligned tensors start at the left edge')
        bar.addWidget(self.bpr)
        self.color_by = QComboBox()
        self.color_by.addItems(['color by dtype', 'color by module', 'color by layer'])
        bar.addWidget(self.color_by)
        self.legend = QHBoxLayout()
        self.legend.setSpacing(10)
        bar.addLayout(self.legend)
        bar.addStretch(1)
        self.only_issues = QCheckBox('Table: padding and issues only')
        bar.addWidget(self.only_issues)
        lay.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        map_panel = Panel()
        ml = QVBoxLayout(map_panel)
        ml.setContentsMargins(8, 8, 8, 8)
        self.map = ByteMap()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setWidget(self.map)
        ml.addWidget(self.scroll)
        split.addWidget(map_panel)

        table_panel = Panel()
        tl = QVBoxLayout(table_panel)
        tl.setContentsMargins(1, 1, 1, 1)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
        hh.resizeSection(6, 90)
        tl.addWidget(self.table)
        split.addWidget(table_panel)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([440, 800])
        lay.addWidget(split, 1)

        self.bpr.currentIndexChanged.connect(lambda _: self.map.set_bytes_per_row(self.bpr.currentData()))
        self.color_by.currentTextChanged.connect(self._on_color_by)
        self.only_issues.toggled.connect(lambda _: self._fill_table())
        self.map.activated.connect(self._select_from_map)
        self.map.opened.connect(self._open_region)
        self.table.currentCellChanged.connect(self._select_from_table)
        self.table.cellDoubleClicked.connect(self._open_row)

    def set_layout(self, layout: Layout | None) -> None:
        self._layout = layout
        self.map.set_layout(layout)
        if layout is None:
            self.table.setRowCount(0)
            return
        self.bpr.blockSignals(True)
        self.bpr.clear()
        pitch = _ROW_H + _ROW_GAP
        sizes = [n for n in _BYTES_PER_ROW if math.ceil(layout.file_size / n) * pitch <= _MAX_MAP_HEIGHT]
        sizes = sizes or [_BYTES_PER_ROW[-1]]
        for n in sizes:
            self.bpr.addItem(f'{human_bytes(n)} / row', n)
        # Aim for a map a few hundred rows tall
        target = layout.file_size / 600
        self.bpr.setCurrentIndex(next((i for i, n in enumerate(sizes) if n >= target), len(sizes) - 1))
        self.bpr.blockSignals(False)
        dtypes = {r.dtype for r in layout.regions if r.kind == 'tensor'}
        self.color_by.blockSignals(True)
        self.color_by.setCurrentText('color by dtype' if len(dtypes) > 1 else 'color by layer')
        self.color_by.blockSignals(False)
        self.map.set_color_by(self.color_by.currentText().rsplit(' ', 1)[-1])
        self.map.set_bytes_per_row(self.bpr.currentData())
        self._fill_cards()
        self._fill_legend()
        self._fill_table()

    def _on_color_by(self, text: str) -> None:
        self.map.set_color_by(text.rsplit(' ', 1)[-1])
        self._fill_legend()

    def _fill_cards(self) -> None:
        clear_layout(self.cards)
        L = self._layout
        assert L is not None
        h = L.header
        n_issues = len(L.issues)
        pad_pct = L.padding_bytes / max(1, L.file_size)
        cards = [
            (
                'Header',
                f'{h.magic_text} · v{h.version} · aux 0x{h.aux:X}',
                f'{HEADER_SIZE} bytes, little endian. Version field {h.version}, aux bits 0x{h.aux:08X}',
            ),
            ('Data section', f'0x{h.blob_offset:X}', f'{h.blob_span:,} bytes, file alignment {FILE_BLOB_ALIGN}'),
            ('Payload', human_bytes(L.payload_bytes), f'{L.payload_bytes:,} bytes of tensor data'),
            ('Padding', f'{human_bytes(L.padding_bytes)} · {pad_pct:.2%}', f'{L.padding_bytes:,} bytes, tensor alignment {TENSOR_BLOB_ALIGN}'),
            ('Byte order', 'big endian' if h.big_endian else 'little endian', 'From aux bit 0'),
            ('Issues', 'none' if not n_issues else f'{n_issues} found', '\n'.join(L.issues) or 'Every byte is attributed and every rule holds'),
        ]
        for i, (t, v, tip) in enumerate(cards):
            c = StatCard(t)
            c.set_value(v, tip)
            if t == 'Issues':
                c.value_lbl.setStyleSheet(f'color: {theme.OK if not n_issues else theme.ERROR};')
            self.cards.addWidget(c, 0, i)

    def _fill_legend(self) -> None:
        clear_layout(self.legend)
        groups = [(k, c.name()) for k, c in self.map.palette_keys().items()]
        fixed = [('header', _KIND_COLORS['header']), ('metadata', _KIND_COLORS['metadata']), ('padding', _KIND_COLORS['padding'])]
        shown = groups if len(groups) <= 4 else []
        for name, color in shown + fixed:
            dot = label('■')
            dot.setStyleSheet(f'color: {color};')
            self.legend.addWidget(dot)
            self.legend.addWidget(label(name, 'Subtitle'))
        if not shown and groups:
            more = label(f'{len(groups)} tensor groups', 'Subtitle')
            more.setToolTip('<br>'.join(f'<span style="color:{c}">■</span> {k}' for k, c in groups))
            self.legend.insertWidget(0, more)

    def _visible_regions(self) -> list[int]:
        if self._layout is None:
            return []
        regions = self._layout.regions
        if not self.only_issues.isChecked():
            return list(range(len(regions)))
        return [i for i, r in enumerate(regions) if r.kind != 'tensor' or r.issues]

    def _fill_table(self) -> None:
        L = self._layout
        if L is None:
            return
        rows = self._visible_regions()
        self.table.setRowCount(len(rows))
        mono = theme.mono_font(9)
        err = QColor(theme.ERROR)
        dim = QColor(theme.TEXT_FAINT)
        for row, idx in enumerate(rows):
            r = L.regions[idx]
            align = TENSOR_BLOB_ALIGN if r.kind == 'tensor' else FILE_BLOB_ALIGN if r.kind != 'header' else 1
            aligned = r.start % align == 0
            cells = [
                r.kind,
                r.name,
                f'0x{r.start:X}',
                f'0x{r.end:X}',
                f'{r.size:,}',
                ('✓' if aligned else f'+{r.start % align}') if r.kind == 'tensor' or r.kind == 'metadata' else '',
                '; '.join(r.issues),
            ]
            for col, text in enumerate(cells):
                it = QTableWidgetItem(text)
                it.setData(Qt.ItemDataRole.UserRole, idx)
                if col in (2, 3, 4, 5):
                    it.setFont(mono)
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 0:
                    it.setForeground(QColor(_KIND_COLORS.get(r.kind, theme.ACCENT)) if r.kind != 'padding' else dim)
                if r.issues:
                    it.setForeground(err)
                elif r.kind in ('padding',) and col > 0:
                    it.setForeground(dim)
                self.table.setItem(row, col, it)

    def _row_of(self, idx: int) -> int:
        for row in range(self.table.rowCount()):
            it = self.table.item(row, 0)
            if it is not None and it.data(Qt.ItemDataRole.UserRole) == idx:
                return row
        return -1

    def _select_from_map(self, idx: int) -> None:
        row = self._row_of(idx)
        if row >= 0:
            self.table.blockSignals(True)
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0), QAbstractItemView.ScrollHint.PositionAtCenter)
            self.table.blockSignals(False)

    def _select_from_table(self, row: int, *_: Any) -> None:
        it = self.table.item(row, 0)
        if it is None:
            return
        idx = it.data(Qt.ItemDataRole.UserRole)
        self.map.select(idx)
        y = self.map.region_rect_top(idx)
        self.scroll.ensureVisible(0, y, 0, self.scroll.viewport().height() // 3)

    def _open_row(self, row: int, _col: int) -> None:
        it = self.table.item(row, 0)
        if it is not None:
            self._open_region(it.data(Qt.ItemDataRole.UserRole))

    def _open_region(self, idx: int) -> None:
        if self._layout is None:
            return
        r = self._layout.regions[idx]
        if r.kind == 'tensor':
            self.tensor_requested.emit(r.name)
