from __future__ import annotations

import datetime
import json
from collections import Counter
from typing import Any

from PyQt6.QtCore import QItemSelectionModel, QModelIndex, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .json_model import JsonModel, search
from .snapshot import Snapshot, TensorInfo, human_bytes, human_count, is_float
from .tensor_model import StatsJob, TensorTableModel
from .views import ImagePanel, PointCloudPanel, point_orientations
from .widgets import DTypeBar, Histogram, Panel, StatCard, StatRow, clear_layout, label

_DETAIL_LIMIT = 4_000_000


class OverviewPage(QScrollArea):
    tensor_requested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        self.setWidget(body)
        self.lay = QVBoxLayout(body)
        self.lay.setContentsMargins(24, 20, 24, 24)
        self.lay.setSpacing(16)

    def show_snapshot(self, snap: Snapshot) -> None:
        clear_layout(self.lay)
        m = snap.manifest

        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label(snap.path.rsplit('/', 1)[-1], 'Title'))
        titles.addWidget(label(snap.path, 'Subtitle', selectable=True))
        head.addLayout(titles, 1)
        badge = label('✓ valid layout', 'BadgeOk') if snap.validation_error is None else label('⚠ validation failed', 'BadgeWarn')
        if snap.validation_error:
            badge.setToolTip(snap.validation_error)
        head.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        self.lay.addLayout(head)

        ts = m.get('timestamp', '')
        try:
            ts_disp = datetime.datetime.fromisoformat(ts).strftime('%Y-%m-%d %H:%M:%S %z')
        except ValueError:
            ts_disp = ts or '—'
        cards = [
            ('File size', human_bytes(snap.file_size), f'{snap.file_size:,} bytes'),
            ('Tensors', f'{len(snap.tensors):,}', None),
            ('Parameters', human_count(snap.total_params), f'{snap.total_params:,}'),
            ('Data section', human_bytes(snap.blob_numbytes), f'{snap.blob_numbytes:,} bytes'),
            ('Metadata', human_bytes(len(snap.metadata_raw.encode())), None),
            ('Format', '.'.join(map(str, snap.format_version)), 'On disk format version'),
            ('Snapshot ver', str(m.get('snapshot_ver', '—')), None),
            ('Magnetron ver', str(m.get('magnetron_ver', '—')), None),
            ('Manifest ver', str(m.get('manifest_ver', '—')), None),
            ('Created', ts_disp, ts),
        ]
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, (t, v, tip) in enumerate(cards):
            c = StatCard(t)
            c.set_value(v, tip)
            grid.addWidget(c, i // 5, i % 5)
        self.lay.addLayout(grid)

        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(self._model_panel(snap), 3)
        right = QVBoxLayout()
        right.setSpacing(16)
        right.addWidget(self._dtype_panel(snap))
        right.addWidget(self._largest_panel(snap), 1)
        row.addLayout(right, 2)
        self.lay.addLayout(row)
        self.lay.addStretch(1)

    def _model_panel(self, snap: Snapshot) -> Panel:
        p = Panel()
        lay = QVBoxLayout(p)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.addWidget(label('USER METADATA', 'SectionTitle'))
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        for r, (k, v) in enumerate(snap.usr_metadata.items()):
            if isinstance(v, dict | list):
                disp = f'{{ {len(v)} keys }}' if isinstance(v, dict) else f'[ {len(v)} items ]'
            elif isinstance(v, str) and len(v) > 80:
                disp = f'string · {len(v):,} chars'
            else:
                disp = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            grid.addWidget(label(k, 'Subtitle'), r, 0, Qt.AlignmentFlag.AlignTop)
            val = label(disp, selectable=True)
            val.setFont(theme.mono_font(10))
            val.setWordWrap(True)
            grid.addWidget(val, r, 1)
        cfg = snap.usr_metadata.get('model_config')
        if isinstance(cfg, dict):
            lay.addLayout(grid)
            lay.addSpacing(10)
            lay.addWidget(label('MODEL CONFIG', 'SectionTitle'))
            cgrid = QGridLayout()
            cgrid.setHorizontalSpacing(18)
            cgrid.setVerticalSpacing(4)
            cgrid.setColumnStretch(1, 1)
            cgrid.setColumnStretch(3, 1)
            scalars = [(k, v) for k, v in cfg.items() if not isinstance(v, dict | list)]
            half = (len(scalars) + 1) // 2
            for i, (k, v) in enumerate(scalars):
                col = 0 if i < half else 2
                rr = i if i < half else i - half
                cgrid.addWidget(label(k, 'Subtitle'), rr, col)
                val = label(json.dumps(v, ensure_ascii=False), selectable=True)
                val.setFont(theme.mono_font(10))
                cgrid.addWidget(val, rr, col + 1)
            lay.addLayout(cgrid)
        else:
            lay.addLayout(grid)
        lay.addStretch(1)
        return p

    def _dtype_panel(self, snap: Snapshot) -> Panel:
        p = Panel()
        lay = QVBoxLayout(p)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.addWidget(label('DTYPES', 'SectionTitle'))
        by_bytes: Counter[str] = Counter()
        by_count: Counter[str] = Counter()
        for t in snap.tensors.values():
            by_bytes[t.dtype] += t.nbytes
            by_count[t.dtype] += 1
        parts = by_bytes.most_common()
        bar = DTypeBar()
        bar.set_parts(parts)
        lay.addWidget(bar)
        total = sum(by_bytes.values()) or 1
        for i, (dt, nb) in enumerate(parts):
            row = QHBoxLayout()
            dot = label('●')
            dot.setStyleSheet(f'color: {bar.color(i).name()};')
            row.addWidget(dot)
            row.addWidget(label(dt))
            row.addStretch(1)
            row.addWidget(label(f'{by_count[dt]} tensors · {human_bytes(nb)} · {nb / total:.1%}', 'Subtitle'))
            lay.addLayout(row)
        return p

    def _largest_panel(self, snap: Snapshot) -> Panel:
        p = Panel()
        lay = QVBoxLayout(p)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.addWidget(label('LARGEST TENSORS', 'SectionTitle'))
        for t in sorted(snap.tensors.values(), key=lambda t: t.nbytes, reverse=True)[:10]:
            btn = QPushButton(f'{t.name}')
            btn.setFlat(True)
            btn.setStyleSheet(
                f'QPushButton {{ text-align: left; border: none; background: transparent; padding: 3px 0; color: {theme.TEXT}; }}'
                f'QPushButton:hover {{ color: {theme.ACCENT}; }}'
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, n=t.name: self.tensor_requested.emit(n))
            row = QHBoxLayout()
            row.addWidget(btn, 1)
            row.addWidget(label(f'{t.shape_str}  {human_bytes(t.nbytes)}', 'Subtitle'))
            lay.addLayout(row)
        lay.addStretch(1)
        return p


class MetadataPage(QWidget):
    export_requested = pyqtSignal(str, str)  # Suggested name, text

    def __init__(self) -> None:
        super().__init__()
        self._data: Any = {}
        self._matches: list[tuple[int, ...]] = []
        self._match_i = -1

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(10)

        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search keys and values…  (Enter for next)')
        self.search.setClearButtonEnabled(True)
        self.search.returnPressed.connect(self._next)
        self._debounce = QTimer(self, singleShot=True, interval=250)
        self._debounce.timeout.connect(self._run_search)
        self.search.textChanged.connect(lambda _: self._debounce.start())
        self.match_lbl = label('', 'Hint')
        prev_btn, next_btn = QPushButton('↑'), QPushButton('↓')
        prev_btn.setToolTip('Previous match (Shift+Enter)')
        next_btn.setToolTip('Next match (Enter)')
        prev_btn.clicked.connect(self._prev)
        next_btn.clicked.connect(self._next)
        QShortcut(QKeySequence('Shift+Return'), self.search, activated=self._prev)
        expand_btn, collapse_btn = QPushButton('Expand level'), QPushButton('Collapse all')
        expand_btn.clicked.connect(self._expand_level)
        collapse_btn.clicked.connect(lambda: self.tree.collapseAll())
        bar.addWidget(self.search, 1)
        bar.addWidget(self.match_lbl)
        bar.addWidget(prev_btn)
        bar.addWidget(next_btn)
        bar.addSpacing(8)
        bar.addWidget(expand_btn)
        bar.addWidget(collapse_btn)
        lay.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        tree_panel = Panel()
        tl = QVBoxLayout(tree_panel)
        tl.setContentsMargins(1, 1, 1, 1)
        self.model = JsonModel()
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setAnimated(False)
        self.tree.setIndentation(18)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.tree.header().resizeSection(0, 280)
        self.tree.header().resizeSection(2, 70)
        self.tree.selectionModel().currentChanged.connect(self._on_current)
        tl.addWidget(self.tree)
        split.addWidget(tree_panel)

        detail = Panel()
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(14, 12, 14, 14)
        dl.setSpacing(8)
        top = QHBoxLayout()
        self.kind_lbl = label('', 'Badge')
        self.size_lbl = label('', 'Hint')
        copy_path, copy_val = QPushButton('Copy path'), QPushButton('Copy value')
        copy_path.clicked.connect(self._copy_path)
        copy_val.clicked.connect(self._copy_value)
        export_val = QPushButton('Export…')
        export_val.setToolTip('Save this value to a file')
        export_val.clicked.connect(self._export_value)
        top.addWidget(self.kind_lbl)
        top.addWidget(self.size_lbl)
        top.addStretch(1)
        top.addWidget(copy_path)
        top.addWidget(copy_val)
        top.addWidget(export_val)
        dl.addLayout(top)
        self.path_lbl = label('$', selectable=True)
        self.path_lbl.setFont(theme.mono_font(10))
        self.path_lbl.setWordWrap(True)
        self.path_lbl.setStyleSheet(f'color: {theme.ACCENT};')
        dl.addWidget(self.path_lbl)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(theme.mono_font(10))
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        dl.addWidget(self.text, 1)
        split.addWidget(detail)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 500])
        lay.addWidget(split, 1)

    def set_data(self, data: Any) -> None:
        self._data = data
        self.model.set_data(data)
        self._matches, self._match_i = [], -1
        self.match_lbl.setText('')
        self.tree.expandToDepth(0)
        first = self.model.index(0, 0)
        if first.isValid():
            self.tree.setCurrentIndex(first)
        if self.search.text():
            self._run_search()

    def _on_current(self, cur: QModelIndex, _prev: QModelIndex) -> None:
        if not cur.isValid():
            return
        n = self.model.node(cur.siblingAtColumn(0))
        self.path_lbl.setText(n.path())
        self.kind_lbl.setText(n.kind if n.kind != 'json' else 'json string')
        self.kind_lbl.setStyleSheet(f'color: {theme.JSON_COLORS.get(n.kind, theme.ACCENT)};')
        if n.kind in ('dict', 'list', 'json'):
            self.size_lbl.setText(f'{n.child_count():,} children')
        elif n.kind == 'str':
            self.size_lbl.setText(f'{len(n.value):,} chars')
        else:
            self.size_lbl.setText('')
        txt = n.full_text()
        if len(txt) > _DETAIL_LIMIT:
            txt = txt[:_DETAIL_LIMIT] + f'\n\n… truncated, {len(txt) - _DETAIL_LIMIT:,} more chars (use Copy value for everything)'
        self.text.setPlainText(txt)

    def _current_node(self) -> Any:
        idx = self.tree.currentIndex()
        return self.model.node(idx.siblingAtColumn(0)) if idx.isValid() else None

    def _copy_path(self) -> None:
        n = self._current_node()
        if n is not None:
            QGuiApplication.clipboard().setText(n.path())

    def _copy_value(self) -> None:
        n = self._current_node()
        if n is not None:
            QGuiApplication.clipboard().setText(n.full_text())

    def _export_value(self) -> None:
        n = self._current_node()
        if n is not None:
            self.export_requested.emit(n.key.strip('[]') or 'value', n.full_text())

    def _expand_level(self) -> None:
        depth = 0
        idx = self.tree.currentIndex()
        while idx.isValid():
            depth += 1
            idx = idx.parent()
        self.tree.expandToDepth(max(1, depth))

    def _run_search(self) -> None:
        q = self.search.text().strip()
        if not q:
            self._matches, self._match_i = [], -1
            self.match_lbl.setText('')
            return
        self._matches = search(self._data, q)
        self._match_i = -1
        if self._matches:
            self._next()
        else:
            self.match_lbl.setText('no matches')

    def _goto(self, i: int) -> None:
        if not self._matches:
            return
        self._match_i = i % len(self._matches)
        idx = QModelIndex()
        for row in self._matches[self._match_i]:
            if idx.isValid():
                self.tree.expand(idx)
            idx = self.model.index(row, 0, idx)
        self.tree.setCurrentIndex(idx)
        self.tree.selectionModel().select(idx, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
        self.tree.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)
        more = '+' if len(self._matches) >= 20000 else ''
        self.match_lbl.setText(f'{self._match_i + 1:,} / {len(self._matches):,}{more}')

    def _next(self) -> None:
        if self._debounce.isActive():
            self._debounce.stop()
            self._run_search()
            return
        self._goto(self._match_i + 1)

    def _prev(self) -> None:
        self._goto(self._match_i - 1)


class TensorPage(QWidget):
    status = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._snap: Snapshot | None = None
        self._info: TensorInfo | None = None
        self._token = 0
        self._job: StatsJob | None = None
        self._slice_spins: list[QSpinBox] = []
        self._stale: set[str] = set()
        self._pool = QThreadPool.globalInstance()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 16)
        lay.setSpacing(10)

        head = QHBoxLayout()
        self.title = label('No tensor selected', 'Title')
        self.title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.title.setMinimumWidth(120)
        head.addWidget(self.title)
        head.addSpacing(8)
        self.badges = QHBoxLayout()
        self.badges.setSpacing(6)
        head.addLayout(self.badges)
        head.addStretch(1)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QPushButton] = {}
        for i, (mode, tip) in enumerate(
            (
                ('Grid', 'Values as a scrollable table'),
                ('Image', 'The 2D slice through a colormap'),
                ('Points', 'Rows or columns as a 2D/3D point cloud'),
            )
        ):
            b = QPushButton(mode)
            b.setCheckable(True)
            b.setObjectName('Segment')
            b.setToolTip(tip)
            self.mode_group.addButton(b, i)
            self.mode_buttons[mode] = b
            head.addWidget(b)
        self.mode_buttons['Grid'].setChecked(True)
        self.mode_group.idClicked.connect(self._on_mode)
        lay.addLayout(head)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.slice_box = QHBoxLayout()
        self.slice_box.setSpacing(6)
        controls.addLayout(self.slice_box)
        controls.addStretch(1)
        self.grid_controls = QWidget()
        outer_controls = controls
        controls = QHBoxLayout(self.grid_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        outer_controls.addWidget(self.grid_controls)
        controls.addWidget(label('Format', 'Subtitle'))
        self.fmt = QComboBox()
        self.fmt.addItems(TensorTableModel.FORMATS)
        controls.addWidget(self.fmt)
        controls.addWidget(label('Precision', 'Subtitle'))
        self.prec = QSpinBox()
        self.prec.setRange(0, 17)
        self.prec.setValue(5)
        controls.addWidget(self.prec)
        self.heat = QCheckBox('Heatmap')
        controls.addWidget(self.heat)
        controls.addSpacing(12)
        controls.addWidget(label('Go to', 'Subtitle'))
        self.go_row, self.go_col = QSpinBox(), QSpinBox()
        for sb, tip in ((self.go_row, 'Row'), (self.go_col, 'Column')):
            sb.setToolTip(tip)
            sb.setMinimumWidth(80)
            sb.setPrefix(f'{tip[0].lower()} ')
            controls.addWidget(sb)
        go = QPushButton('Jump')
        go.clicked.connect(self._jump)
        self.go_row.editingFinished.connect(self._jump)
        self.go_col.editingFinished.connect(self._jump)
        controls.addWidget(go)
        lay.addLayout(outer_controls)

        split = QSplitter(Qt.Orientation.Horizontal)
        grid_panel = Panel()
        gl = QVBoxLayout(grid_panel)
        gl.setContentsMargins(1, 1, 1, 1)
        self.model = TensorTableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setWordWrap(False)
        self.table.setShowGrid(True)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        for h in (self.table.horizontalHeader(), self.table.verticalHeader()):
            h.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
            h.setHighlightSections(True)
        self.table.horizontalHeader().setDefaultSectionSize(96)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.verticalHeader().setMinimumWidth(56)
        self.table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.table.selectionModel().currentChanged.connect(self._on_cell)
        QShortcut(QKeySequence.StandardKey.Copy, self.table, activated=self._copy_selection)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.table)
        self.image = ImagePanel(self._value_text)
        self.points = PointCloudPanel()
        self.stack.addWidget(self.image)
        self.stack.addWidget(self.points)
        for v in (self.image, self.points):
            v.status.connect(self.status)
            v.activated.connect(self._open_cell)
        gl.addWidget(self.stack)
        split.addWidget(grid_panel)

        side = Panel()
        side.setMinimumWidth(260)
        side.setMaximumWidth(380)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(16, 14, 16, 14)
        sl.setSpacing(4)
        sl.addWidget(label('STATISTICS', 'SectionTitle'))
        self.stat_rows: dict[str, StatRow] = {}
        for k in ('Elements', 'Min', 'Max', 'Mean', 'Std', 'Abs max', 'Zeros', 'NaN', 'Inf'):
            r = StatRow(k)
            self.stat_rows[k] = r
            sl.addWidget(r)
        sl.addSpacing(12)
        hh = QHBoxLayout()
        hh.addWidget(label('DISTRIBUTION', 'SectionTitle'))
        hh.addStretch(1)
        self.log_cb = QCheckBox('log')
        hh.addWidget(self.log_cb)
        sl.addLayout(hh)
        self.hist = Histogram()
        self.log_cb.toggled.connect(self.hist.set_log)
        sl.addWidget(self.hist)
        sl.addSpacing(12)
        sl.addWidget(label('STORAGE', 'SectionTitle'))
        self.storage_rows: dict[str, StatRow] = {}
        for k in ('Offset', 'Bytes', 'Item size', 'Strides'):
            r = StatRow(k)
            self.storage_rows[k] = r
            sl.addWidget(r)
        sl.addStretch(1)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1000, 300])
        lay.addWidget(split, 1)

        self.fmt.currentTextChanged.connect(self._apply_format)
        self.prec.valueChanged.connect(self._apply_format)
        self.heat.toggled.connect(self.model.set_heatmap)

    # Lifecycle

    def clear(self) -> None:
        self._cancel_job()
        self._pool.waitForDone()
        self._snap = None
        self._info = None
        self.points.cancel()
        self.model.set_tensor(None, 'float32')
        self.image.clear()
        self.points.set_view(None)
        self.title.setText('No tensor selected')
        clear_layout(self.badges)
        clear_layout(self.slice_box)

    def set_snapshot(self, snap: Snapshot) -> None:
        self.clear()
        self._snap = snap

    def _cancel_job(self) -> None:
        if self._job is not None:
            self._job.cancelled = True
            self._job = None

    def show_tensor(self, name: str) -> None:
        if self._snap is None or name not in self._snap.tensors:
            return
        info = self._snap.tensors[name]
        self._info = info
        tensor = self._snap.tensor(name)
        self.status.emit(f'{name}  ·  {info.shape_str} {info.dtype}  ·  click a cell to inspect it')
        self.title.setText(name)
        self.title.setToolTip(name)
        clear_layout(self.badges)
        for text in (info.shape_str, info.dtype, human_bytes(info.nbytes)):
            self.badges.addWidget(label(text, 'Badge'))

        self.model.set_tensor(tensor, info.dtype)
        self._build_slice_controls(info.shape)
        self._sync_goto_ranges()
        self._views_changed()
        self.heat.setEnabled(info.dtype != 'boolean')
        self.prec.setEnabled(is_float(info.dtype))

        item_size = tensor.dtype.size
        strides = tuple(tensor.strides) if tensor.rank else ()
        self.storage_rows['Offset'].set(f'{info.offset:,}')
        self.storage_rows['Bytes'].set(f'{info.nbytes:,}')
        self.storage_rows['Item size'].set(f'{item_size} B')
        self.storage_rows['Strides'].set(str(strides) if strides else '—')

        for r in self.stat_rows.values():
            r.set('…')
        self.stat_rows['Elements'].set(f'{info.numel:,}')
        self.hist.set_data(None, None)
        self.model.set_absmax(0.0)
        self._cancel_job()
        self._token += 1
        job = StatsJob(self._token, tensor, info.dtype)
        job.signals.done.connect(self._on_stats)
        self._job = job
        self._pool.start(job)
        self.table.scrollToTop()

    def current_name(self) -> str | None:
        return self._info.name if self._info is not None else None

    def slice_suffix(self) -> str:
        """Filename suffix naming the current slice, e.g. '[3]' for dim0 = 3."""
        if not self._slice_spins:
            return ''
        return '[' + ','.join(str(sb.value()) for sb in self._slice_spins) + ']'

    def _build_slice_controls(self, shape: tuple[int, ...]) -> None:
        clear_layout(self.slice_box)
        self._slice_spins = []
        if len(shape) <= 2:
            if not shape:
                desc = 'scalar'
            elif len(shape) == 1:
                desc = f'1D · {shape[0]:,} rows'
            else:
                desc = f'{shape[0]:,} rows × {shape[1]:,} cols'
            self.slice_box.addWidget(label(desc, 'Hint'))
            return
        self.slice_box.addWidget(label('Slice', 'Subtitle'))
        for d, n in enumerate(shape[:-2]):
            sb = QSpinBox()
            sb.setRange(0, max(0, n - 1))
            sb.setPrefix(f'dim{d} ')
            sb.setSuffix(f' / {n}')
            sb.valueChanged.connect(self._on_slice)
            self._slice_spins.append(sb)
            self.slice_box.addWidget(sb)
        self.slice_box.addWidget(label(f'→ {shape[-2]:,} × {shape[-1]:,}', 'Hint'))

    def _on_slice(self) -> None:
        self.model.set_slice(tuple(sb.value() for sb in self._slice_spins))
        self._views_changed()

    # Grid / Image / Points

    def _views_changed(self) -> None:
        """The 2D slice changed: update which modes apply and refresh the visible one, the others refresh lazily."""
        rows, cols = self.model.rowCount(), self.model.columnCount()
        can_image = rows * cols > 1
        can_points = bool(point_orientations(rows, cols))
        self.mode_buttons['Image'].setEnabled(can_image)
        self.mode_buttons['Points'].setEnabled(can_points)
        self.mode_buttons['Points'].setToolTip(
            'Rows or columns as a 2D/3D point cloud' if can_points else 'Needs a 2D slice with at least 3 points of 2+ coordinates'
        )
        self._stale = {'Image', 'Points'}
        mode = self._mode()
        if (mode == 'Image' and not can_image) or (mode == 'Points' and not can_points):
            self.mode_buttons['Grid'].setChecked(True)
        self._on_mode(self.mode_group.checkedId())

    def _mode(self) -> str:
        b = self.mode_group.checkedButton()
        return b.text() if b is not None else 'Grid'

    def _on_mode(self, idx: int) -> None:
        self.stack.setCurrentIndex(max(0, idx))
        mode = self._mode()
        self.grid_controls.setVisible(mode == 'Grid')
        if mode in self._stale:
            self._stale.discard(mode)
            view = self.model.view_tensor
            if mode == 'Image':
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    self.image.set_view(view)
                finally:
                    QApplication.restoreOverrideCursor()
            else:
                self.points.set_view(view)
        if mode != 'Points':
            self.points.cloud.set_spin(False)

    def _value_text(self, row: int, col: int) -> str:
        v = self.model.value(row, col)
        return self.model.format_value(v, row, col)

    def _open_cell(self, row: int, col: int) -> None:
        self.mode_buttons['Grid'].setChecked(True)
        self._on_mode(0)
        self.go_row.setValue(row)
        self.go_col.setValue(col)
        self._jump()

    def _sync_goto_ranges(self) -> None:
        self.go_row.setRange(0, max(0, self.model.rowCount() - 1))
        self.go_col.setRange(0, max(0, self.model.columnCount() - 1))

    def _apply_format(self) -> None:
        self.model.set_format(self.fmt.currentText(), self.prec.value())

    def _on_stats(self, token: int, res: dict) -> None:
        if token != self._token:
            return
        self._job = None
        if 'error' in res:
            self.status.emit(f'Stats failed: {res["error"]}')
            return

        def f(k: str) -> str:
            v = res.get(k)
            return '—' if v is None else f'{v:.6g}'

        self.stat_rows['Min'].set(f('min'))
        self.stat_rows['Max'].set(f('max'))
        self.stat_rows['Mean'].set(f('mean'))
        self.stat_rows['Std'].set(f('std'))
        self.stat_rows['Abs max'].set(f('absmax'))
        n = max(1, res.get('numel', 1))
        z = res.get('zeros', 0)
        self.stat_rows['Zeros'].set(f'{z:,} ({z / n:.2%})')
        self.stat_rows['NaN'].set(f'{res.get("nan", 0):,}')
        self.stat_rows['Inf'].set(f'{res.get("inf", 0):,}')
        self.hist.set_data(res.get('hist'), res.get('hist_range'))
        self.model.set_absmax(float(res.get('absmax') or 0.0))

    def _index_tuple(self, row: int, col: int) -> tuple[int, ...]:
        if self._info is None:
            return ()
        nd = len(self._info.shape)
        if nd == 0:
            return ()
        if nd == 1:
            return (row,)
        lead = tuple(sb.value() for sb in self._slice_spins) if nd > 2 else ()
        return (*lead, row, col)

    def _on_cell(self, cur: QModelIndex, _prev: QModelIndex) -> None:
        if not cur.isValid() or self._info is None:
            return
        r, c = cur.row(), cur.column()
        v = self.model.value(r, c)
        full = repr(float(v)) if is_float(self.model.dtype_name) else str(v)
        idx = ', '.join(map(str, self._index_tuple(r, c)))
        flat = 0
        stride = 1
        for i, d in zip(reversed(self._index_tuple(r, c)), reversed(self._info.shape), strict=False):
            flat += i * stride
            stride *= d
        self.status.emit(f'[{idx}]   value {full}   bits {self.model.raw_bits(r, c)}   flat #{flat:,}')

    def _jump(self) -> None:
        idx = self.model.index(self.go_row.value(), self.go_col.value())
        if idx.isValid():
            self.table.setCurrentIndex(idx)
            self.table.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _copy_selection(self) -> None:
        ranges = self.table.selectionModel().selection()
        if ranges.isEmpty():
            return
        r0, r1 = min(r.top() for r in ranges), max(r.bottom() for r in ranges)
        c0, c1 = min(r.left() for r in ranges), max(r.right() for r in ranges)
        rows, cols = range(r0, r1 + 1), range(c0, c1 + 1)
        if len(rows) * len(cols) > 2_000_000:
            self.status.emit('Selection too large to copy')
            return
        lines = ['\t'.join(self.model.format_value(self.model.value(r, c), r, c) for c in cols) for r in rows]
        QGuiApplication.clipboard().setText('\n'.join(lines))
        self.status.emit(f'Copied {len(rows)}×{len(cols)} cells')
