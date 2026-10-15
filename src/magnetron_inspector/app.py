from __future__ import annotations

import os
import shutil
import subprocess
import sys

from PyQt6.QtCore import QModelIndex, QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from . import export, icons, theme
from .layout import Layout, build_layout
from .layout_page import LayoutPage
from .pages import MetadataPage, OverviewPage, TensorPage
from .snapshot import Snapshot, human_bytes
from .tensor_model import TENSOR_ROLE, TensorFilterProxy, TensorTreeModel
from .widgets import Panel, label

_MAX_RECENT = 10


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Magnetron Inspector')
        self.setWindowIcon(icons.app_icon())
        self.resize(1500, 920)
        self.setAcceptDrops(True)
        self.settings = QSettings('magnetron', 'inspector')
        self.snap: Snapshot | None = None
        self.layout_data: Layout | None = None

        self._build_toolbar()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_empty())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setContentsMargins(10, 10, 10, 10)
        split.addWidget(self._build_sidebar())

        self.tabs = QTabWidget()
        self.overview = OverviewPage()
        self.metadata = MetadataPage()
        self.tensor = TensorPage()
        self.tabs.addTab(self.overview, 'Overview')
        self.tabs.addTab(self.metadata, 'Metadata')
        self.tabs.addTab(self.tensor, 'Tensor')
        self.layout_page = LayoutPage()
        self.tabs.addTab(self.layout_page, 'Layout')
        self.overview.tensor_requested.connect(self.select_tensor)
        self.layout_page.tensor_requested.connect(self.select_tensor)
        self.metadata.export_requested.connect(self._export_metadata_node)
        self.tensor.status.connect(lambda s: self.status_lbl.setText(s))
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([460, 1040])
        self.stack.addWidget(split)

        self.status_lbl = label('Open a .mag snapshot to begin')
        self.status_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_lbl.setFont(theme.mono_font(9))
        self.file_lbl = label('')
        self.statusBar().addWidget(self.status_lbl, 1)
        self.statusBar().addPermanentWidget(self.file_lbl)

    # UI construction

    def _build_toolbar(self) -> None:
        tb = QToolBar('Main')
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(tb)
        mark = label()
        mark.setPixmap(icons.logo_pixmap(30, dpr=self.devicePixelRatioF()))
        mark.setStyleSheet('padding-left: 4px;')
        tb.addWidget(mark)
        brand = label('Magnetron Inspector')
        brand.setStyleSheet(f'color: {theme.TEXT}; font-weight: 700; font-size: 11pt; padding: 0 10px 0 2px;')
        tb.addWidget(brand)

        open_act = QAction('Open…', self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.open_dialog)
        tb.addAction(open_act)

        self.recent_menu = QMenu(self)
        self.recent_btn = QToolButton()
        self.recent_btn.setText('Recent ▾')
        self.recent_btn.setMenu(self.recent_menu)
        self.recent_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tb.addWidget(self.recent_btn)
        self._refresh_recent()

        reload_act = QAction('Reload', self)
        reload_act.setShortcut(QKeySequence.StandardKey.Refresh)
        reload_act.triggered.connect(self.reload)
        tb.addAction(reload_act)

        self.export_menu = QMenu(self)
        self.export_menu.aboutToShow.connect(self._refresh_export_menu)
        self.export_actions: dict[str, QAction] = {}
        for key, text, slot in (
            ('meta', 'Metadata as JSON (pretty)…', lambda: self._export_metadata(raw=False)),
            ('meta_raw', 'Metadata as JSON (exact bytes from file)…', lambda: self._export_metadata(raw=True)),
            (None, None, None),
            ('slice', 'Current tensor slice as CSV…', lambda: self._export_tensor(whole=False)),
            ('tensor', 'Whole tensor as CSV…', lambda: self._export_tensor(whole=True)),
            (None, None, None),
            ('manifest', 'Tensor manifest as CSV…', self._export_manifest),
            ('layout', 'Byte layout as CSV…', self._export_layout),
        ):
            if key is None:
                self.export_menu.addSeparator()
                continue
            act = self.export_menu.addAction(text)
            act.triggered.connect(slot)
            self.export_actions[key] = act
        self.export_btn = QToolButton()
        self.export_btn.setText('Export ▾')
        self.export_btn.setMenu(self.export_menu)
        self.export_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_btn.setEnabled(False)
        tb.addWidget(self.export_btn)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self.path_lbl = label('')
        tb.addWidget(self.path_lbl)

        find_act = QAction(self)
        find_act.setShortcut(QKeySequence.StandardKey.Find)
        find_act.triggered.connect(self._focus_search)
        self.addAction(find_act)

    def _build_empty(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch(1)
        mark = label()
        mark.setPixmap(icons.logo_pixmap(150, dpr=self.devicePixelRatioF()))
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(mark)
        t = label('Magnetron Inspector', 'Title')
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        s = label('Drop a .mag snapshot here, or open one to explore its metadata and tensors', 'Subtitle')
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(s)
        lay.addSpacing(16)
        btn = QPushButton('Open snapshot…')
        btn.setObjectName('Primary')
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(self.open_dialog)
        btn.setFixedWidth(200)
        lay.addWidget(btn, 0, Qt.AlignmentFlag.AlignCenter)
        lay.addStretch(2)
        return w

    def _build_sidebar(self) -> QWidget:
        panel = Panel()
        panel.setMinimumWidth(300)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 12, 10, 10)
        lay.setSpacing(8)
        lay.addWidget(label('TENSORS', 'SectionTitle'))
        self.tensor_filter = QLineEdit()
        self.tensor_filter.setPlaceholderText('Filter tensors…  e.g. layers.0.attn')
        self.tensor_filter.setClearButtonEnabled(True)
        lay.addWidget(self.tensor_filter)

        self.tree_model = TensorTreeModel()
        self.proxy = TensorFilterProxy()
        self.proxy.setSourceModel(self.tree_model)
        self.tree = QTreeView()
        self.tree.setModel(self.proxy)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.setIndentation(14)
        self.tree.setSortingEnabled(True)
        self.tree.header().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            self.tree.header().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.selectionModel().currentChanged.connect(self._on_tree_current)
        lay.addWidget(self.tree, 1)
        self.tree_footer = label('', 'Hint')
        lay.addWidget(self.tree_footer)

        self._filter_timer = QTimer(self, singleShot=True, interval=150)
        self._filter_timer.timeout.connect(self._apply_filter)
        self.tensor_filter.textChanged.connect(lambda _: self._filter_timer.start())
        return panel

    # Actions

    def _focus_search(self) -> None:
        if self.tabs.currentWidget() is self.metadata:
            self.metadata.search.setFocus()
            self.metadata.search.selectAll()
        else:
            self.tensor_filter.setFocus()
            self.tensor_filter.selectAll()

    def _apply_filter(self) -> None:
        text = self.tensor_filter.text().strip()
        self.proxy.setFilterFixedString(text)
        if text:
            self.tree.expandAll()
        self._update_footer()

    def _update_footer(self) -> None:
        if self.snap is None:
            self.tree_footer.setText('')
            return
        total = len(self.snap.tensors)
        shown = self._count_visible_leaves(QModelIndex())
        self.tree_footer.setText(f'{shown} of {total} tensors' if shown != total else f'{total} tensors')

    def _count_visible_leaves(self, parent: QModelIndex) -> int:
        n = 0
        for r in range(self.proxy.rowCount(parent)):
            idx = self.proxy.index(r, 0, parent)
            n += 1 if idx.data(TENSOR_ROLE) else self._count_visible_leaves(idx)
        return n

    def _on_tree_current(self, cur: QModelIndex, _prev: QModelIndex) -> None:
        name = cur.siblingAtColumn(0).data(TENSOR_ROLE) if cur.isValid() else None
        if name:
            self.tensor.show_tensor(name)
            self.tabs.setCurrentWidget(self.tensor)

    def select_tensor(self, name: str) -> None:
        self.tensor_filter.clear()
        self._apply_filter()
        src = self._find_source(name)
        if src is None:
            return
        idx = self.proxy.mapFromSource(src)
        self.tree.scrollTo(idx)
        self.tree.setCurrentIndex(idx)

    def _find_source(self, name: str) -> QModelIndex | None:
        def walk(parent: QModelIndex) -> QModelIndex | None:
            for r in range(self.tree_model.rowCount(parent)):
                idx = self.tree_model.index(r, 0, parent)
                if idx.data(TENSOR_ROLE) == name:
                    return idx
                full = idx.data(Qt.ItemDataRole.UserRole + 1) or ''
                if name.startswith(full + '.'):
                    found = walk(idx)
                    if found is not None:
                        return found
            return None

        return walk(QModelIndex())

    def open_dialog(self) -> None:
        start = self.settings.value('last_dir', os.path.expanduser('~/Desktop'))
        path, _ = QFileDialog.getOpenFileName(self, 'Open Magnetron snapshot', start, 'Magnetron snapshots (*.mag);;All files (*)')
        if path:
            self.open_path(path)

    def reload(self) -> None:
        if self.snap is not None:
            self.open_path(self.snap.path)

    def open_path(self, path: str) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            new = Snapshot(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, 'Could not open snapshot', f'{path}\n\n{e}')
            return
        try:
            self._close_snapshot()
            self.snap = new
            self.tensor.set_snapshot(new)
            self.tree_model.populate(new)
            self.tree.expandToDepth(0)
            self.overview.show_snapshot(new)
            self.metadata.set_data(new.manifest)
            try:
                self.layout_data = build_layout(new)
            except Exception as e:
                self.layout_data = None
                self.status_lbl.setText(f'Could not decode the byte layout: {e}')
            self.layout_page.set_layout(self.layout_data)
            self.export_btn.setEnabled(True)
            self._update_footer()
            self.stack.setCurrentIndex(1)
            self.tabs.setCurrentWidget(self.overview)
            self.setWindowTitle(f'{os.path.basename(path)} — Magnetron Inspector')
            self.path_lbl.setText(path)
            self.file_lbl.setText(f'{len(new.tensors)} tensors · {human_bytes(new.file_size)}')
            if new.validation_error is not None:
                self.status_lbl.setText(f'Validation: {new.validation_error}')
            elif self.layout_data is not None:
                n = len(self.layout_data.issues)
                self.status_lbl.setText('Ready' if not n else f'Ready · {n} layout issue(s), see the Layout tab')
            self.settings.setValue('last_dir', os.path.dirname(new.path))
            self._push_recent(new.path)
        finally:
            QApplication.restoreOverrideCursor()

    def _close_snapshot(self) -> None:
        # Views must drop every tensor borrowed from the mmap before the reader closes
        self.tensor.clear()
        self.layout_page.set_layout(None)
        self.layout_data = None
        self.export_btn.setEnabled(False)
        if self.snap is not None:
            self.snap.close()
            self.snap = None

    # Export

    def _save_path(self, title: str, name: str, filters: str) -> str | None:
        start = self.settings.value('export_dir', os.path.dirname(self.snap.path) if self.snap else os.path.expanduser('~'))
        safe = ''.join(c if c.isalnum() or c in '._-' else '_' for c in name)
        path, _ = QFileDialog.getSaveFileName(self, title, os.path.join(start, safe), filters)
        if path:
            self.settings.setValue('export_dir', os.path.dirname(path))
        return path or None

    def _stem(self) -> str:
        return os.path.splitext(os.path.basename(self.snap.path))[0] if self.snap else 'snapshot'

    def _refresh_export_menu(self) -> None:
        has_tensor = self.tensor.current_name() is not None
        self.export_actions['slice'].setEnabled(has_tensor)
        self.export_actions['tensor'].setEnabled(has_tensor)
        self.export_actions['layout'].setEnabled(self.layout_data is not None)
        name = self.tensor.current_name()
        self.export_actions['slice'].setText(f'Current slice of {name} as CSV…' if name else 'Current tensor slice as CSV…')
        self.export_actions['tensor'].setText(f'Whole {name} as CSV…' if name else 'Whole tensor as CSV…')

    def _done(self, what: str, path: str) -> None:
        self.status_lbl.setText(f'Exported {what} to {path}')

    def _fail(self, path: str, e: Exception) -> None:
        QMessageBox.critical(self, 'Export failed', f'{path}\n\n{e}')

    def _export_metadata(self, raw: bool) -> None:
        if self.snap is None:
            return
        path = self._save_path('Export metadata', f'{self._stem()}.metadata.json', 'JSON (*.json);;All files (*)')
        if not path:
            return
        try:
            export.write_metadata_json(path, self.snap, raw=raw)
        except OSError as e:
            self._fail(path, e)
            return
        self._done('metadata', path)

    def _export_metadata_node(self, key: str, text: str) -> None:
        path = self._save_path('Export metadata value', f'{key}.json', 'JSON (*.json);;Text (*.txt);;All files (*)')
        if not path:
            return
        try:
            export.write_json_value(path, text)
        except OSError as e:
            self._fail(path, e)
            return
        self._done(key, path)

    def _export_tensor(self, whole: bool) -> None:
        name = self.tensor.current_name()
        if self.snap is None or name is None:
            return
        tensor = self.snap.tensor(name) if whole else self.tensor.model.view_tensor
        if tensor is None:
            return
        suffix = '' if whole else self.tensor.slice_suffix()
        path = self._save_path('Export tensor as CSV', f'{name}{suffix}.csv', 'CSV (*.csv);;All files (*)')
        if not path:
            return
        rows = export.as_matrix(tensor).shape[0]
        dlg = QProgressDialog(f'Writing {name}…', 'Cancel', 0, rows, self)
        dlg.setWindowTitle('Export CSV')
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(400)

        def progress(done: int, total: int) -> bool:
            dlg.setValue(done)
            QApplication.processEvents()
            return not dlg.wasCanceled()

        try:
            finished = export.write_tensor_csv(path, tensor, progress)
        except OSError as e:
            dlg.close()
            self._fail(path, e)
            return
        dlg.close()
        if not finished:
            os.remove(path)  # Our own partial output
            self.status_lbl.setText('CSV export cancelled')
            return
        shape = 'x'.join(map(str, export.as_matrix(tensor).shape))
        self._done(f'{name} ({shape})', path)

    def _export_manifest(self) -> None:
        if self.snap is None:
            return
        path = self._save_path('Export tensor manifest', f'{self._stem()}.tensors.csv', 'CSV (*.csv);;All files (*)')
        if not path:
            return
        try:
            export.write_manifest_csv(path, self.snap, self.layout_data)
        except OSError as e:
            self._fail(path, e)
            return
        self._done('tensor manifest', path)

    def _export_layout(self) -> None:
        if self.layout_data is None:
            return
        path = self._save_path('Export byte layout', f'{self._stem()}.layout.csv', 'CSV (*.csv);;All files (*)')
        if not path:
            return
        try:
            export.write_layout_csv(path, self.layout_data)
        except OSError as e:
            self._fail(path, e)
            return
        self._done('byte layout', path)

    # Recent files

    def _recent(self) -> list[str]:
        v = self.settings.value('recent', [])
        return [v] if isinstance(v, str) else list(v or [])

    def _push_recent(self, path: str) -> None:
        items = [p for p in self._recent() if p != path]
        items.insert(0, path)
        self.settings.setValue('recent', items[:_MAX_RECENT])
        self._refresh_recent()

    def _refresh_recent(self) -> None:
        self.recent_menu.clear()
        items = [p for p in self._recent() if os.path.exists(p)]
        for p in items:
            act = self.recent_menu.addAction(os.path.basename(p))
            act.setToolTip(p)
            act.triggered.connect(lambda _=False, p=p: self.open_path(p))
        self.recent_btn.setEnabled(bool(items))

    # Drag and drop

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent) -> None:
        for u in e.mimeData().urls():
            if u.isLocalFile():
                self.open_path(u.toLocalFile())
                break

    def closeEvent(self, e) -> None:  # noqa: ANN001
        self._close_snapshot()
        super().closeEvent(e)


def install_desktop_entry() -> None:
    """Register a .desktop entry and icons so Linux shells (GNOME/KDE, Wayland) show the logo in the dock and launcher."""
    data = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
    hicolor = os.path.join(data, 'icons', 'hicolor')
    app_dir = os.path.join(data, 'applications')
    os.makedirs(app_dir, exist_ok=True)
    written: list[str] = []
    # Every common size plus the SVG, so the shell never upscales or falls back to a stale size
    for size in icons.ICON_SIZES:
        d = os.path.join(hicolor, f'{size}x{size}', 'apps')
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, 'magnetron-inspector.png')
        icons.round_logo(size).save(path)  # Exact pixel size, QIcon.pixmap would scale by the display ratio
        written.append(path)
    scalable = os.path.join(hicolor, 'scalable', 'apps')
    os.makedirs(scalable, exist_ok=True)
    svg = os.path.join(scalable, 'magnetron-inspector.svg')
    shutil.copyfile(icons.ROUND_LOGO_PATH, svg)
    written.append(svg)
    exe = shutil.which('magnetron-inspector') or f'{sys.executable} -m magnetron_inspector'
    entry = os.path.join(app_dir, 'magnetron-inspector.desktop')
    with open(entry, 'w', encoding='utf-8') as f:
        f.write(
            '[Desktop Entry]\nType=Application\nName=Magnetron Inspector\nComment=Inspect Magnetron .mag snapshots\n'
            f'Exec={exe} %f\nIcon=magnetron-inspector\nTerminal=false\nCategories=Development;Science;\n'
            'StartupWMClass=magnetron-inspector\n'
        )
    # A stale icon-theme.cache makes the shell keep serving the previous icon
    os.utime(hicolor)
    tool = shutil.which('gtk-update-icon-cache') or shutil.which('gtk4-update-icon-cache')
    if tool:
        subprocess.run([tool, '-f', '-t', hicolor], check=False, capture_output=True)
    if shutil.which('update-desktop-database'):
        subprocess.run(['update-desktop-database', app_dir], check=False, capture_output=True)
    print(f'Installed {entry}')
    print(f'Installed {len(written)} icons under {hicolor}')


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if '--install-desktop' in argv:
        _ = QApplication(argv[:1])
        install_desktop_entry()
        return 0
    app = QApplication(argv)
    app.setApplicationName('Magnetron Inspector')
    app.setDesktopFileName('magnetron-inspector')
    app.setWindowIcon(icons.app_icon())
    theme.apply(app)
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        win.open_path(argv[1])
    return app.exec()
