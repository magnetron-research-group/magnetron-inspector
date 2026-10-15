from __future__ import annotations

import json
from typing import Any

from PyQt6.QtCore import QAbstractItemModel, QModelIndex, Qt
from PyQt6.QtGui import QColor, QFont

from . import theme

_ROOT = QModelIndex()

_PREVIEW_LEN = 160


def _try_json_string(v: str) -> Any | None:
    s = v.lstrip()
    if len(s) < 2 or s[0] not in '{[':
        return None
    try:
        parsed = json.loads(v)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict | list) else None


def kind_of(v: Any) -> str:
    if v is None:
        return 'null'
    if isinstance(v, bool):
        return 'bool'
    if isinstance(v, int):
        return 'int'
    if isinstance(v, float):
        return 'float'
    if isinstance(v, dict):
        return 'dict'
    if isinstance(v, list):
        return 'list'
    return 'str'


def _children_of(v: Any) -> list[tuple[str, Any]] | None:
    if isinstance(v, dict):
        return [(str(k), x) for k, x in v.items()]
    if isinstance(v, list):
        return [(f'[{i}]', x) for i, x in enumerate(v)]
    return None


class JsonNode:
    __slots__ = ('_children', 'embedded', 'key', 'kind', 'parent', 'row', 'value')

    def __init__(self, key: str, value: Any, parent: JsonNode | None, row: int) -> None:
        self.key = key
        self.value = value
        self.parent = parent
        self.row = row
        self.kind = kind_of(value)
        self.embedded: Any | None = _try_json_string(value) if self.kind == 'str' else None
        if self.embedded is not None:
            self.kind = 'json'
        self._children: list[JsonNode] | None = None

    @property
    def container(self) -> Any:
        return self.embedded if self.kind == 'json' else self.value

    def child_count(self) -> int:
        c = self.container
        return len(c) if isinstance(c, dict | list) else 0

    def children(self) -> list[JsonNode]:
        if self._children is None:
            items = _children_of(self.container) or []
            self._children = [JsonNode(k, v, self, i) for i, (k, v) in enumerate(items)]
        return self._children

    def path(self) -> str:
        parts: list[str] = []
        node: JsonNode | None = self
        while node is not None and node.parent is not None:
            parts.append(node.key)
            node = node.parent
        out = ''
        for p in reversed(parts):
            out += p if p.startswith('[') else (f'.{p}' if out else p)
        return out or '$'

    def preview(self) -> str:
        c = self.container
        if self.kind == 'dict':
            return f'{{ {len(c)} keys }}'
        if self.kind == 'list':
            return f'[ {len(c)} items ]'
        if self.kind == 'json':
            n = len(c)
            inner = f'{n} keys' if isinstance(c, dict) else f'{n} items'
            return f'embedded JSON · {inner} · {len(self.value):,} chars'
        if self.kind == 'str':
            s = self.value.replace('\n', '⏎')
            return json.dumps(s[:_PREVIEW_LEN], ensure_ascii=False)[:-1] + ('…"' if len(s) > _PREVIEW_LEN else '"')
        return json.dumps(self.value)

    def full_text(self) -> str:
        if self.kind in ('dict', 'list'):
            return json.dumps(self.value, indent=2, ensure_ascii=False)
        if self.kind == 'json':
            return json.dumps(self.embedded, indent=2, ensure_ascii=False)
        if self.kind == 'str':
            return self.value
        return json.dumps(self.value)


class JsonModel(QAbstractItemModel):
    COLUMNS = ('Key', 'Value', 'Type')

    def __init__(self, data: Any | None = None) -> None:
        super().__init__()
        self._root = JsonNode('$', data if data is not None else {}, None, 0)
        self._key_font = QFont()
        self._key_font.setWeight(QFont.Weight.DemiBold)
        self._mono = theme.mono_font(10)
        self._colors = {k: QColor(v) for k, v in theme.JSON_COLORS.items()}
        self._dim = QColor(theme.TEXT_DIM)

    def set_data(self, data: Any) -> None:
        self.beginResetModel()
        self._root = JsonNode('$', data, None, 0)
        self.endResetModel()

    def node(self, index: QModelIndex) -> JsonNode:
        return index.internalPointer() if index.isValid() else self._root

    def index(self, row: int, column: int, parent: QModelIndex = _ROOT) -> QModelIndex:
        p = self.node(parent)
        kids = p.children()
        if 0 <= row < len(kids):
            return self.createIndex(row, column, kids[row])
        return QModelIndex()

    def parent(self, index: QModelIndex = _ROOT) -> QModelIndex:  # type: ignore[override]
        if not index.isValid():
            return QModelIndex()
        p = index.internalPointer().parent
        if p is None or p is self._root:
            return QModelIndex()
        return self.createIndex(p.row, 0, p)

    def rowCount(self, parent: QModelIndex = _ROOT) -> int:
        if parent.column() > 0:
            return 0
        return self.node(parent).child_count()

    def hasChildren(self, parent: QModelIndex = _ROOT) -> bool:
        return self.rowCount(parent) > 0

    def columnCount(self, parent: QModelIndex = _ROOT) -> int:
        return 3

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        n: JsonNode = index.internalPointer()
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return n.key
            if col == 1:
                return n.preview()
            return n.kind
        if role == Qt.ItemDataRole.ForegroundRole:
            if col == 1:
                return self._dim if n.kind in ('dict', 'list') else self._colors.get(n.kind)
            if col == 2:
                return self._colors.get(n.kind)
        if role == Qt.ItemDataRole.FontRole:
            return self._key_font if col == 0 else self._mono
        if role == Qt.ItemDataRole.ToolTipRole and col == 1 and n.kind == 'str' and len(n.value) > _PREVIEW_LEN:
            return n.value[:2000]
        return None


def search(data: Any, needle: str, limit: int = 20000) -> list[tuple[int, ...]]:
    """Return row paths of nodes whose key or scalar value contains needle (case insensitive)."""
    needle = needle.lower()
    out: list[tuple[int, ...]] = []

    def walk(v: Any, path: tuple[int, ...]) -> None:
        items = _children_of(v)
        if items is None:
            return
        for i, (k, x) in enumerate(items):
            if len(out) >= limit:
                return
            p = (*path, i)
            embedded = _try_json_string(x) if isinstance(x, str) else None
            scalar = None if isinstance(x, dict | list) or embedded is not None else x
            if needle in k.lower() or (scalar is not None and needle in str(scalar).lower()):
                out.append(p)
            if embedded is not None:
                walk(embedded, p)
            elif isinstance(x, dict | list):
                walk(x, p)

    walk(data, ())
    return out
