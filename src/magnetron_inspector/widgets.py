from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLayout, QSizePolicy, QVBoxLayout, QWidget

from . import theme


def label(text: str = '', obj: str | None = None, *, selectable: bool = False) -> QLabel:
    lbl = QLabel(text)
    if obj:
        lbl.setObjectName(obj)
    if selectable:
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lbl


def clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            continue
        w = item.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


class Panel(QFrame):
    def __init__(self, parent: QWidget | None = None, obj: str = 'Panel') -> None:
        super().__init__(parent)
        self.setObjectName(obj)


class StatCard(Panel):
    def __init__(self, title: str, value: str = '—', parent: QWidget | None = None) -> None:
        super().__init__(parent, 'Card')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self.value_lbl = label(value, 'CardValue', selectable=True)
        self.title_lbl = label(title.upper(), 'CardLabel')
        lay.addWidget(self.title_lbl)
        lay.addWidget(self.value_lbl)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, v: str, tooltip: str | None = None) -> None:
        self.value_lbl.setText(v)
        self.value_lbl.setToolTip(tooltip or '')


class StatRow(QWidget):
    """Compact key/value row for the tensor stats panel."""

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        k = label(key, 'Subtitle')
        self.v = label('—', selectable=True)
        self.v.setFont(theme.mono_font(10))
        self.v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(k)
        lay.addStretch(1)
        lay.addWidget(self.v)

    def set(self, text: str) -> None:
        self.v.setText(text)


class Histogram(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hist: list[int] = []
        self._range: tuple[float, float] = (0.0, 1.0)
        self._log = False
        self._hover: int = -1
        self.setMinimumHeight(150)
        self.setMouseTracking(True)

    def set_data(self, hist: list[int] | None, rng: tuple[float, float] | None) -> None:
        self._hist = list(hist or [])
        self._range = rng or (0.0, 1.0)
        self._hover = -1
        self.update()

    def set_log(self, on: bool) -> None:
        self._log = on
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(8, 8, self.width() - 16, self.height() - 34)

    def mouseMoveEvent(self, e) -> None:  # noqa: ANN001
        r = self._plot_rect()
        if self._hist and r.contains(e.position()):
            i = int((e.position().x() - r.left()) / r.width() * len(self._hist))
            self._hover = max(0, min(len(self._hist) - 1, i))
        else:
            self._hover = -1
        self.update()

    def leaveEvent(self, e) -> None:  # noqa: ANN001
        self._hover = -1
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._plot_rect()
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        if not self._hist:
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'computing…')
            return
        vals = [math.log1p(v) if self._log else float(v) for v in self._hist]
        top = max(vals) or 1.0
        n = len(vals)
        bw = r.width() / n
        lo, hi = self._range

        # Zero marker
        if lo < 0 < hi:
            zx = r.left() + (0 - lo) / (hi - lo) * r.width()
            p.setPen(QPen(QColor(theme.TEXT_FAINT), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(zx, r.top()), QPointF(zx, r.bottom()))

        path = QPainterPath()
        path.moveTo(r.left(), r.bottom())
        for i, v in enumerate(vals):
            h = v / top * r.height()
            path.lineTo(r.left() + i * bw, r.bottom() - h)
            path.lineTo(r.left() + (i + 1) * bw, r.bottom() - h)
        path.lineTo(r.right(), r.bottom())
        path.closeSubpath()
        fill = QColor(theme.ACCENT)
        fill.setAlpha(46)
        p.fillPath(path, fill)
        p.setPen(QPen(QColor(theme.ACCENT), 1.2))
        p.drawPath(path)

        if 0 <= self._hover < n:
            hx = r.left() + self._hover * bw
            hl = QColor(theme.INK)
            hl.setAlpha(18)
            p.fillRect(QRectF(hx, r.top(), bw, r.height()), hl)
            b0 = lo + (hi - lo) * self._hover / n
            b1 = lo + (hi - lo) * (self._hover + 1) / n
            txt = f'[{b0:.4g}, {b1:.4g})  {self._hist[self._hover]:,}'
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(r.left(), r.top(), r.width(), 16), Qt.AlignmentFlag.AlignRight, txt)

        p.setPen(QColor(theme.TEXT_DIM))
        p.setFont(theme.mono_font(8))
        lab = QRectF(r.left(), r.bottom() + 4, r.width(), 18)
        p.drawText(lab, Qt.AlignmentFlag.AlignLeft, f'{lo:.4g}')
        p.drawText(lab, Qt.AlignmentFlag.AlignRight, f'{hi:.4g}')
        if lo < 0 < hi:
            p.drawText(lab, Qt.AlignmentFlag.AlignHCenter, '0' if abs(lo + hi) < (hi - lo) * 0.1 else '')


class DTypeBar(QWidget):
    """Horizontal stacked bar showing byte share per dtype."""

    COLORS = theme.CATEGORICAL

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._parts: list[tuple[str, int]] = []
        self.setFixedHeight(10)

    def set_parts(self, parts: list[tuple[str, int]]) -> None:
        self._parts = parts
        self.update()

    def color(self, i: int) -> QColor:
        return QColor(self.COLORS[i % len(self.COLORS)])

    def paintEvent(self, _e) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        total = sum(v for _, v in self._parts) or 1
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 5, 5)
        p.setClipPath(clip)
        p.fillRect(self.rect(), QColor(theme.SURFACE_3))
        x = 0.0
        for i, (_, v) in enumerate(self._parts):
            w = v / total * self.width()
            p.fillRect(QRectF(x, 0, w, self.height()), self.color(i))
            x += w
