from __future__ import annotations

import math
import random
from typing import Any

from magnetron import Tensor
from magnetron._magnetron_bindings import dtype as mag_dtype
from PyQt6.QtCore import QObject, QPointF, QRectF, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QLinearGradient, QMouseEvent, QPainter, QPen, QPixmap, QPolygonF, QTransform, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import label

# Colormaps, as evenly spaced anchor colors

COLORMAPS: dict[str, tuple[str, ...]] = {
    'RdBu': ('#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b'),
    'Viridis': ('#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'),
    'Magma': ('#000004', '#1c1044', '#4f127b', '#812581', '#b5367a', '#e55064', '#fb8761', '#fec287', '#fcfdbf'),
    'Grayscale': ('#000000', '#ffffff'),
}
DIVERGING = {'RdBu'}
_LEVELS = 255  # Data maps to 0..254, index 255 marks NaN, Inf and padding
_MISSING = '#d9d7cf'
MAX_PIXELS = 1 << 25
MAX_POINTS = 20000
_GRAM_LIMIT = 2048


def _combo(items: list[str], tip: str) -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    c.setToolTip(tip)
    c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(6)
    return c


def ramp(name: str, n: int) -> list[QColor]:
    stops = [QColor(c) for c in COLORMAPS[name]]
    out: list[QColor] = []
    for i in range(n):
        t = i / max(1, n - 1) * (len(stops) - 1)
        k = min(int(t), len(stops) - 2)
        f = t - k
        a, b = stops[k], stops[k + 1]
        out.append(
            QColor(
                round(a.red() + (b.red() - a.red()) * f), round(a.green() + (b.green() - a.green()) * f), round(a.blue() + (b.blue() - a.blue()) * f)
            )
        )
    return out


def color_table(name: str) -> list[int]:
    return [c.rgba() for c in ramp(name, _LEVELS)] + [QColor(_MISSING).rgba()]


def _finite(f: Tensor) -> Tensor:
    return (f == f) & (f.abs() != math.inf)


def _count(mask: Tensor) -> int:
    return int(mask.cast(mag_dtype.int64).sum().item())


def finite_range(f: Tensor) -> tuple[float, float, bool]:
    """Min and max over finite values, plus whether everything was finite."""
    fin = _finite(f)
    nfin = _count(fin)
    if nfin == 0:
        return 0.0, 0.0, False
    if nfin == f.numel:
        return f.amin().item(), f.amax().item(), True
    return Tensor.where(fin, f, math.inf).amin().item(), Tensor.where(fin, f, -math.inf).amax().item(), False


# Image rendering


class ImageRender:
    """A 2D tensor slice mapped through a colormap into an indexed QImage, built with magnetron ops."""

    def __init__(self, view: Tensor, cmap: str, transform: str, symmetric: bool, clip_sigma: float = 0.0) -> None:
        rows, cols = view.shape
        self.rows, self.cols = rows, cols
        self.wrap = cols == 1 and rows > 1  # 1D data wraps into a near square image
        self.step = max(1, math.ceil(math.sqrt(view.numel / MAX_PIXELS)))
        v = view
        if self.step > 1 and not self.wrap:
            v = v.view_slice(0, 0, math.ceil(rows / self.step), self.step).view_slice(1, 0, math.ceil(cols / self.step), self.step)
        elif self.step > 1:
            v = v.view_slice(0, 0, math.ceil(rows / self.step), self.step)
        f = v.contiguous().cast(mag_dtype.float32)
        if transform == '|x|':
            f = f.abs()
        elif transform == 'log10 |x|':
            f = f.abs().log() * (1.0 / math.log(10.0))
        lo, hi, all_finite = finite_range(f)
        if clip_sigma > 0.0 and hi > lo:
            # Clip to mean +- k sigma so a few outliers do not wash out the rest of the image
            fin_f = f if all_finite else Tensor.where(_finite(f), f, 0.0)
            n = f.numel if all_finite else max(1, _count(_finite(f)))
            mean = fin_f.sum().item() / n
            d = fin_f - mean if all_finite else Tensor.where(_finite(f), f - mean, 0.0)
            sigma = math.sqrt((d * d).sum().item() / n)
            lo, hi = max(lo, mean - clip_sigma * sigma), min(hi, mean + clip_sigma * sigma)
        if symmetric and transform == 'x':
            a = max(abs(lo), abs(hi))
            lo, hi = -a, a
        if hi <= lo:
            hi = lo + 1.0
        self.lo, self.hi = lo, hi
        idx = ((f - lo) * ((_LEVELS - 1) / (hi - lo))).clamp(0, _LEVELS - 1).round()
        idx = Tensor.where(_finite(f), idx, float(_LEVELS)).cast(mag_dtype.uint8)
        data = idx.tobytes()
        if self.wrap:
            n = len(data)
            self.width = max(1, math.ceil(math.sqrt(n)))
            self.height = math.ceil(n / self.width)
            data += bytes([_LEVELS]) * (self.width * self.height - n)
        else:
            self.height, self.width = f.shape
        img = QImage(data, self.width, self.height, self.width, QImage.Format.Format_Indexed8).copy()
        img.setColorTable(color_table(cmap))
        self.image = img

    def cell_at(self, x: int, y: int) -> tuple[int, int] | None:
        """Map an image pixel back to (row, col) of the 2D slice."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        if self.wrap:
            i = (y * self.width + x) * self.step
            return (i, 0) if i < self.rows else None
        return y * self.step, x * self.step


class ImageView(QGraphicsView):
    hovered = pyqtSignal(int, int)
    left = pyqtSignal()
    activated = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setBackgroundBrush(QColor(theme.SURFACE_2))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setMouseTracking(True)
        self.fill = True

    def set_image(self, img: QImage, keep_view: bool = False) -> None:
        self.item.setPixmap(QPixmap.fromImage(img))
        self.scene().setSceneRect(QRectF(self.item.pixmap().rect()))
        self._apply_aspect()
        if not keep_view:
            self.fit()

    def _apply_aspect(self) -> None:
        pm = self.item.pixmap()
        if pm.isNull():
            return
        sy = 1.0
        if self.fill and pm.height() and self.viewport().width():
            # Stretch rows so very tall or very wide tensors fill the view instead of becoming a sliver
            target = pm.width() * self.viewport().height() / self.viewport().width()
            sy = target / pm.height()
        self.item.setTransform(QTransform.fromScale(1.0, sy))
        self.scene().setSceneRect(self.item.sceneBoundingRect())

    def fit(self) -> None:
        self.resetTransform()
        self.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_smoothing()

    def one_to_one(self) -> None:
        self.resetTransform()
        self.item.setTransform(QTransform())
        self.scene().setSceneRect(self.item.sceneBoundingRect())
        self._update_smoothing()

    def _update_smoothing(self) -> None:
        # Average when shrinking so dense tensors do not alias, keep crisp pixels when magnifying
        t = self.item.sceneTransform() * self.transform()
        shrinking = min(abs(t.m11()), abs(t.m22())) < 1.0
        mode = Qt.TransformationMode.SmoothTransformation if shrinking else Qt.TransformationMode.FastTransformation
        self.item.setTransformationMode(mode)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, shrinking)

    def set_fill(self, on: bool) -> None:
        self.fill = on
        self._apply_aspect()
        self.fit()

    def resizeEvent(self, e: Any) -> None:
        super().resizeEvent(e)
        if self.fill:
            self._apply_aspect()
            self.fit()

    def wheelEvent(self, e: QWheelEvent) -> None:
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._update_smoothing()

    def _pixel(self, e: QMouseEvent) -> tuple[int, int]:
        p = self.item.mapFromScene(self.mapToScene(e.position().toPoint()))
        return math.floor(p.x()), math.floor(p.y())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        super().mouseMoveEvent(e)
        self.hovered.emit(*self._pixel(e))

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        self.activated.emit(*self._pixel(e))

    def leaveEvent(self, e: Any) -> None:
        super().leaveEvent(e)
        self.left.emit()


class ColorBar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._cmap = 'RdBu'
        self._lo, self._hi = 0.0, 1.0
        self._caption = ''
        self.setFixedHeight(40)

    def set_state(self, cmap: str, lo: float, hi: float, caption: str) -> None:
        self._cmap, self._lo, self._hi, self._caption = cmap, lo, hi, caption
        self.update()

    def paintEvent(self, _e: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = QRectF(10, 6, self.width() - 20, 10)
        grad = QLinearGradient(bar.topLeft(), bar.topRight())
        cols = ramp(self._cmap, 32)
        for i, c in enumerate(cols):
            grad.setColorAt(i / (len(cols) - 1), c)
        p.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
        p.setBrush(grad)
        p.drawRoundedRect(bar, 3, 3)
        p.setFont(theme.mono_font(8))
        p.setPen(QColor(theme.TEXT_DIM))
        txt = QRectF(bar.left(), bar.bottom() + 3, bar.width(), 16)
        p.drawText(txt, Qt.AlignmentFlag.AlignLeft, f'{self._lo:.4g}')
        p.drawText(txt, Qt.AlignmentFlag.AlignRight, f'{self._hi:.4g}')
        p.drawText(txt, Qt.AlignmentFlag.AlignHCenter, self._caption)


class ImagePanel(QWidget):
    status = pyqtSignal(str)
    activated = pyqtSignal(int, int)

    def __init__(self, value_text: Any) -> None:
        super().__init__()
        self._value_text = value_text  # Callable (row, col) -> str supplied by the grid model
        self._view: Tensor | None = None
        self._render: ImageRender | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.cmap = _combo(list(COLORMAPS), 'Colormap')
        self.transform = _combo(['x', '|x|', 'log10 |x|'], 'Value transform before coloring')
        self.clip = _combo(['±3σ', '±2σ', 'Full'], 'Color range: clip to mean ± kσ so outliers do not wash out the image, or use the full min..max')
        for w in (self.cmap, self.transform, self.clip):
            bar.addWidget(w)
        self.symmetric = QCheckBox('Center 0')
        self.symmetric.setChecked(True)
        bar.addWidget(self.symmetric)
        self.fill = QCheckBox('Stretch')
        self.fill.setChecked(True)
        self.fill.setToolTip('Stretch rows so tall or wide tensors fill the view')
        bar.addWidget(self.fill)
        bar.addStretch(1)
        self.info = label('', 'Hint')
        self.info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        fit_btn, one_btn = QPushButton('Fit'), QPushButton('1:1')
        fit_btn.clicked.connect(lambda: self.view.fit())
        one_btn.clicked.connect(lambda: self.view.one_to_one())
        bar.addWidget(fit_btn)
        bar.addWidget(one_btn)
        lay.addLayout(bar)
        self.view = ImageView()
        lay.addWidget(self.view, 1)
        self.colorbar = ColorBar()
        lay.addWidget(self.colorbar)
        lay.addWidget(self.info)

        self.cmap.currentTextChanged.connect(self._on_cmap)
        self.transform.currentTextChanged.connect(lambda _: self.rerender(keep_view=True))
        self.symmetric.toggled.connect(lambda _: self.rerender(keep_view=True))
        self.clip.currentTextChanged.connect(lambda _: self.rerender(keep_view=True))
        self.fill.toggled.connect(self.view.set_fill)
        self.view.hovered.connect(self._on_hover)
        self.view.activated.connect(self._on_activate)

    def set_view(self, view: Tensor | None) -> None:
        self._view = view
        if view is not None and view.rank == 2 and view.numel:
            r, c = view.shape
            # Square pixels unless the slice is so lopsided it would render as a sliver
            self.fill.blockSignals(True)
            self.fill.setChecked(c > 1 and max(r / c, c / r) > 4)
            self.fill.blockSignals(False)
            self.view.fill = self.fill.isChecked()
        self.rerender()

    def clear(self) -> None:
        self._view = None
        self._render = None
        self.view.item.setPixmap(QPixmap())

    def _on_cmap(self, name: str) -> None:
        self.symmetric.blockSignals(True)
        self.symmetric.setChecked(name in DIVERGING)
        self.symmetric.blockSignals(False)
        self.rerender(keep_view=True)

    def rerender(self, keep_view: bool = False) -> None:
        if self._view is None or self._view.numel == 0:
            self.clear()
            self.info.setText('nothing to render')
            return
        sigma = {'±3σ': 3.0, '±2σ': 2.0}.get(self.clip.currentText(), 0.0)
        r = ImageRender(self._view, self.cmap.currentText(), self.transform.currentText(), self.symmetric.isChecked(), sigma)
        self._render = r
        self.view.set_image(r.image, keep_view)
        note = f'{r.width:,} × {r.height:,} px'
        if r.wrap:
            note += ' · 1D wrapped row-major'
        if r.step > 1:
            note += f' · 1 in {r.step} sampled'
        self.info.setText(note)
        self.colorbar.set_state(self.cmap.currentText(), r.lo, r.hi, self.transform.currentText())

    def _on_hover(self, x: int, y: int) -> None:
        if self._render is None:
            return
        cell = self._render.cell_at(x, y)
        if cell is not None:
            self.status.emit(f'[{cell[0]}, {cell[1]}]   {self._value_text(*cell)}   · double-click to open in grid')

    def _on_activate(self, x: int, y: int) -> None:
        if self._render is not None and (cell := self._render.cell_at(x, y)) is not None:
            self.activated.emit(*cell)


# Point clouds


def point_orientations(rows: int, cols: int) -> list[str]:
    """Which axis of a 2D slice can serve as the point set: each point needs >= 2 coordinates and the cloud >= 3 points."""
    out = []
    if cols >= 2 and rows >= 3:
        out.append('rows')
    if rows >= 2 and cols >= 3:
        out.append('columns')
    return out


class PointSignals(QObject):
    done = pyqtSignal(int, dict)


class PointCloudJob(QRunnable):
    """Builds 2D/3D point coordinates from a slice: direct coordinates for 2 or 3 dims, PCA via power iteration otherwise."""

    def __init__(self, token: int, view: Tensor, orientation: str) -> None:
        super().__init__()
        self.token = token
        self.view = view
        self.orientation = orientation
        self.signals = PointSignals()
        self.cancelled = False

    def run(self) -> None:
        try:
            res = self._compute()
        except Exception as e:
            res = {'error': str(e)}
        if not self.cancelled:
            self.signals.done.emit(self.token, res)

    def _compute(self) -> dict[str, Any]:
        v = self.view.transpose(0, 1) if self.orientation == 'columns' else self.view
        n, d = v.shape
        limit = MAX_POINTS if d <= _GRAM_LIMIT else _GRAM_LIMIT
        step = max(1, math.ceil(n / limit))
        if step > 1:
            v = v.view_slice(0, 0, math.ceil(n / step), step)
        x = v.contiguous().cast(mag_dtype.float32)
        # Non-finite coordinates would poison every projection, zero them
        x = Tensor.where(_finite(x), x, 0.0)
        ids = list(range(0, n, step))
        if d <= 3:
            cols = (x - x.mean(0, keepdim=True)).tolist()
            pts = [(r[0], r[1], r[2] if d == 3 else 0.0) for r in cols]
            axes = ['x', 'y', 'z'] if d == 3 else ['x', 'y']
            return self._pack(pts, ids, axes, d, None, step, n)
        scores, explained = self._pca(x)
        pts = list(zip(scores[0], scores[1], scores[2], strict=True))
        return self._pack(pts, ids, ['PC1', 'PC2', 'PC3'], 3, explained, step, n)

    def _pca(self, x: Tensor) -> tuple[list[list[float]], list[float]]:
        n, d = x.shape
        xc = (x - x.mean(0, keepdim=True)).contiguous()
        gram = d > _GRAM_LIMIT
        m = xc.matmul(xc.transpose(0, 1).contiguous()) if gram else xc.transpose(0, 1).contiguous().matmul(xc)
        total = (xc * xc).sum().item() or 1.0
        rng = random.Random(0)
        size = m.shape[0]
        scores: list[list[float]] = []
        explained: list[float] = []
        for _ in range(3):
            vec = Tensor([[rng.gauss(0.0, 1.0)] for _ in range(size)])
            lam = 0.0
            for _ in range(120):
                if self.cancelled:
                    return [[0.0] * n] * 3, [0.0] * 3
                w = m.matmul(vec)
                nrm = (w * w).sum().sqrt().item()
                if nrm == 0.0:
                    break
                vec = w * (1.0 / nrm)
                lam = nrm
            lam = max(0.0, lam)
            explained.append(lam / total)
            flat = vec.flatten()
            s = (flat * math.sqrt(lam)).tolist() if gram else xc.matmul(vec).flatten().tolist()
            scores.append(s)
            m = m - flat.outer(flat) * lam  # Deflate so the next pass finds the next component
        return scores, explained

    def _pack(
        self, pts: list[tuple[float, float, float]], ids: list[int], axes: list[str], dims: int, explained: list[float] | None, step: int, n: int
    ) -> dict[str, Any]:
        # Scale by RMS so the bulk fills the cube, outliers may sit outside it instead of squashing everything
        rms = math.sqrt(math.fsum(a * a + b * b + c * c for a, b, c in pts) / max(1, len(pts) * dims))
        peak = max((max(abs(a), abs(b), abs(c)) for a, b, c in pts), default=1.0)
        scale = min(peak, 2.5 * rms) or peak or 1.0
        pts = [(a / scale, b / scale, c / scale) for a, b, c in pts]
        return {'pts': pts, 'ids': ids, 'axes': axes, 'dims': dims, 'explained': explained, 'step': step, 'total': n}


class PointCloudView(QWidget):
    hovered = pyqtSignal(int)
    clicked = pyqtSignal(int)

    BUCKETS = 24

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumHeight(200)
        self._pts: list[tuple[float, float, float]] = []
        self._buckets: list[int] = []
        self._axes: list[str] = []
        self._dims = 3
        self._proj: list[tuple[float, float]] = []
        self._hover = -1
        self._drag: QPointF | None = None
        self._moved = False
        self.point_size = 3.0
        self._colors = ramp('Viridis', self.BUCKETS)
        self.reset_view()
        self._spin = QTimer(self, interval=30)
        self._spin.timeout.connect(self._step_spin)

    def reset_view(self) -> None:
        self.yaw, self.pitch, self.zoom = (0.65, -0.45, 1.0) if getattr(self, '_dims', 3) == 3 else (0.0, 0.0, 1.0)
        self.pan = QPointF(0, 0)
        self.update()

    def set_points(self, pts: list[tuple[float, float, float]], axes: list[str], dims: int) -> None:
        self._pts = pts
        n = max(1, len(pts) - 1)
        self._buckets = [min(self.BUCKETS - 1, i * self.BUCKETS // (n + 1)) for i in range(len(pts))]
        self._axes, self._dims = axes, dims
        self._hover = -1
        self.point_size = 4.0 if len(pts) < 2000 else 2.5 if len(pts) < 10000 else 2.0
        self.reset_view()

    def set_spin(self, on: bool) -> None:
        if on and self._dims == 3:
            self._spin.start()
        else:
            self._spin.stop()

    def _step_spin(self) -> None:
        self.yaw += 0.012
        self.update()

    def _rotation(self) -> tuple[float, ...]:
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        # Yaw around y, then pitch around x
        return (cy, 0.0, sy, sy * sp, cp, -cy * sp, -sy * cp, sp, cy * cp)

    def _frame(self) -> tuple[float, float, float]:
        return self.width() / 2 + self.pan.x(), self.height() / 2 + self.pan.y(), min(self.width(), self.height()) * 0.38 * self.zoom

    def paintEvent(self, _e: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.SURFACE))
        if not self._pts:
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'computing…')
            return
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = self._rotation()
        cx, cy, s = self._frame()

        def project(x: float, y: float, z: float) -> tuple[float, float, float]:
            return cx + (r00 * x + r01 * y + r02 * z) * s, cy - (r10 * x + r11 * y + r12 * z) * s, r20 * x + r21 * y + r22 * z

        # Reference frame: a cube for 3D, a square with a cross for 2D
        p.setPen(QPen(QColor(theme.BORDER), 1))
        if self._dims == 3:
            corners = [project(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            for i in range(8):
                for bit in (1, 2, 4):
                    j = i | bit
                    if j != i:
                        p.drawLine(QPointF(corners[i][0], corners[i][1]), QPointF(corners[j][0], corners[j][1]))
        else:
            a, b = project(-1, -1, 0), project(1, 1, 0)
            p.drawRect(QRectF(QPointF(a[0], b[1]), QPointF(b[0], a[1])))
            for u, w in ((project(-1, 0, 0), project(1, 0, 0)), (project(0, -1, 0), project(0, 1, 0))):
                p.drawLine(QPointF(u[0], u[1]), QPointF(w[0], w[1]))

        polys = [QPolygonF() for _ in range(self.BUCKETS)]
        proj: list[tuple[float, float]] = []
        for (x, y, z), bk in zip(self._pts, self._buckets, strict=True):
            px, py, _ = project(x, y, z)
            proj.append((px, py))
            polys[bk].append(QPointF(px, py))
        self._proj = proj
        for poly, col in zip(polys, self._colors, strict=True):
            c = QColor(col)
            c.setAlpha(210)
            p.setPen(QPen(c, self.point_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawPoints(poly)

        if 0 <= self._hover < len(proj):
            hx, hy = proj[self._hover]
            p.setPen(QPen(QColor(theme.INK), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(hx, hy), self.point_size + 4, self.point_size + 4)

        self._draw_gizmo(p)

    def _draw_gizmo(self, p: QPainter) -> None:
        r00, r01, r02, r10, r11, r12, *_ = self._rotation()
        ox, oy, ln = 34.0, self.height() - 34.0, 22.0
        colors = ('#c0392b', '#27ae60', theme.ACCENT)
        p.setFont(theme.mono_font(8))
        for i, name in enumerate(self._axes[: self._dims]):
            e = [0.0, 0.0, 0.0]
            e[i] = 1.0
            dx = (r00 * e[0] + r01 * e[1] + r02 * e[2]) * ln
            dy = -(r10 * e[0] + r11 * e[1] + r12 * e[2]) * ln
            p.setPen(QPen(QColor(colors[i]), 2))
            p.drawLine(QPointF(ox, oy), QPointF(ox + dx, oy + dy))
            p.drawText(QPointF(ox + dx * 1.3 - 6, oy + dy * 1.3 + 4), name)

    def _nearest(self, pos: QPointF, radius: float = 8.0) -> int:
        best, best_d = -1, radius * radius
        x, y = pos.x(), pos.y()
        for i, (px, py) in enumerate(self._proj):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_d:
                best, best_d = i, d
        return best

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self._drag = e.position()
        self._moved = False

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._drag is not None:
            d = e.position() - self._drag
            self._drag = e.position()
            self._moved = self._moved or abs(d.x()) + abs(d.y()) > 2
            if e.buttons() & Qt.MouseButton.LeftButton and self._dims == 3:
                self.yaw += d.x() * 0.01
                self.pitch = max(-1.55, min(1.55, self.pitch - d.y() * 0.01))
            else:
                self.pan += d
            self.update()
            return
        h = self._nearest(e.position())
        if h != self._hover:
            self._hover = h
            self.update()
            if h >= 0:
                self.hovered.emit(h)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if not self._moved:
            h = self._nearest(e.position())
            if h >= 0:
                self.clicked.emit(h)
        self._drag = None

    def mouseDoubleClickEvent(self, _e: QMouseEvent) -> None:
        self.reset_view()

    def wheelEvent(self, e: QWheelEvent) -> None:
        self.zoom = max(0.1, min(50.0, self.zoom * (1.15 if e.angleDelta().y() > 0 else 1 / 1.15)))
        self.update()


class PointCloudPanel(QWidget):
    status = pyqtSignal(str)
    activated = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self._view: Tensor | None = None
        self._token = 0
        self._job: PointCloudJob | None = None
        self._res: dict[str, Any] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.orientation = _combo([], 'Which axis of the slice forms the points')
        self.orientation.setMinimumContentsLength(14)
        bar.addWidget(self.orientation)
        self.size = QSpinBox()
        self.size.setPrefix('size ')
        self.size.setToolTip('Point size')
        self.size.setRange(1, 12)
        self.size.setValue(3)
        bar.addWidget(self.size)
        self.spin = QCheckBox('Rotate')
        self.spin.setToolTip('Spin the cloud slowly')
        bar.addWidget(self.spin)
        bar.addStretch(1)
        self.info = label('', 'Hint')
        self.info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        reset = QPushButton('Reset')
        reset.setToolTip('Reset rotation, zoom and pan (or double-click the cloud)')
        bar.addWidget(reset)
        lay.addLayout(bar)
        self.cloud = PointCloudView()
        lay.addWidget(self.cloud, 1)
        self.cloud.setToolTip('Drag to rotate · right-drag to pan · scroll to zoom · click a point to open it in the grid · double-click to reset')
        lay.addWidget(self.info)

        reset.clicked.connect(self.cloud.reset_view)
        self.spin.toggled.connect(self.cloud.set_spin)
        self.size.valueChanged.connect(self._on_size)
        self.orientation.currentTextChanged.connect(lambda _: self._start())
        self.cloud.hovered.connect(self._on_hover)
        self.cloud.clicked.connect(self._on_click)

    def _orientation(self) -> str:
        return self.orientation.currentData() or ''

    def _on_size(self, v: int) -> None:
        self.cloud.point_size = float(v)
        self.cloud.update()

    def set_view(self, view: Tensor | None) -> None:
        self._view = view
        opts = point_orientations(*view.shape) if view is not None and view.rank == 2 else []
        self.orientation.blockSignals(True)
        self.orientation.clear()
        for o in opts:
            self.orientation.addItem(f'{o} as points', o)
        # Prefer the axis whose partner has 2 or 3 entries, those are literal coordinates
        if view is not None and 'columns' in opts and view.shape[0] in (2, 3) and view.shape[1] not in (2, 3):
            self.orientation.setCurrentIndex(opts.index('columns'))
        self.orientation.blockSignals(False)
        self._start()

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancelled = True
            self._job = None

    def _start(self) -> None:
        self.cancel()
        self._token += 1
        self.cloud.set_points([], [], 3)
        o = self._orientation()
        if self._view is None or not o:
            self.info.setText('needs a 2D slice with at least 3 points of 2+ coordinates')
            return
        self.info.setText('computing…')
        job = PointCloudJob(self._token, self._view, o)
        job.signals.done.connect(self._on_done)
        self._job = job
        QThreadPool.globalInstance().start(job)

    def _on_done(self, token: int, res: dict) -> None:
        if token != self._token:
            return
        self._job = None
        if 'error' in res:
            self.info.setText(f'failed: {res["error"]}')
            return
        self._res = res
        self.cloud.set_points(res['pts'], res['axes'], res['dims'])
        self.size.blockSignals(True)
        self.size.setValue(round(self.cloud.point_size))
        self.size.blockSignals(False)
        self.cloud.set_spin(self.spin.isChecked())
        n = len(res['pts'])
        note = f'{n:,} points'
        if res['step'] > 1:
            note += f' (1 in {res["step"]} of {res["total"]:,})'
        if res['explained']:
            ev = res['explained']
            note += f' · PCA, {sum(ev):.1%} of variance (' + ' / '.join(f'{e:.1%}' for e in ev) + ')'
        else:
            note += f' · raw {res["dims"]}D coordinates'
        note += ' · color = index, purple → yellow'
        self.info.setText(note)

    def _cell(self, i: int) -> tuple[int, int]:
        idx = self._res['ids'][i]
        return (idx, 0) if self._orientation() == 'rows' else (0, idx)

    def _on_hover(self, i: int) -> None:
        if not self._res:
            return
        x, y, z = self._res['pts'][i]
        which = 'row' if self._orientation() == 'rows' else 'column'
        coords = ', '.join(f'{c:.4g}' for c in (x, y, z)[: self._res['dims']])
        self.status.emit(f'{which} {self._res["ids"][i]}   ({coords}) normalized')

    def _on_click(self, i: int) -> None:
        if self._res:
            self.activated.emit(*self._cell(i))
