"""
filters/ecg_game.py  –  ECG Flappy Bird Easter Egg

A filter that opens a mini-game instead of doing signal processing.
The filter is NOT added to the chain when the player loses (or closes).
It is only added (as a passthrough) if they achieve a score > 0 and
choose to keep playing — actually it never modifies the chain, it just
opens the game window.

Game mechanics
--------------
• A horizontal scrolling lane shows a synthetic ECG waveform "target".
• The player controls a glowing heart cursor.
  - Tap SPACE / click = heart goes UP  (mimics R-peak)
  - Release = heart slowly falls (mimics diastole)
• PQRST gates slide in from the right:
    P  gate  – gentle rise needed (narrow, mid-height)
    Q  gate  – slight dip
    R  gate  – must spike HIGH  (narrow, very high)
    S  gate  – must dip LOW  (brief)
    T  gate  – moderate hump, then settle back to baseline
• Missing a gate = miss.  3 misses = game over.
• Every ~4 beats the heart rate increases by 5 BPM.
• PVC events (random): an extra narrow spike gate appears between beats.
• Skipped-beat events (random): the expected gate sequence is delayed/absent
  and a "SKIPPED" warning flashes — player must keep heart near baseline.
• Successful gate count shown top-right; high score persisted to ~/.digiscope_hs
"""

from __future__ import annotations
import math, random, time, os, json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QWidget
)
from PyQt5.QtCore import Qt, QTimer, QRect, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QRadialGradient, QPainterPath, QFontMetrics
)

from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal

# ── High score file ────────────────────────────────────────────────────────
HS_FILE = os.path.join(os.path.expanduser("~"), ".digiscope_hs")

def _load_hs() -> int:
    try:
        with open(HS_FILE) as f:
            return int(json.load(f).get("ecg_game", 0))
    except Exception:
        return 0

def _save_hs(score: int):
    try:
        data = {}
        try:
            with open(HS_FILE) as f:
                data = json.load(f)
        except Exception:
            pass
        data["ecg_game"] = max(score, data.get("ecg_game", 0))
        with open(HS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

# ── Gate definition ────────────────────────────────────────────────────────
@dataclass
class Gate:
    """A PQRST gate that slides across the screen."""
    label:     str        # "P", "Q", "R", "S", "T", "PVC", "SKIP"
    x:         float      # current x position (pixels, 0=left edge)
    y_center:  float      # centre of the opening (0=top, 1=bottom of lane)
    gap:       float      # half-height of the opening (fraction of lane)
    width:     float = 18 # pixel width of the gate pillar
    passed:    bool = False
    missed:    bool = False
    color:     tuple = (80, 160, 255)

    @property
    def y_lo(self) -> float:
        return self.y_center - self.gap

    @property
    def y_hi(self) -> float:
        return self.y_center + self.gap

# ── PQRST gate templates (y_center, gap, colour) ──────────────────────────
# y values are fractions: 0 = top of lane, 1 = bottom
# baseline sits at y=0.65 (heart at rest is below mid)
GATE_TEMPLATES = {
    "P":   (0.45, 0.18, (100, 180, 100)),   # gentle rise
    "Q":   (0.72, 0.16, (200, 160,  60)),   # slight dip
    "R":   (0.15, 0.14, (220,  60,  60)),   # spike HIGH
    "S":   (0.78, 0.16, (200, 100, 200)),   # dip
    "T":   (0.40, 0.20, (60,  160, 220)),   # hump
    "PVC": (0.10, 0.13, (255,  80,  80)),   # early spike
    "SKIP":(0.65, 0.24, (140, 140, 140)),   # stay near baseline
}
BEAT_SEQUENCE = ["P", "Q", "R", "S", "T"]

# ── Colours ────────────────────────────────────────────────────────────────
BG_TOP    = QColor(10,  12,  30)
BG_BOT    = QColor(15,  20,  50)
GRID_COL  = QColor(25,  35,  70)
TAIL_COL  = QColor(80, 220, 180)
HEART_COL = QColor(255, 80,  120)
TEXT_COL  = QColor(200, 220, 255)
MISS_COL  = QColor(255, 60,  60)
HIT_COL   = QColor(80,  255, 160)

# ── Game widget ────────────────────────────────────────────────────────────
class GameWidget(QWidget):
    TICK_MS    = 16           # ~60 fps
    LANE_H     = 480          # px
    LANE_W     = 900          # px
    BASELINE   = 0.65         # heart rest position (fraction of lane height)
    # Spring-toward-baseline physics:
    #   Releasing = spring pulls heart back to BASELINE naturally
    #   Pressing  = upward acceleration fights the spring
    #   Result: heart always wants to sit at baseline; player holds to rise,
    #           releases and it floats back — controllable and predictable.
    K_SPRING   = 0.003        # spring constant toward BASELINE (per ms)
    ACCEL      = 0.00055      # upward acceleration per ms while pressing
    VEL_MAX    = 0.016        # terminal velocity cap
    DRAG       = 0.90         # velocity multiplier per tick
    SCROLL     = 160          # px/s
    MAX_MISSES = 3
    TAIL_LEN   = 180          # px

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.LANE_W, self.LANE_H)
        self.setFocusPolicy(Qt.StrongFocus)

        self._reset()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(self.TICK_MS)
        self._high_score = _load_hs()

    def _reset(self):
        self._running   = True
        self._heart_y   = self.BASELINE   # fraction
        self._vel       = 0.0
        self._pressing  = False
        self._score     = 0
        self._misses    = 0
        self._bpm       = 10
        self._beats_at_bpm = 0
        self._gates: list[Gate] = []
        self._tail: list[tuple[float,float]] = []  # (x, y_px)
        self._tail_x    = float(self.LANE_W // 3)  # heart x position
        self._last_time = time.perf_counter()
        self._next_gate_x = float(self.LANE_W)
        self._beat_phase  = 0              # which step in BEAT_SEQUENCE
        self._flash: Optional[tuple] = None  # (text, color, ttl_ms)
        self._pvc_pending = False
        self._skip_pending = False
        self._gate_spacing = self._bpm_to_spacing(self._bpm)
        self._dead = False
        self._dead_time = 0.0

    def _bpm_to_spacing(self, bpm: float) -> float:
        """Pixel distance between gates at current scroll speed."""
        beat_s = 60.0 / bpm
        # 5 gates per beat, each spaced evenly
        return (self.SCROLL * beat_s) / len(BEAT_SEQUENCE)

    # ── Game loop ──────────────────────────────────────────────────────────
    def _tick(self):
        now = time.perf_counter()
        dt_ms = (now - self._last_time) * 1000
        self._last_time = now

        if self._dead:
            self._dead_time += dt_ms
            self.update()
            return

        self._update_physics(dt_ms)
        self._update_gates(dt_ms)
        self._check_collisions()
        self._maybe_spawn_gate()
        self._increase_difficulty()
        self.update()

    def _update_physics(self, dt_ms: float):
        if self._pressing:
            # Upward acceleration while button held
            self._vel -= self.ACCEL * dt_ms
        else:
            # Spring force: always pulls heart back toward baseline.
            # This replaces gravity — the heart naturally rests at BASELINE
            # without any input, making the controls feel predictable.
            spring = -self.K_SPRING * (self._heart_y - self.BASELINE) * dt_ms
            # Tiny constant gravity only when above baseline (adds slight weight)
            gravity = 0.00006 * dt_ms if self._heart_y < self.BASELINE else 0.0
            self._vel += spring + gravity

        self._vel *= self.DRAG
        self._vel = max(-self.VEL_MAX, min(self.VEL_MAX, self._vel))
        self._heart_y = max(0.02, min(0.97, self._heart_y + self._vel))

        # Record tail
        hx = self._tail_x
        hy = self._heart_y * self.LANE_H
        self._tail.append((hx, hy))
        # Scroll tail leftward with gates
        dx = self.SCROLL * self.TICK_MS / 1000
        self._tail = [(x - dx, y) for x, y in self._tail
                      if x - dx > hx - self.TAIL_LEN]

    def _update_gates(self, dt_ms: float):
        dx = self.SCROLL * dt_ms / 1000
        self._next_gate_x -= dx
        for g in self._gates:
            g.x -= dx

        # Remove off-screen gates (but score misses first)
        remaining = []
        for g in self._gates:
            if g.x + g.width < 0:
                if not g.passed and not g.missed and g.label != "SKIP":
                    self._register_miss(g.label)
            else:
                remaining.append(g)
        self._gates = remaining

    def _maybe_spawn_gate(self):
        # Spawn next gate when _next_gate_x reaches the right side
        if self._next_gate_x > self.LANE_W:
            return

        # Decide what to spawn
        if self._pvc_pending:
            self._pvc_pending = False
            self._spawn("PVC")
            self._next_gate_x = self.LANE_W + self._gate_spacing * 0.6
            return

        if self._skip_pending:
            self._skip_pending = False
            self._spawn("SKIP")
            self._flash = ("SKIPPED BEAT!", QColor(255, 200, 50), 1200)
            self._next_gate_x = self.LANE_W + self._gate_spacing * 2.5
            self._beat_phase = 0
            return

        label = BEAT_SEQUENCE[self._beat_phase]
        self._spawn(label)
        self._beat_phase = (self._beat_phase + 1) % len(BEAT_SEQUENCE)

        # At end of a full beat, maybe schedule PVC / skip
        if self._beat_phase == 0:
            self._beats_at_bpm += 1
            roll = random.random()
            if roll < 0.08:
                self._pvc_pending = True
            elif roll < 0.13:
                self._skip_pending = True

        self._next_gate_x = self.LANE_W + self._gate_spacing

    def _spawn(self, label: str):
        yc, gap, col = GATE_TEMPLATES[label]
        # Add slight randomness to keep it interesting
        yc = yc + random.uniform(-0.05, 0.05)
        yc = max(gap + 0.05, min(1 - gap - 0.05, yc))
        self._gates.append(Gate(
            label=label, x=float(self.LANE_W),
            y_center=yc, gap=gap, color=col
        ))

    def _check_collisions(self):
        hx = self._tail_x
        hy = self._heart_y

        for g in self._gates:
            # Gate is at the heart's x column?
            if g.x <= hx <= g.x + g.width and not g.passed and not g.missed:
                if g.y_lo <= hy <= g.y_hi:
                    # PASS
                    g.passed = True
                    self._score += 1
                    if g.label == "R":
                        self._flash = (f"♥  {self._bpm:.0f} BPM", HIT_COL, 600)
                    else:
                        self._flash = (f"+1  {g.label}", HIT_COL, 400)
                else:
                    # MISS
                    g.missed = True
                    self._register_miss(g.label)

    def _register_miss(self, label: str):
        self._misses += 1
        self._flash = (f"MISS  {label}!", MISS_COL, 800)
        if self._misses >= self.MAX_MISSES:
            self._game_over()

    def _game_over(self):
        self._dead = True
        self._running = False
        if self._score > self._high_score:
            self._high_score = self._score
            _save_hs(self._score)

    def _increase_difficulty(self):
        # Every 15 successful gates raise BPM by 2, starting from 10
        threshold = 15
        if self._score > 0 and self._score % threshold == 0:
            new_bpm = 10 + (self._score // threshold) * 2
            if new_bpm != self._bpm and new_bpm <= 180:
                self._bpm = new_bpm
                self._gate_spacing = self._bpm_to_spacing(self._bpm)

    # ── Input ──────────────────────────────────────────────────────────────
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._pressing = True
            if self._dead:
                self._reset()

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._pressing = False

    def mousePressEvent(self, e):
        self._pressing = True
        if self._dead:
            self._reset()

    def mouseReleaseEvent(self, e):
        self._pressing = False

    # ── Rendering ──────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        self._draw_background(p)
        self._draw_grid(p)
        self._draw_ecg_target(p)
        self._draw_gates(p)
        self._draw_tail(p)
        self._draw_heart(p)
        self._draw_hud(p)

        if self._flash:
            text, col, _ = self._flash
            self._draw_flash(p, text, col)
            # Decrement TTL — handled via tick but we approximate here
            self._flash = (text, col, self._flash[2] - self.TICK_MS)
            if self._flash[2] <= 0:
                self._flash = None

        if self._dead:
            self._draw_game_over(p)

        p.end()

    def _draw_background(self, p: QPainter):
        grad = QLinearGradient(0, 0, 0, self.LANE_H)
        grad.setColorAt(0, BG_TOP)
        grad.setColorAt(1, BG_BOT)
        p.fillRect(0, 0, self.LANE_W, self.LANE_H, grad)

    def _draw_grid(self, p: QPainter):
        p.setPen(QPen(GRID_COL, 1))
        # Horizontal lines at 10% intervals
        for i in range(1, 10):
            y = int(self.LANE_H * i / 10)
            p.drawLine(0, y, self.LANE_W, y)
        # Baseline
        p.setPen(QPen(QColor(50, 80, 140), 1, Qt.DashLine))
        p.drawLine(0, int(self.BASELINE * self.LANE_H),
                   self.LANE_W, int(self.BASELINE * self.LANE_H))

    def _draw_ecg_target(self, p: QPainter):
        """Draw a faint ghost ECG target waveform scrolling in background."""
        # Generate one synthetic beat worth of points
        # We'll draw 3 beats worth stretched across the lane
        n = 400
        t = np.linspace(0, 3, n)
        wave = _make_ecg_wave(t, bpm=self._bpm)
        # Normalise to lane
        wave_n = 0.65 - wave * 0.28   # centre around baseline, scale

        path = QPainterPath()
        scroll_offset = (time.perf_counter() * self.SCROLL) % (self.LANE_W / 3)
        for i, (ti, yi) in enumerate(zip(t, wave_n)):
            px = (ti / 3) * self.LANE_W - scroll_offset
            px = px % self.LANE_W   # wrap
            py = yi * self.LANE_H
            if i == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)

        pen = QPen(QColor(40, 80, 120, 80), 1.5)
        p.setPen(pen)
        p.drawPath(path)

    def _draw_gates(self, p: QPainter):
        for g in self._gates:
            r, gv, b = g.color
            col = QColor(r, gv, b, 200 if not g.passed else 80)
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)

            x = int(g.x)
            w = int(g.width)
            y_lo_px = int(g.y_lo * self.LANE_H)
            y_hi_px = int(g.y_hi * self.LANE_H)

            # Top block (above opening)
            p.drawRect(x, 0, w, y_lo_px)
            # Bottom block (below opening)
            p.drawRect(x, y_hi_px, w, self.LANE_H - y_hi_px)

            # Gate label
            if not g.passed:
                font = QFont("Consolas", 9, QFont.Bold)
                p.setFont(font)
                p.setPen(QPen(QColor(255, 255, 255, 200)))
                p.drawText(x - 2, y_lo_px - 6, g.label)

                # Opening highlight
                pen = QPen(QColor(r, gv, b, 180), 2)
                p.setPen(pen)
                p.drawLine(x, y_lo_px, x + w, y_lo_px)
                p.drawLine(x, y_hi_px, x + w, y_hi_px)

    def _draw_tail(self, p: QPainter):
        """Draw the ECG-style trailing waveform behind the heart."""
        if len(self._tail) < 2:
            return
        path = QPainterPath()
        path.moveTo(self._tail[0][0], self._tail[0][1])
        for x, y in self._tail[1:]:
            path.lineTo(x, y)

        pen = QPen(TAIL_COL, 2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawPath(path)

    def _draw_heart(self, p: QPainter):
        hx = self._tail_x
        hy = self._heart_y * self.LANE_H

        # Glow
        glow = QRadialGradient(hx, hy, 20)
        glow.setColorAt(0, QColor(255, 80, 120, 120))
        glow.setColorAt(1, QColor(255, 80, 120, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(hx - 20), int(hy - 20), 40, 40)

        # Heart shape via path
        p.setBrush(QBrush(HEART_COL))
        path = QPainterPath()
        s = 10  # half-size
        path.moveTo(hx, hy + s)
        path.cubicTo(hx - s * 1.5, hy, hx - s * 2, hy - s * 1.2, hx, hy - s * 0.4)
        path.cubicTo(hx + s * 2, hy - s * 1.2, hx + s * 1.5, hy, hx, hy + s)
        p.drawPath(path)

    def _draw_hud(self, p: QPainter):
        font = QFont("Consolas", 11, QFont.Bold)
        p.setFont(font)

        # Misses (top-left)
        hearts = "♥" * (self.MAX_MISSES - self._misses) + "♡" * self._misses
        p.setPen(QPen(HEART_COL))
        p.drawText(12, 24, hearts)

        # BPM (top-centre)
        p.setPen(QPen(TEXT_COL))
        bpm_str = f"{self._bpm:.0f} BPM"
        fm = QFontMetrics(font)
        bw = fm.horizontalAdvance(bpm_str)
        p.drawText(self.LANE_W // 2 - bw // 2, 24, bpm_str)

        # Score + high score (top-right)
        score_str = f"Score: {self._score}   Best: {self._high_score}"
        sw = fm.horizontalAdvance(score_str)
        p.drawText(self.LANE_W - sw - 12, 24, score_str)

        # Instructions (bottom)
        if self._score == 0 and not self._dead:
            p.setPen(QPen(QColor(150, 170, 220, 180)))
            p.setFont(QFont("Consolas", 10))
            tip = "SPACE or CLICK to make the heart beat  ·  Hit each gate at the right height"
            tw = QFontMetrics(QFont("Consolas", 10)).horizontalAdvance(tip)
            p.drawText(self.LANE_W // 2 - tw // 2, self.LANE_H - 10, tip)

    def _draw_flash(self, p: QPainter, text: str, col: QColor):
        font = QFont("Consolas", 18, QFont.Bold)
        p.setFont(font)
        p.setPen(QPen(col))
        fm = QFontMetrics(font)
        w = fm.horizontalAdvance(text)
        p.drawText(self.LANE_W // 2 - w // 2, self.LANE_H // 2 - 20, text)

    def _draw_game_over(self, p: QPainter):
        # Semi-transparent overlay
        p.fillRect(0, 0, self.LANE_W, self.LANE_H, QColor(0, 0, 0, 160))

        font_big = QFont("Consolas", 28, QFont.Bold)
        font_sm  = QFont("Consolas", 13)
        p.setFont(font_big)
        p.setPen(QPen(MISS_COL))

        line1 = "FLATLINE"
        fm = QFontMetrics(font_big)
        p.drawText(self.LANE_W // 2 - fm.horizontalAdvance(line1) // 2,
                   self.LANE_H // 2 - 50, line1)

        new_hs = self._score >= self._high_score and self._score > 0
        line2 = f"Score: {self._score}" + ("  ★ NEW BEST!" if new_hs else
                                            f"   Best: {self._high_score}")
        p.setFont(font_sm)
        p.setPen(QPen(TEXT_COL))
        fm2 = QFontMetrics(font_sm)
        p.drawText(self.LANE_W // 2 - fm2.horizontalAdvance(line2) // 2,
                   self.LANE_H // 2 + 5, line2)

        line3 = "Click or SPACE to restart  ·  Close window to exit"
        p.setPen(QPen(QColor(150, 170, 210)))
        p.drawText(self.LANE_W // 2 - fm2.horizontalAdvance(line3) // 2,
                   self.LANE_H // 2 + 35, line3)

        # Flat ECG line
        p.setPen(QPen(QColor(80, 220, 120, 120), 2))
        mid = self.LANE_H // 2 + 80
        # draw with a tiny blip at centre
        cx = self.LANE_W // 2
        p.drawLine(0, mid, cx - 40, mid)
        p.drawLine(cx - 40, mid, cx - 20, mid - 25)
        p.drawLine(cx - 20, mid - 25, cx, mid + 15)
        p.drawLine(cx, mid + 15, cx + 10, mid)
        p.drawLine(cx + 10, mid, self.LANE_W, mid)


# ── Synthetic ECG helper ───────────────────────────────────────────────────
def _make_ecg_wave(t: np.ndarray, bpm: float = 70) -> np.ndarray:
    """Simple Gaussian-component ECG for the background ghost."""
    T = 60.0 / bpm
    wave = np.zeros_like(t)
    components = [
        # (offset_frac, sigma_frac, amp)
        (-0.30, 0.028, 0.15),   # P
        (-0.05, 0.008, -0.08),  # Q
        ( 0.00, 0.010, 1.00),   # R
        ( 0.05, 0.008, -0.15),  # S
        ( 0.20, 0.045, 0.25),   # T
    ]
    for beat_t in np.arange(0, t[-1], T):
        for off, sig, amp in components:
            centre = beat_t + off * T
            wave += amp * np.exp(-0.5 * ((t - centre) / (sig * T)) ** 2)
    return wave


# ── Dialog wrapper ─────────────────────────────────────────────────────────
class ECGGameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Runner")
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._game = GameWidget(self)
        layout.addWidget(self._game)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 4, 8, 4)
        lbl = QLabel("♥ Keep your heart in the gates!  SPACE/click to beat.")
        lbl.setStyleSheet("color:#9ab; font-size:11px;")
        bar.addWidget(lbl)
        bar.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        bar.addWidget(close_btn)
        layout.addLayout(bar)

        self.setFixedSize(self._game.LANE_W, self._game.LANE_H + 36)
        self._game.setFocus()


# ── BaseFilter shell (never modifies the chain) ───────────────────────────
class ECGGameFilter(BaseFilter):
    """
    Easter egg: opens the ECG Runner game.
    Returns the input signal unmodified so the chain is unaffected.
    configure() returns False so the filter is NOT added to the chain
    when the player closes/loses.
    """
    name = "♥ ECG Runner"
    passthrough = True
    num_plots = 1

    def configure(self, parent=None) -> bool:
        dlg = ECGGameDialog(parent)
        dlg.exec_()
        # Always return False — never add this filter to the chain
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        return FilterResult(
            data=signal.data.copy(), t=signal.t.copy(),
            rate=signal.rate, passthrough=True,
            output_text="ECG Runner: passthrough"
        )

    def plot(self, axes, result):
        colour = result.extras.get("_colour", "#1565C0")
        ax = axes[0]; ax.clear()
        ax.plot(result.t, result.data, color=colour, linewidth=0.9)
        ax.set_title("ECG Runner (passthrough)", fontsize=9)
        ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
