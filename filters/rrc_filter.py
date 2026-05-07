"""
filters/rrc_filter.py  –  Root Raised Cosine (RRC) pulse-shaping filter.

Used for baseband communications:  apply to a bitstream to shape each symbol
so that the matched-filter pair (two RRC filters in cascade) gives zero ISI.

Parameters
----------
alpha    : roll-off factor  0 ≤ α ≤ 1   (0 = brick-wall, 1 = maximum roll-off)
T_sym    : symbol period in seconds  (1/symbol_rate)
N_taps   : filter length in samples (should be odd; typically 6*T_sym*fs + 1)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox,
    QLineEdit, QLabel, QComboBox
)
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


def _rrc_taps(alpha: float, T_sym: float, fs: float, N: int) -> np.ndarray:
    """
    Generate RRC filter coefficients.

    Parameters
    ----------
    alpha  : roll-off (0–1)
    T_sym  : symbol period (s)
    fs     : sample rate (Hz)
    N      : number of taps (odd recommended)

    Returns
    -------
    h : np.ndarray of length N, normalised so sum(h^2) = 1
    """
    Ts = T_sym        # symbol period
    if N % 2 == 0:
        N += 1        # force odd
    half = (N - 1) // 2
    t = (np.arange(N) - half) / fs   # time vector centred on 0

    h = np.zeros(N)
    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = (1 / Ts) * (1 + alpha * (4 / np.pi - 1))
        elif abs(ti) == Ts / (4 * alpha) and alpha != 0:
            h[i] = (alpha / (Ts * np.sqrt(2))) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            num = (np.sin(np.pi * ti / Ts * (1 - alpha))
                   + 4 * alpha * ti / Ts * np.cos(np.pi * ti / Ts * (1 + alpha)))
            den = (np.pi * ti / Ts * (1 - (4 * alpha * ti / Ts) ** 2))
            h[i] = num / (den * Ts) if den != 0 else 0.0

    # Normalise to unit energy
    energy = np.sqrt(np.sum(h ** 2))
    return h / energy if energy > 0 else h


class RRCFilter(BaseFilter):
    """Root Raised Cosine pulse-shaping filter."""

    name = "RRC Filter"
    passthrough = True
    num_plots = 1

    def __init__(self):
        super().__init__()
        self.alpha:   float = 0.35    # roll-off factor
        self.T_sym:   float = 1.0 / 1000.0   # symbol period (s) — default 1000 sym/s
        self.use_freq: bool = True    # True=enter symbol rate, False=enter period
        self.N_taps:  int   = 0       # 0 = auto (6 * sps)
        self._taps: np.ndarray = np.array([1.0])

    # ------------------------------------------------------------------
    def configure(self, parent=None) -> bool:
        dlg = _RRCDialog(self, parent)
        if dlg.exec_() == QDialog.Accepted:
            self.alpha    = dlg.alpha()
            self.T_sym    = dlg.T_sym()
            self.use_freq = dlg.use_freq()
            self.N_taps   = dlg.N_taps()
            sym_rate = 1.0 / self.T_sym
            self.name = (f"RRC  α={self.alpha:.3g}  "
                         f"Rs={sym_rate:.4g} sym/s")
            return True
        return False

    # ------------------------------------------------------------------
    def calculate(self, signal: Signal) -> FilterResult:
        fs   = signal.rate
        sps  = fs * self.T_sym          # samples per symbol
        N    = self.N_taps if self.N_taps > 0 else int(6 * sps) * 2 + 1
        N    = max(N, 3)

        self._taps = _rrc_taps(self.alpha, self.T_sym, fs, N)
        delay = (len(self._taps) - 1) // 2

        # Causal filter then compensate delay by trimming
        filtered = lfilter(self._taps, [1.0], signal.data)
        # Remove group delay
        if delay < len(filtered):
            out = np.concatenate([filtered[delay:],
                                  np.zeros(min(delay, len(filtered)))])
        else:
            out = filtered

        t = np.arange(len(out)) / fs
        sym_rate = 1.0 / self.T_sym
        return FilterResult(
            data=out, t=t, rate=fs,
            output_text=(f"RRC: α={self.alpha:.3g}  "
                         f"Rs={sym_rate:.4g} sym/s  "
                         f"sps={sps:.2f}  taps={len(self._taps)}"),
            passthrough=True,
            extras={"taps": self._taps},
        )

    # ------------------------------------------------------------------
    def plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear()
        try:
            from ui.app import _style_ax
            _style_ax(ax)
        except ImportError:
            ax.grid(True, alpha=0.3)
        ax.plot(result.t, result.data, color=colour, linewidth=0.8)
        ax.set_title(self.name, fontsize=9)
        ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")


class _RRCDialog(QDialog):
    def __init__(self, filt: RRCFilter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Root Raised Cosine Filter Settings")
        layout = QFormLayout(self)

        self._alpha = QLineEdit(str(filt.alpha))
        layout.addRow("Roll-off factor α  (0–1):", self._alpha)

        self._mode = QComboBox()
        self._mode.addItems(["Symbol Rate (sym/s)", "Symbol Period T (s)"])
        self._mode.setCurrentIndex(0 if filt.use_freq else 1)
        layout.addRow("Specify by:", self._mode)

        sym_rate = 1.0 / filt.T_sym
        self._freq_val = QLineEdit(f"{sym_rate:.6g}")
        self._per_val  = QLineEdit(f"{filt.T_sym:.6g}")
        self._freq_lbl = QLabel("Symbol Rate (sym/s):")
        self._per_lbl  = QLabel("Symbol Period T (s):")
        layout.addRow(self._freq_lbl, self._freq_val)
        layout.addRow(self._per_lbl,  self._per_val)

        self._ntaps = QLineEdit("0" if filt.N_taps == 0 else str(filt.N_taps))
        layout.addRow("# Taps (0=auto 6×sps):", self._ntaps)

        note = QLabel(
            "Cascade two RRC filters (using ⎘ Copy) to get a Raised Cosine filter pair.\n"
            "α=0 → brick-wall, α=1 → maximum roll-off. Typical: α=0.35 or 0.5.\n"
            "Symbol Rate = 1 / Symbol Period."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#4a5270; font-size:10px;")
        layout.addRow(note)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def alpha(self) -> float:
        try:
            v = float(self._alpha.text())
            return max(0.0, min(1.0, v))
        except ValueError:
            return 0.35

    def use_freq(self) -> bool:
        return self._mode.currentIndex() == 0

    def T_sym(self) -> float:
        try:
            if self._mode.currentIndex() == 0:   # symbol rate
                return 1.0 / float(self._freq_val.text())
            else:
                return float(self._per_val.text())
        except (ValueError, ZeroDivisionError):
            return 1.0 / 1000.0

    def N_taps(self) -> int:
        try:
            return max(0, int(self._ntaps.text()))
        except ValueError:
            return 0
