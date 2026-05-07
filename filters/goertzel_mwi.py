"""
filters/goertzel_mwi.py  –  Goertzel detector and Moving Window Integrator

Goertzel
--------
Computes the power (magnitude squared of DFT coefficient) at a single target
frequency using the Goertzel algorithm.  Equivalent to computing one bin of
the DFT but far cheaper when you only care about one frequency.

Use cases:
  • DTMF tone detection
  • 50/60 Hz mains interference measurement
  • Monitoring power at a specific harmonic

The output is the squared magnitude of the DFT coefficient at f_target,
computed in sliding windows of N samples, giving a time-series of power
at that frequency.

Moving Window Integrator (MWI)
-------------------------------
FIR averaging filter with N equal-weight taps and a normalising denominator
of N (unity gain).  Equivalent to a rectangular moving average.

  H(z) = (1/N) * (1 + z^-1 + z^-2 + ... + z^-(N-1))
  b = [1, 1, ..., 1]  (N ones)
  a = [N]

Used in Pan-Tompkins as the final smoothing stage after squaring to integrate
the squared derivative signal into a smooth envelope.

Reference: Pan J, Tompkins WJ (1985). A real-time QRS detection algorithm.
           IEEE Trans Biomed Eng 32(3):230-236.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox,
    QLineEdit, QLabel, QVBoxLayout, QGroupBox, QSpinBox,
    QDoubleSpinBox
)
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


# ── Goertzel Filter ───────────────────────────────────────────────────────────

class GoertzelFilter(BaseFilter):
    """
    Goertzel single-frequency power detector.

    Computes |X(k)|² at the target frequency in sliding windows of N samples.
    Output is a time-series of power values at the target frequency.
    """
    name = "Goertzel"
    passthrough = False   # output is power, not the same signal type
    num_plots   = 1

    def __init__(self):
        super().__init__()
        self.target_hz : float = 50.0    # target frequency to measure
        self.window_n  : int   = 205     # DFT window size (samples)
        self.hop_n     : int   = 0       # hop size (0 = use window_n / 4)

    def configure(self, parent=None) -> bool:
        dlg = _GoertzelDialog(self, parent)
        if dlg.exec_() == QDialog.Accepted:
            self.target_hz = dlg.target_hz()
            self.window_n  = dlg.window_n()
            self.hop_n     = dlg.hop_n()
            self.name = f"Goertzel  {self.target_hz:.1f} Hz  N={self.window_n}"
            return True
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        data = signal.data
        fs   = signal.rate
        N    = min(self.window_n, len(data))
        hop  = self.hop_n if self.hop_n > 0 else max(1, N // 4)
        if N < 2:
            return FilterResult(data=np.array([0.0]), t=np.array([0.0]),
                                rate=fs / hop, passthrough=False,
                                output_text="Goertzel: signal too short")

        # Goertzel coefficient
        k     = round(self.target_hz * N / fs)
        omega = 2 * np.pi * k / N
        coeff = 2 * np.cos(omega)

        # Sliding window
        n_frames = max(1, (len(data) - N) // hop + 1)
        power    = np.zeros(n_frames)
        t_out    = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop
            block = data[start:start + N]
            if len(block) < N:
                break
            # Goertzel recursion
            s0, s1, s2 = 0.0, 0.0, 0.0
            for x in block:
                s0 = x + coeff * s1 - s2
                s2 = s1; s1 = s0
            real = s1 - s2 * np.cos(omega)
            imag = s2 * np.sin(omega)
            power[i] = real * real + imag * imag
            t_out[i] = (start + N / 2) / fs

        out_rate = fs / hop
        actual_f = k * fs / N
        return FilterResult(
            data=power[:i+1], t=t_out[:i+1], rate=out_rate,
            passthrough=False,
            output_text=(f"Goertzel: target={self.target_hz:.1f} Hz  "
                         f"bin={k}  actual={actual_f:.2f} Hz  "
                         f"N={N}  hop={hop}"),
            extras={"target_hz": actual_f},
        )

    def plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear()
        try:
            from ui.app import _style_ax
            _style_ax(ax)
        except ImportError:
            ax.grid(True, alpha=0.3)
        ax.plot(result.t, result.data, color=colour, linewidth=0.9)
        ax.set_title(self.name, fontsize=9)
        ax.set_ylabel("Power  |X(k)|²"); ax.set_xlabel("Time (s)")


class _GoertzelDialog(QDialog):
    def __init__(self, filt: GoertzelFilter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Goertzel Filter Settings")
        layout = QVBoxLayout(self)

        grp = QGroupBox("Parameters")
        fl  = QFormLayout(grp)

        self._freq = QDoubleSpinBox()
        self._freq.setRange(0.1, 100000.0); self._freq.setDecimals(2)
        self._freq.setValue(filt.target_hz); self._freq.setSuffix(" Hz")
        fl.addRow("Target Frequency:", self._freq)

        self._N = QSpinBox()
        self._N.setRange(4, 65536); self._N.setValue(filt.window_n)
        fl.addRow("Window Size N (samples):", self._N)

        self._hop = QSpinBox()
        self._hop.setRange(0, 65536); self._hop.setValue(filt.hop_n)
        self._hop.setSpecialValueText("auto (N/4)")
        fl.addRow("Hop Size (0 = auto):", self._hop)

        note = QLabel(
            "Computes |DFT(k)|² at the nearest frequency bin to the target.\n"
            "Output is a time-series of power at that frequency.\n"
            "Actual frequency = round(f × N / fs) × fs / N"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#4a5270; font-size:10px;")
        fl.addRow(note)
        layout.addWidget(grp)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def target_hz(self) -> float: return self._freq.value()
    def window_n(self)  -> int:   return self._N.value()
    def hop_n(self)     -> int:   return self._hop.value()


# ── Moving Window Integrator ──────────────────────────────────────────────────

class MWIFilter(BaseFilter):
    """
    Moving Window Integrator — FIR rectangular averaging filter.

    H(z) = (1/N)(1 + z^-1 + ... + z^-(N-1))
    b = [1, 1, ..., 1]  (N ones)    a = [N]

    Used as the final stage of the Pan-Tompkins QRS detector to smooth
    the squared derivative signal into a broad QRS envelope.
    The window width N should correspond to the expected QRS duration
    — Pan & Tompkins suggest ~150 ms (30 samples at 200 Hz).
    """
    name = "MWI (N=30)"
    passthrough = True
    num_plots   = 1

    def __init__(self):
        super().__init__()
        self.N : int = 30      # number of averaging taps

    def configure(self, parent=None) -> bool:
        dlg = _MWIDialog(self, parent)
        if dlg.exec_() == QDialog.Accepted:
            self.N    = dlg.N()
            self.name = f"MWI (N={self.N})"
            return True
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        N = max(1, self.N)
        b = np.ones(N, dtype=float)
        a = np.array([float(N)])
        out = lfilter(b, a, signal.data)
        return FilterResult(
            data=out, t=signal.t.copy(), rate=signal.rate,
            passthrough=True,
            output_text=(f"MWI: N={N}  "
                         f"window={N/signal.rate*1000:.1f} ms  "
                         f"b=[{N}×1]  a=[{N}]"),
            extras={"N": N, "b": b.tolist(), "a": a.tolist()},
        )

    def plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear()
        try:
            from ui.app import _style_ax
            _style_ax(ax)
        except ImportError:
            ax.grid(True, alpha=0.3)
        ax.plot(result.t, result.data, color=colour, linewidth=0.9)
        ax.set_title(self.name, fontsize=9)
        ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")


class _MWIDialog(QDialog):
    def __init__(self, filt: MWIFilter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Moving Window Integrator")
        layout = QVBoxLayout(self)

        grp = QGroupBox("Parameters")
        fl  = QFormLayout(grp)

        self._n = QSpinBox()
        self._n.setRange(1, 10000); self._n.setValue(filt.N)
        fl.addRow("N (number of taps):", self._n)

        note = QLabel(
            "Computes a rectangular moving average of N samples.\n\n"
            "Transfer function:\n"
            "  H(z) = (1/N)(1 + z⁻¹ + z⁻² + … + z⁻⁽ᴺ⁻¹⁾)\n"
            "  b = [1, 1, …, 1]  (N ones)    a = [N]\n\n"
            "Pan-Tompkins guideline: N ≈ 150 ms × sample_rate\n"
            "  e.g. N=30 at 200 Hz,  N=54 at 360 Hz"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#4a5270; font-size:10px;")
        fl.addRow(note)
        layout.addWidget(grp)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def N(self) -> int:
        return self._n.value()
