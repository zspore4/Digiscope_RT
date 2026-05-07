"""
filters/signal_processing.py  –  Basic signal-processing steps.

Covers:
  • RemoveMean      (ECGremoveMean.m)
  • AddNoise        (ECGnoise.m)
  • FullWaveRect    (DSFullWave.m)
  • Squaring        (ECGsquare.m)
  • DerivativeDetect(DSderivDetect.m)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox, QDialogButtonBox
)
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


# ---------------------------------------------------------------------------
class RemoveMeanFilter(BaseFilter):
    name = "Remove Mean"
    passthrough = True
    num_plots = 1

    def configure(self, parent=None) -> bool:
        return False  # no settings

    def calculate(self, signal: Signal) -> FilterResult:
        d = signal.data - np.mean(signal.data)
        t = np.arange(len(d)) / signal.rate
        return FilterResult(data=d, t=t, rate=signal.rate,
                            output_text="Remove Mean: done", passthrough=True)

    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8)
        ax.set_title("Remove Mean", fontsize=9)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
class AddNoiseFilter(BaseFilter):
    name = "Add Noise"
    passthrough = True
    num_plots = 1

    def __init__(self):
        super().__init__()
        self.amplitude: float = 0.1   # mV
        self.mode: int = 3            # 1=60Hz, 2=50Hz, 3=random

    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("Add Noise Settings")
        layout = QFormLayout(dlg)

        amp_edit = QLineEdit(str(self.amplitude * 1000))
        layout.addRow("Noise Amplitude (mV)", amp_edit)

        mode_combo = QComboBox()
        mode_combo.addItems(["60 Hz", "50 Hz", "Random (Gaussian)"])
        mode_combo.setCurrentIndex(self.mode - 1)
        layout.addRow("Noise Type", mode_combo)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)

        if dlg.exec_() == QDialog.Accepted:
            try:
                self.amplitude = float(amp_edit.text()) / 1000
                self.mode = mode_combo.currentIndex() + 1
            except ValueError:
                pass
            return True
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        t = np.arange(len(signal.data)) / signal.rate
        if self.mode == 1:
            noise = self.amplitude * np.sin(2 * np.pi * t * 60)
        elif self.mode == 2:
            noise = self.amplitude * np.sin(2 * np.pi * t * 50)
        else:
            noise = self.amplitude * np.random.randn(len(signal.data))

        d = signal.data + noise
        mode_name = {1: "60Hz", 2: "50Hz", 3: "Random"}[self.mode]
        return FilterResult(data=d, t=t, rate=signal.rate,
                            output_text=f"Add Noise: {mode_name}, amp={self.amplitude*1000:.1f} mV",
                            passthrough=True)

    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8)
        ax.set_title("Noisy ECG", fontsize=9)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
class FullWaveRectFilter(BaseFilter):
    name = "Full-Wave Rectifier"
    passthrough = True
    num_plots = 1

    def configure(self, parent=None) -> bool:
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        d = np.abs(signal.data)
        t = np.arange(len(d)) / signal.rate
        return FilterResult(data=d, t=t, rate=signal.rate,
                            output_text="Full-Wave Rectifier: done", passthrough=True)

    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8)
        ax.set_title("Full-Wave Rectified", fontsize=9)
        ax.set_ylabel("|Amplitude|")
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
class SquaringFilter(BaseFilter):
    name = "Squaring"
    passthrough = True
    num_plots = 1

    def configure(self, parent=None) -> bool:
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        d = signal.data ** 2
        t = np.arange(len(d)) / signal.rate
        return FilterResult(data=d, t=t, rate=signal.rate,
                            output_text="Squaring: done", passthrough=True)

    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8)
        ax.set_title("Squared Signal", fontsize=9)
        ax.set_ylabel("Amplitude²")
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
class DerivativeDetectFilter(BaseFilter):
    """
    Combines 1st and 2nd derivative energy to enhance QRS edges.
    From DSderivDetect.m: d1 + d2 weighted sum.
    """
    name = "Derivative Detect"
    passthrough = True
    num_plots = 1

    def configure(self, parent=None) -> bool:
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        x = signal.data
        b_ma = np.ones(5) / 5          # 5-point moving average

        # first difference: [1 0 -1]
        d1_raw = lfilter([1, 0, -1], [1], x)
        d1 = lfilter(b_ma, [1], np.abs(d1_raw))

        # second difference: [1 0 -2 0 1]
        d2_raw = lfilter([1, 0, -2, 0, 1], [1], x)
        d2 = lfilter(b_ma, [1], np.abs(d2_raw))

        d = 1.3 * d1 + 1.1 * d2
        t = np.arange(len(d)) / signal.rate
        return FilterResult(data=d, t=t, rate=signal.rate,
                            output_text="Deriv Detect: done", passthrough=True)

    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8)
        ax.set_title("Derivative Detect", fontsize=9)
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.3)
