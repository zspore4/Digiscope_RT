"""
filters/ecg_filter.py  –  Configurable IIR / FIR filter.

Replaces ECGFilter.m.  Uses scipy.signal for filter design instead of
MATLAB's Filter Designer GUI.  Supports:
  • Butterworth low-pass / high-pass / band-pass / band-stop
  • Notch (60 Hz or 50 Hz)
  • Custom coefficient entry
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos, lfilter
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox,
    QDialogButtonBox, QDoubleSpinBox, QSpinBox, QLabel
)
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


FILTER_TYPES = ["Lowpass", "Highpass", "Bandpass", "Bandstop", "Notch 60Hz", "Notch 50Hz"]


class ECGFilterFilter(BaseFilter):
    name = "ECG Filter"
    passthrough = True
    num_plots = 1

    def __init__(self):
        super().__init__()
        self.filter_type: int = 0     # index into FILTER_TYPES
        self.order: int = 4
        self.cutoff_low: float = 0.5  # Hz
        self.cutoff_high: float = 40.0  # Hz (for bp/bs)
        self.notch_q: float = 30.0

    # ------------------------------------------------------------------
    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("ECG Filter Settings")
        layout = QFormLayout(dlg)

        type_combo = QComboBox()
        type_combo.addItems(FILTER_TYPES)
        type_combo.setCurrentIndex(self.filter_type)
        layout.addRow("Filter Type", type_combo)

        order_spin = QSpinBox()
        order_spin.setRange(1, 10)
        order_spin.setValue(self.order)
        layout.addRow("Order", order_spin)

        low_edit  = QLineEdit(str(self.cutoff_low))
        high_edit = QLineEdit(str(self.cutoff_high))
        notch_q_edit = QLineEdit(str(self.notch_q))
        layout.addRow("Cutoff Low (Hz)", low_edit)
        layout.addRow("Cutoff High (Hz) [BP/BS only]", high_edit)
        layout.addRow("Notch Q factor", notch_q_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)

        if dlg.exec_() == QDialog.Accepted:
            try:
                self.filter_type  = type_combo.currentIndex()
                self.order        = order_spin.value()
                self.cutoff_low   = float(low_edit.text())
                self.cutoff_high  = float(high_edit.text())
                self.notch_q      = float(notch_q_edit.text())
            except ValueError:
                pass
            return True
        return False

    # ------------------------------------------------------------------
    def calculate(self, signal: Signal) -> FilterResult:
        fs  = signal.rate
        nyq = fs / 2.0
        x   = signal.data

        try:
            ft = FILTER_TYPES[self.filter_type]
            if ft == "Lowpass":
                sos = butter(self.order, self.cutoff_low / nyq, btype="low", output="sos")
            elif ft == "Highpass":
                sos = butter(self.order, self.cutoff_low / nyq, btype="high", output="sos")
            elif ft == "Bandpass":
                sos = butter(self.order,
                             [self.cutoff_low / nyq, self.cutoff_high / nyq],
                             btype="band", output="sos")
            elif ft == "Bandstop":
                sos = butter(self.order,
                             [self.cutoff_low / nyq, self.cutoff_high / nyq],
                             btype="bandstop", output="sos")
            elif ft == "Notch 60Hz":
                b, a = iirnotch(60.0 / nyq, self.notch_q)
                sos  = tf2sos(b, a)
            elif ft == "Notch 50Hz":
                b, a = iirnotch(50.0 / nyq, self.notch_q)
                sos  = tf2sos(b, a)
            else:
                sos = None

            d = sosfiltfilt(sos, x) if sos is not None else x.copy()
        except Exception as e:
            d = x.copy()
            return FilterResult(data=d, t=np.arange(len(d)) / fs, rate=fs,
                                output_text=f"Filter error: {e}", passthrough=True)

        t = np.arange(len(d)) / fs
        desc = f"ECG Filter: {FILTER_TYPES[self.filter_type]}, order={self.order}"
        return FilterResult(data=d, t=t, rate=fs, output_text=desc, passthrough=True)

    # ------------------------------------------------------------------
    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8)
        ax.set_title(f"ECG Filter  ({FILTER_TYPES[self.filter_type]})", fontsize=9)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
