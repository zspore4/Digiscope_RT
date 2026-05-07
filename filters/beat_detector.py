"""
filters/beat_detector.py  –  Threshold-based QRS beat detector.

Translated from ECGbeatDetector.m.
"""

from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox,
    QDialogButtonBox, QLabel
)
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


class BeatDetectorFilter(BaseFilter):
    name = "Beat Detector"
    passthrough = True
    num_plots = 1
    scroll = True

    def __init__(self):
        super().__init__()
        self.threshold: float = 0.5
        self.edge: int = 0          # 0 = rising, 1 = falling
        self.min_gap: int = 15      # minimum samples between beats

    # ------------------------------------------------------------------
    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("Beat Detector Settings")
        layout = QFormLayout(dlg)

        thresh_edit = QLineEdit(str(self.threshold))
        layout.addRow("Threshold", thresh_edit)

        edge_combo = QComboBox()
        edge_combo.addItems(["Rising Edge", "Falling Edge"])
        edge_combo.setCurrentIndex(self.edge)
        layout.addRow("Edge", edge_combo)

        gap_edit = QLineEdit(str(self.min_gap))
        layout.addRow("Min Gap (samples)", gap_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)

        if dlg.exec_() == QDialog.Accepted:
            try:
                self.threshold = float(thresh_edit.text())
                self.edge = edge_combo.currentIndex()
                self.min_gap = int(gap_edit.text())
            except ValueError:
                pass
            return True
        return False

    # ------------------------------------------------------------------
    def calculate(self, signal: Signal) -> FilterResult:
        data = signal.data
        fs = signal.rate

        if self.edge == 1:  # falling
            crossings = np.where(data <= self.threshold)[0]
        else:               # rising
            crossings = np.where(data >= self.threshold)[0]

        # Keep only first sample of each contiguous run (gap > min_gap)
        if len(crossings) == 0:
            qrs = np.array([], dtype=int)
        else:
            diffs = np.diff(crossings)
            keep = np.concatenate([[0], np.where(diffs > self.min_gap)[0] + 1])
            qrs = crossings[keep]

        bpm = 0.0
        if len(qrs) > 1:
            bpm = fs / np.mean(np.diff(qrs)) * 60

        t = np.arange(len(data)) / fs
        return FilterResult(
            data=data.copy(),
            t=t,
            rate=fs,
            output_text=f"Beat Detector: {len(qrs)} beats, Avg BPM = {bpm:.2f}",
            passthrough=True,
            extras={"qrs": qrs, "bpm": bpm, "threshold": self.threshold},
        )

    # ------------------------------------------------------------------
    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8, label="ECG")

        qrs = result.extras.get("qrs", np.array([]))
        if len(qrs):
            ax.plot(result.t[qrs], result.data[qrs], "r*", markersize=8, label="Beats")

        thresh = result.extras.get("threshold", self.threshold)
        ax.axhline(thresh, color="#e74c3c", linestyle="--", linewidth=1, label=f"Thresh={thresh:.3f}")
        ax.legend(fontsize=7)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Beat Detector  |  BPM={result.extras.get('bpm', 0):.1f}", fontsize=9)
        ax.grid(True, alpha=0.3)
