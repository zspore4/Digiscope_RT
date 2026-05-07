"""
filters/spectral.py  –  Power spectrum and spectrogram.

Translated from ECGpower.m and ECGspec.m.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox,
    QDialogButtonBox, QCheckBox, QSpinBox
)
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


# ---------------------------------------------------------------------------
class PowerSpectrumFilter(BaseFilter):
    name = "Power Spectrum"
    passthrough = False   # does not feed data downstream
    num_plots = 1
    scroll = False
    update_window = True

    def __init__(self):
        super().__init__()
        self.max_freq: float   = 100.0
        self.nfft: int         = 512
        self.normalize: bool   = True
        self.do_log: bool      = True
        self.window_type: int  = 0    # 0=rect, 1=hanning, 2=triangular
        self.remove_mean: bool = False

    # ------------------------------------------------------------------
    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("Power Spectrum Settings")
        layout = QFormLayout(dlg)

        max_f_edit  = QLineEdit(str(self.max_freq))
        nfft_edit   = QLineEdit(str(self.nfft))
        norm_check  = QCheckBox();  norm_check.setChecked(self.normalize)
        log_check   = QCheckBox();  log_check.setChecked(self.do_log)
        rm_check    = QCheckBox();  rm_check.setChecked(self.remove_mean)

        win_combo = QComboBox()
        win_combo.addItems(["Rectangular", "Hanning", "Triangular"])
        win_combo.setCurrentIndex(self.window_type)

        layout.addRow("Max Frequency (Hz)", max_f_edit)
        layout.addRow("NFFT", nfft_edit)
        layout.addRow("Normalize", norm_check)
        layout.addRow("Log10 scale", log_check)
        layout.addRow("Window", win_combo)
        layout.addRow("Remove Mean", rm_check)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)

        if dlg.exec_() == QDialog.Accepted:
            try:
                self.max_freq    = float(max_f_edit.text())
                self.nfft        = int(nfft_edit.text())
                self.normalize   = norm_check.isChecked()
                self.do_log      = log_check.isChecked()
                self.window_type = win_combo.currentIndex()
                self.remove_mean = rm_check.isChecked()
            except ValueError:
                pass
            return True
        return False

    # ------------------------------------------------------------------
    def calculate(self, signal: Signal) -> FilterResult:
        sig = signal.data.copy()
        if self.remove_mean:
            sig -= np.mean(sig)

        n = len(sig)
        if self.window_type == 1:
            win = np.hanning(n)
        elif self.window_type == 2:
            win = np.bartlett(n)
        else:
            win = np.ones(n)

        nfft = max(self.nfft, n * 2)
        D = 2 * np.abs(np.fft.fft(sig * win, nfft) / n)
        f = signal.rate * np.linspace(0, 1, nfft)

        if self.normalize:
            peak = np.max(D)
            if peak > 0:
                D /= peak

        D = np.clip(D, 1e-6, None)
        if self.do_log:
            D = 20 * np.log10(D)

        peak_idx = int(np.argmax(D))
        return FilterResult(
            data=D, t=f, rate=signal.rate,
            output_text=f"Power: Peak at {f[peak_idx]:.2f} Hz",
            passthrough=False,
            extras={"f": f, "D": D},
        )

    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        f = result.extras.get("f", result.t)
        D = result.extras.get("D", result.data)
        mask = f <= self.max_freq
        ax.plot(f[mask], D[mask], color="#1a1a2e", linewidth=0.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)" if self.do_log else "|Y(f)|")
        ax.set_title("Power Spectrum", fontsize=9)
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
class SpectrogramFilter(BaseFilter):
    name = "Spectrogram"
    passthrough = False
    num_plots = 1
    scroll = False

    def __init__(self):
        super().__init__()
        self.win_len: int  = 64
        self.overlap: int  = 32
        self.nfft: int     = 512
        self.max_freq: float = 100.0

    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("Spectrogram Settings")
        layout = QFormLayout(dlg)

        wl = QLineEdit(str(self.win_len))
        ov = QLineEdit(str(self.overlap))
        nf = QLineEdit(str(self.nfft))
        mf = QLineEdit(str(self.max_freq))
        layout.addRow("Window Length (samples)", wl)
        layout.addRow("Overlap (samples)", ov)
        layout.addRow("NFFT", nf)
        layout.addRow("Max Freq (Hz)", mf)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)

        if dlg.exec_() == QDialog.Accepted:
            try:
                self.win_len  = int(wl.text())
                self.overlap  = int(ov.text())
                self.nfft     = int(nf.text())
                self.max_freq = float(mf.text())
            except ValueError:
                pass
            return True
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        f, t, Sxx = scipy_spectrogram(
            signal.data,
            fs=signal.rate,
            nperseg=self.win_len,
            noverlap=self.overlap,
            nfft=self.nfft,
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        return FilterResult(
            data=Sxx_db, t=t, rate=signal.rate,
            output_text="Spectrogram: computed",
            passthrough=False,
            extras={"f": f, "t": t, "Sxx": Sxx_db},
        )

    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        f   = result.extras.get("f", [])
        t   = result.extras.get("t", result.t)
        Sxx = result.extras.get("Sxx", result.data)
        if len(f) and len(t):
            ax.pcolormesh(t, f, Sxx, shading="gouraud", cmap="viridis")
            ax.set_ylim(0, self.max_freq)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Spectrogram", fontsize=9)
