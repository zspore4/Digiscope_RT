"""
filters/fft_filter.py  –  FFT-bin energy filter.

Port of FFTfilter.m.

Computes a short-time spectrogram of the signal, extracts the mean power
in user-selected frequency bins, and outputs the result as a new time-series
(the instantaneous band power over time).

This is useful for:
  • Extracting power in a specific frequency band (e.g. 0.5-40 Hz ECG band)
  • Tracking how band power changes over time
  • Removing out-of-band noise by selecting only the relevant bins

Parameters
----------
win_len   : STFT window length (samples)
overlap   : STFT overlap (samples)
nfft      : FFT size
bin_width : width of each selectable bin (Hz)
bins      : list of bin indices to average (1-based, like MATLAB original)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QDialogButtonBox,
    QLineEdit, QLabel, QHBoxLayout, QVBoxLayout,
    QGroupBox, QPushButton, QListWidget, QListWidgetItem,
    QAbstractItemView, QMessageBox, QWidget
)
from PyQt5.QtCore import Qt
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


class FFTBandFilter(BaseFilter):
    """
    FFT bin-energy filter — extracts mean power in selected frequency bins
    over time using a sliding STFT window.
    """
    name = "FFT Band Filter"
    passthrough = True    # output the band-power envelope as the new signal
    num_plots = 1

    def __init__(self):
        super().__init__()
        self.win_len   : int   = 64      # window length in samples
        self.overlap   : int   = 32      # overlap in samples
        self.nfft      : int   = 512     # FFT size
        self.bin_width : float = 1.0     # Hz per selectable bin
        self.bins      : list  = [1]     # which bins to average (1-based)
        self._last_freqs: list = []      # stored for display

    # ──────────────────────────────────────────────────────────────────────
    def configure(self, parent=None) -> bool:
        dlg = _FFTBandDialog(self, parent)
        if dlg.exec_() == QDialog.Accepted:
            self.win_len   = dlg.win_len()
            self.overlap   = dlg.overlap()
            self.nfft      = dlg.nfft()
            self.bin_width = dlg.bin_width()
            self.bins      = dlg.bins()
            bin_str = ",".join(str(b) for b in self.bins[:4])
            if len(self.bins) > 4:
                bin_str += "…"
            self.name = f"FFT Band [{bin_str}]  bw={self.bin_width:.1f}Hz"
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    def calculate(self, signal: Signal) -> FilterResult:
        fs   = signal.rate
        data = signal.data
        n    = len(data)

        win_len = min(self.win_len, n // 2)
        overlap = min(self.overlap, win_len - 1)
        nfft    = max(self.nfft, win_len)

        try:
            f, t_seg, Sxx = scipy_spectrogram(
                data, fs=fs,
                nperseg=win_len,
                noverlap=overlap,
                nfft=nfft,
            )
        except Exception as e:
            return FilterResult(
                data=data.copy(), t=signal.t.copy(), rate=fs,
                output_text=f"FFT Band error: {e}", passthrough=True)

        # Find frequency bin indices matching bin_width
        bin_hz  = f[1] - f[0] if len(f) > 1 else 1.0
        bins_per_group = max(1, int(round(self.bin_width / bin_hz)))

        # Convert 1-based bin numbers to 0-based frequency indices
        freq_indices = []
        for b in self.bins:
            lo = (b - 1) * bins_per_group
            for k in range(bins_per_group):
                idx = lo + k
                if idx < len(f):
                    freq_indices.append(idx)

        if not freq_indices:
            freq_indices = list(range(min(bins_per_group, len(f))))

        # Store the actual frequencies for the output label
        self._last_freqs = [float(f[i]) for i in freq_indices if i < len(f)]

        # Mean power in selected bins over time
        band_power = np.mean(Sxx[freq_indices, :], axis=0)

        # Build output time vector aligned to the segment centres
        out_rate = float(1.0 / (t_seg[1] - t_seg[0])) if len(t_seg) > 1 else fs
        t_out    = np.arange(len(band_power)) / out_rate

        f_lo = min(self._last_freqs) if self._last_freqs else 0.0
        f_hi = max(self._last_freqs) if self._last_freqs else 0.0

        return FilterResult(
            data=band_power, t=t_out, rate=out_rate,
            output_text=(f"FFT Band: {f_lo:.1f}–{f_hi:.1f} Hz  "
                         f"bins={self.bins}  "
                         f"out_rate={out_rate:.1f} Hz"),
            passthrough=True,
            extras={"f": f, "Sxx": Sxx, "t_seg": t_seg,
                    "freq_indices": freq_indices},
        )

    # ──────────────────────────────────────────────────────────────────────
    def plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear()
        try:
            from ui.app import _style_ax
            _style_ax(ax)
        except ImportError:
            ax.grid(True, alpha=0.3)
        ax.plot(result.t, result.data, color=colour, linewidth=0.9)
        f_lo = min(self._last_freqs) if self._last_freqs else "?"
        f_hi = max(self._last_freqs) if self._last_freqs else "?"
        ax.set_title(f"FFT Band Power  {f_lo:.1f}–{f_hi:.1f} Hz", fontsize=9)
        ax.set_ylabel("Power"); ax.set_xlabel("Time (s)")


# ── Config dialog ──────────────────────────────────────────────────────────────

class _FFTBandDialog(QDialog):
    """
    Configuration dialog for the FFT Band Filter.

    Shows a live frequency axis so the user can see exactly which
    frequencies correspond to each bin number before confirming.
    """

    def __init__(self, filt: FFTBandFilter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FFT Band Filter Settings")
        self.setMinimumWidth(420)
        root = QVBoxLayout(self)

        # ── STFT parameters ───────────────────────────────────────────────
        grp_stft = QGroupBox("STFT Parameters")
        fl = QFormLayout(grp_stft)
        self._win  = QLineEdit(str(filt.win_len))
        self._ovlp = QLineEdit(str(filt.overlap))
        self._nfft = QLineEdit(str(filt.nfft))
        fl.addRow("Window Length (samples):", self._win)
        fl.addRow("Overlap (samples):",       self._ovlp)
        fl.addRow("NFFT:",                    self._nfft)
        root.addWidget(grp_stft)

        # ── Bin selection ─────────────────────────────────────────────────
        grp_bins = QGroupBox("Frequency Bin Selection")
        bl = QVBoxLayout(grp_bins)

        bw_row = QHBoxLayout()
        bw_row.addWidget(QLabel("Bin Width (Hz):"))
        self._bw = QLineEdit(str(filt.bin_width))
        self._bw.setMaximumWidth(80)
        bw_row.addWidget(self._bw)
        bw_row.addStretch()
        bl.addLayout(bw_row)

        bl.addWidget(QLabel(
            "Bins to include (space or comma separated, 1-based):\n"
            "  e.g.  '1 2 3'  selects the first three bin groups\n"
            "  Each bin group covers 'Bin Width' Hz"))
        self._bins_edit = QLineEdit(
            " ".join(str(b) for b in filt.bins))
        bl.addWidget(self._bins_edit)

        note = QLabel(
            "The output is the mean STFT power in the selected bins over time.\n"
            "Bin 1 = 0 Hz,  Bin 2 = BinWidth Hz,  Bin 3 = 2×BinWidth Hz …\n"
            "Use consecutive bins to select a band, e.g. '1 2 3 4 5' for 0–5 Hz."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#4a5270; font-size:10px;")
        bl.addWidget(note)
        root.addWidget(grp_bins)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def win_len(self)   -> int:   return max(4, int(self._win.text() or "64"))
    def overlap(self)   -> int:   return max(0, int(self._ovlp.text() or "32"))
    def nfft(self)      -> int:   return max(8, int(self._nfft.text() or "512"))
    def bin_width(self) -> float: return max(0.1, float(self._bw.text() or "1"))
    def bins(self)      -> list:
        try:
            return [int(x) for x in
                    self._bins_edit.text().replace(",", " ").split()
                    if x.strip().isdigit()]
        except Exception:
            return [1]
