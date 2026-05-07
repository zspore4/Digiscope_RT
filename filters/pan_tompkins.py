"""
filters/pan_tompkins.py  –  Pan-Tompkins adaptive QRS detector.

Translated from ECGpantompkins.m.
"""

from __future__ import annotations
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox
)
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


class PanTompkinsFilter(BaseFilter):
    name = "PT Thresholding"
    passthrough = True
    num_plots = 1
    scroll = True

    def __init__(self):
        super().__init__()
        self.look_ahead      = 2.0
        self.signal_update   = 0.125
        self.noise_update    = 0.125
        self.signal_sb_update = 0.25
        self.rr_high_perc    = 1.16
        self.rr_low_perc     = 0.92
        self.rr_miss_lim     = 1.66
        self.peak_dur_ms     = 75.0
        self.qrs_update      = 0.475

    # ------------------------------------------------------------------
    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("Pan-Tompkins Settings")
        layout = QFormLayout(dlg)

        fields = {
            "Look-ahead (s)":         ("look_ahead",       self.look_ahead),
            "Signal Update (0-1)":    ("signal_update",    self.signal_update),
            "Noise Update (0-1)":     ("noise_update",     self.noise_update),
            "Searchback Update (0-1)":("signal_sb_update", self.signal_sb_update),
            "RR High Range":          ("rr_high_perc",     self.rr_high_perc),
            "RR Low Range":           ("rr_low_perc",      self.rr_low_perc),
            "RR Miss Limit":          ("rr_miss_lim",      self.rr_miss_lim),
            "Peak Search Range (ms)": ("peak_dur_ms",      self.peak_dur_ms),
            "QRS Update":             ("qrs_update",       self.qrs_update),
        }
        edits = {}
        for label, (attr, val) in fields.items():
            e = QLineEdit(str(val))
            edits[attr] = e
            layout.addRow(label, e)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)

        if dlg.exec_() == QDialog.Accepted:
            try:
                for attr, e in edits.items():
                    setattr(self, attr, float(e.text()))
            except ValueError:
                pass
            return True
        return False

    # ------------------------------------------------------------------
    def calculate(self, signal: Signal) -> FilterResult:
        sig = signal.data
        fs  = signal.rate
        n   = len(sig)

        peak_dur  = int(np.ceil(self.peak_dur_ms / 1000 * fs))
        pre_blank = int(0.2 * fs)

        # Adaptive initial threshold: use signal amplitude if available,
        # not a hardcoded ADC-count value (signal may already be in volts)
        sig_rms = float(np.sqrt(np.mean(sig[:min(len(sig), int(fs))] ** 2)))
        thresh1 = max(2 / 2**12, sig_rms * 0.05)
        thresh2 = thresh1 / 2

        qrs_peak_buf = np.zeros(n)
        qrs_pos      = np.zeros(n, dtype=int)
        qrs_real     = np.zeros(n, dtype=int)
        noise_peak_buf = np.zeros(8)
        rr_buf       = np.full(n, fs, dtype=float)
        thresh1_arr  = np.zeros(n)
        thresh2_arr  = np.zeros(n)

        noise_pos = 0
        pk_cnt    = 0
        qpk_cnt   = 0
        cnt       = 0
        sb_count  = int(1.5 * fs)
        sb_peak   = 0.0
        sb_loc    = 0

        d_max      = 0.0
        pk_tmp2    = 0
        pre_blank_cnt = 0
        temp_peak  = 0.0
        pk_tmp     = 0
        time_since_max = 0
        q_median   = 0.0
        n_median   = 0.0
        init_blank = 0
        init_max   = 0.0
        rset_buf   = np.zeros(8)
        rset_buf_pos = 0

        v2 = 0.0

        for pos in range(n):
            v = sig[pos]

            if time_since_max > 0:
                time_since_max += 1

            # ---- peak detection ----
            pk = 0.0
            if v > v2 and v > d_max:
                d_max   = v
                pk_tmp2 = pos
                if d_max > 2 / 2**signal.resolution:
                    time_since_max = 1
            elif v < d_max / 2:
                pk = d_max
                d_max = 0.0
                time_since_max = 0
            elif time_since_max > int(0.095 * fs):
                pk = d_max
                d_max = 0.0
                time_since_max = 0

            new_peak = 0.0
            if pk != 0.0 and pre_blank_cnt == 0:
                temp_peak     = pk
                pk_tmp        = pk_tmp2
                pre_blank_cnt = pre_blank
            elif pk == 0.0 and pre_blank_cnt != 0:
                pre_blank_cnt -= 1
                if pre_blank_cnt == 0:
                    new_peak = temp_peak
            elif pk != 0.0:
                if pk > temp_peak:
                    temp_peak     = pk
                    pre_blank_cnt = pre_blank
                else:
                    pre_blank_cnt -= 1
                    if pre_blank_cnt == 0:
                        new_peak = temp_peak

            # ---- initialisation phase (first 8 QRS) ----
            if qpk_cnt < 9:
                cnt += 1
                if new_peak > 0:
                    cnt = peak_dur
                init_blank += 1
                if init_blank == fs or new_peak > 0:
                    if new_peak > thresh1:
                        init_blank = 0
                        qrs_peak_buf[pk_cnt] = new_peak
                        qrs_pos[pk_cnt]  = pos
                        qrs_real[pk_cnt] = pk_tmp
                        pk_cnt   += 1
                        init_max  = 0.0
                        qpk_cnt  += 1
                        q_median  = np.median(qrs_peak_buf[:pk_cnt])
                        n_median  = 0.0
                        sb_count  = int(fs * 1.65)
                        thresh1   = self._calc_thresh(q_median, n_median)
                        thresh2   = thresh1 / 2
                    else:
                        noise_peak_buf[noise_pos] = new_peak
                        noise_pos = (noise_pos + 1) % 8
                        n_median  = np.median(noise_peak_buf)
                        thresh1   = self._calc_thresh(q_median, n_median)
                        thresh2   = thresh1 / 2
                if new_peak > init_max:
                    init_max = new_peak
            else:
                cnt += 1
                if new_peak > 0:
                    if new_peak > thresh1:
                        qrs_peak_buf[pk_cnt] = new_peak
                        qrs_pos[pk_cnt]  = pos
                        qrs_real[pk_cnt] = pk_tmp
                        q_median  = np.median(qrs_peak_buf[max(0, pk_cnt-7):pk_cnt+1])
                        thresh1   = self._calc_thresh(q_median, n_median)
                        rr_buf[pk_cnt] = cnt
                        pk_cnt   += 1
                        rr_median = np.median(rr_buf[max(0, pk_cnt-7):pk_cnt])
                        sb_count  = int(rr_median + rr_median / 2 + peak_dur)
                        cnt       = peak_dur
                        sb_peak   = 0.0
                        init_blank = 0
                        init_max   = 0.0
                        rset_buf_pos = 0
                    else:
                        noise_peak_buf[noise_pos] = new_peak
                        noise_pos = (noise_pos + 1) % 8
                        n_median  = np.median(noise_peak_buf)
                        thresh1   = self._calc_thresh(q_median, n_median)
                        thresh2   = thresh1 / 2
                        if new_peak > sb_peak and cnt - peak_dur >= int(fs * 0.36):
                            sb_peak = new_peak
                            sb_loc  = cnt - peak_dur

                if cnt > sb_count and sb_peak > thresh2:
                    qrs_delay = cnt - sb_loc
                    cnt -= sb_loc
                    qrs_peak_buf[pk_cnt] = sb_peak
                    qrs_pos[pk_cnt]  = pos - qrs_delay
                    qrs_real[pk_cnt] = pk_tmp
                    q_median = np.median(qrs_peak_buf[max(0, pk_cnt-8):pk_cnt])
                    thresh1  = self._calc_thresh(q_median, n_median)
                    thresh2  = thresh1 / 2
                    rr_buf[pk_cnt] = sb_loc
                    pk_cnt  += 1
                    rr_median = np.median(rr_buf[max(0, pk_cnt-8):pk_cnt])
                    sb_count  = int(rr_median + rr_median / 2 + peak_dur)
                    sb_peak   = 0.0
                    init_blank = 0
                    init_max   = 0.0
                    rset_buf_pos = 0

            thresh1_arr[pos] = thresh1
            thresh2_arr[pos] = thresh2
            v2 = v

        qrs_pos  = qrs_pos[:pk_cnt]
        qrs_real = qrs_real[:pk_cnt]
        rr_valid = rr_buf[:pk_cnt]

        bpm = 0.0
        if len(rr_valid) > 1:
            bpm = 60.0 * fs / np.mean(rr_valid[1:])

        t = np.arange(n) / fs
        return FilterResult(
            data=sig.copy(),
            t=t,
            rate=fs,
            output_text=f"Pan-Tompkins: {len(qrs_pos)} beats, HR={bpm:.1f} BPM",
            passthrough=True,
            extras={
                "qrs":      qrs_pos,
                "qrs_real": qrs_real,
                "rr":       rr_valid,
                "thresh1":  thresh1_arr,
                "thresh2":  thresh2_arr,
                "bpm":      bpm,
            },
        )

    def _calc_thresh(self, q_median: float, n_median: float) -> float:
        return n_median + self.qrs_update * (q_median - n_median)

    # ------------------------------------------------------------------
    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8, label="ECG")

        qrs = result.extras.get("qrs", [])
        if len(qrs):
            ax.plot(result.t[qrs], result.data[qrs], "r*", markersize=8, label="QRS (detect)")

        qrs_real = result.extras.get("qrs_real", [])
        if len(qrs_real):
            ax.plot(result.t[qrs_real], result.data[qrs_real], "g*", markersize=8, label="QRS (real)")

        t1 = result.extras.get("thresh1")
        if t1 is not None:
            ax.plot(result.t, t1, color="#2ecc71", linewidth=1, linestyle="--", label="Thresh1")

        ax.legend(fontsize=7)
        ax.set_title(f"PT Thresholding  |  BPM={result.extras.get('bpm', 0):.1f}", fontsize=9)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
