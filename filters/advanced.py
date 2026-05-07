"""
filters/advanced.py  –  Higher-level ECG algorithms.

Covers:
  • AveragingFilter      (ECGaveraging.m)
  • TemplateMatchFilter  (ECGtemplate.m)
  • AdaptiveFilter       (ECGadaptiveFilter.m)
  • ResampleFilter       (ECGresample.m)
  • CompressionFilter    (ECGcompress.m – turning-point method)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly
from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QComboBox, QCheckBox, QLabel
)
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


# ---------------------------------------------------------------------------
class AveragingFilter(BaseFilter):
    """
    Adds Gaussian noise to N copies of the ECG, then averages them.
    Demonstrates SNR improvement of sqrt(N).
    """
    name = "ECG Averaging"
    passthrough = False
    num_plots = 2

    def __init__(self):
        super().__init__()
        self.num_avg: int = 16
        self.noise_amp: float = 0.1   # mV (std dev)

    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("ECG Averaging Settings")
        layout = QFormLayout(dlg)

        n_edit = QLineEdit(str(self.num_avg))
        a_edit = QLineEdit(str(self.noise_amp * 1000))
        layout.addRow("# Averages", n_edit)
        layout.addRow("Noise Amplitude (mV)", a_edit)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)

        if dlg.exec_() == QDialog.Accepted:
            try:
                self.num_avg   = int(n_edit.text())
                self.noise_amp = float(a_edit.text()) / 1000
            except ValueError:
                pass
            return True
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        n = len(signal.data)
        noise = self.noise_amp * np.random.randn(n, self.num_avg)
        noisy = signal.data[:, None] + noise           # (n, N)
        averaged = np.mean(noisy, axis=1)

        rms_noise   = np.sqrt(np.mean(noise ** 2, axis=0))
        rms_sig     = np.sqrt(np.mean(signal.data ** 2))
        rms_avg_err = np.sqrt(np.mean((averaged - signal.data) ** 2))
        snr_noisy   = rms_sig / np.mean(rms_noise) if np.mean(rms_noise) > 0 else 0
        snr_avg     = rms_sig / rms_avg_err if rms_avg_err > 0 else float('inf')
        expected_imp = np.sqrt(self.num_avg)

        t = signal.t.copy()
        return FilterResult(
            data=averaged, t=t, rate=signal.rate,
            output_text=(
                f"Averaging: N={self.num_avg}  "
                f"SNR noisy={snr_noisy:.2f}  SNR avg={snr_avg:.2f}  "
                f"Expected improvement={expected_imp:.2f}  "
                f"Actual={snr_avg/snr_noisy if snr_noisy > 0 else 0:.2f}"
            ),
            passthrough=False,
            extras={"noisy": noisy, "averaged": averaged, "t": t},
        )

    def plot(self, axes: list, result: FilterResult) -> None:
        t      = result.extras.get("t", result.t)
        noisy  = result.extras.get("noisy", None)
        avg    = result.extras.get("averaged", result.data)

        if len(axes) >= 1:
            ax = axes[0]
            ax.clear()
            if noisy is not None:
                for k in range(noisy.shape[1]):
                    ax.plot(t, noisy[:, k], alpha=0.15, linewidth=0.5, color="#3498db")
            ax.set_title("Noisy Copies", fontsize=9)
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)

        if len(axes) >= 2:
            ax2 = axes[1]
            ax2.clear()
            ax2.plot(t, avg, color="#1a1a2e", linewidth=0.9)
            ax2.set_title("Averaged ECG", fontsize=9)
            ax2.set_ylabel("Amplitude")
            ax2.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
class TemplateMatchFilter(BaseFilter):
    """Cross-correlation template matching."""
    name = "Template Matching"
    passthrough = True
    num_plots = 1

    def __init__(self):
        super().__init__()
        self.template: np.ndarray = np.array([])
        self.mode: int = 1
        self._tmpl_start: int = 0
        self._tmpl_end:   int = 200

    def configure(self, parent=None, signal_data=None, rate=None) -> bool:
        data = signal_data
        fs   = float(rate) if rate is not None else float(getattr(self, "_last_rate", 360.0))
        if data is None or len(data) < 8:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
            d = QDialog(parent)
            d.setWindowTitle("Template Matching")
            lay = QVBoxLayout(d)
            lay.addWidget(QLabel(
                "No signal available yet.\n\n"
                "Static: load a signal, run the chain, then re-open Config.\n"
                "RT: connect the Pico, let it buffer data, then re-open Config."))
            bb = QDialogButtonBox(QDialogButtonBox.Ok)
            bb.accepted.connect(d.accept)
            lay.addWidget(bb)
            d.exec_()
            return False
        return _run_template_selector(self, data, fs, parent)

    def calculate(self, signal) -> FilterResult:
        self._last_rate = signal.rate
        sig = signal.data
        fs  = signal.rate
        if len(self.template) == 0:
            return FilterResult(
                data=sig.copy(), t=signal.t.copy(), rate=fs,
                output_text="Template Matching: use Config to select a template",
                passthrough=True, extras={})
        temp = self.template - float(np.mean(self.template))
        m = len(temp); n = len(sig)
        if m >= n:
            return FilterResult(
                data=sig.copy(), t=signal.t.copy(), rate=fs,
                output_text="Template Matching: template longer than signal",
                passthrough=True, extras={})
        Rxy = np.zeros(n); Dxy = np.zeros(n)
        t2  = float(np.sum(temp ** 2))
        for i in range(n - m):
            s  = sig[i:i + m] - float(np.mean(sig[i:i + m]))
            s2 = float(np.sum(s ** 2))
            denom = float(np.sqrt(s2 * t2))
            Rxy[i + m - 1] = float(np.sum(s * temp)) / denom if denom > 0 else 0.0
            Dxy[i + m - 1] = float(np.sum(np.abs(s - temp)))
        data_out = Rxy if self.mode == 1 else Dxy
        t_arr = np.arange(n) / fs
        return FilterResult(
            data=data_out, t=t_arr, rate=fs,
            output_text=f"Template Match: {'correlation' if self.mode else 'difference'}, {m} samples",
            passthrough=True,
            extras={"Rxy": Rxy, "Dxy": Dxy, "template": self.template})

    def plot(self, axes, result):
        ax = axes[0]; ax.clear()
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax.plot(result.t, result.data, color=colour, linewidth=0.8)
        label = "Correlation (Rxy)" if self.mode == 1 else "Difference (Dxy)"
        ax.set_title(f"Template Matching ({label})", fontsize=9)
        ax.set_ylabel(label); ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
def _run_template_selector(filt, data, rate, parent):
    """
    Interactive template selector with zoom/pan toolbar.
    Click [Select...] then click two points on the upper plot to define
    the template window.  The lower plot previews the selected template.
    Returns True if saved, False if cancelled.
    """
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
        QPushButton, QLabel, QLineEdit, QGroupBox,
        QRadioButton, QDialogButtonBox, QSizePolicy,
        QWidget, QToolBar)
    from PyQt5.QtCore import Qt
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavToolbar)
    from matplotlib.figure import Figure

    dlg = QDialog(parent)
    dlg.setWindowTitle("Template Selector")
    dlg.resize(1100, 720)
    root = QVBoxLayout(dlg)
    root.setSpacing(4)

    # ── Two-subplot figure ────────────────────────────────────────────────────
    fig = Figure(facecolor="#f5f6fa")
    fig.subplots_adjust(hspace=0.45, left=0.08, right=0.98, top=0.93, bottom=0.10)
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    ax_sig  = fig.add_subplot(2, 1, 1)
    ax_tmpl = fig.add_subplot(2, 1, 2)

    t_full = np.arange(len(data)) / rate

    # Default selection: first beat-width window (0.5s or first 200 samples)
    default_end = min(len(data) - 1, int(getattr(filt, "_tmpl_end",
                                                  min(int(0.5 * rate), len(data) - 1))))
    default_start = int(getattr(filt, "_tmpl_start", 0))
    if default_end <= default_start:
        default_end = min(default_start + int(0.5 * rate), len(data) - 1)

    sel = {
        "s":        default_start,
        "e":        default_end,
        "clicking": False,
        "click1":   None,
    }

    # ── Draw functions ────────────────────────────────────────────────────────
    _vlines = []

    def draw_signal():
        """Redraw the signal axis (preserves zoom if already set)."""
        xlim = ax_sig.get_xlim()
        ylim = ax_sig.get_ylim()
        ax_sig.clear()
        ax_sig.plot(t_full, data, color="#1a1a2e", lw=0.7, zorder=1)
        ax_sig.set_title(
            "Signal  —  click [Select...] then click two points to define template",
            fontsize=9)
        ax_sig.set_ylabel("Amplitude", fontsize=8)
        ax_sig.grid(True, alpha=0.25, color="#cccccc")
        s, e = sel["s"], sel["e"]
        if e > s and e < len(t_full):
            ax_sig.axvline(t_full[s], color="#e74c3c", lw=1.5, zorder=2)
            ax_sig.axvline(t_full[e], color="#e74c3c", lw=1.5, zorder=2)
            ax_sig.axvspan(t_full[s], t_full[e], alpha=0.12,
                           color="#e74c3c", zorder=1)
        # Restore zoom only if it was actually set (not the default full range)
        if xlim != (0.0, 1.0) and xlim[0] != xlim[1]:
            ax_sig.set_xlim(xlim)
            ax_sig.set_ylim(ylim)
        canvas.draw_idle()

    def draw_preview():
        s, e = sel["s"], sel["e"]
        ax_tmpl.clear()
        if e > s and e < len(data):
            tmpl   = data[s : e + 1]
            t_tmpl = np.arange(len(tmpl)) / rate
            ax_tmpl.plot(t_tmpl, tmpl, color="#1565C0", lw=1.0)
            ax_tmpl.set_title(
                f"Template preview  ({len(tmpl)} samples, "
                f"{len(tmpl)/rate*1000:.0f} ms)", fontsize=9)
            start_edit.blockSignals(True); end_edit.blockSignals(True)
            start_edit.setText(f"{t_full[s]:.4f}")
            end_edit.setText(f"{t_full[e]:.4f}")
            start_edit.blockSignals(False); end_edit.blockSignals(False)
        else:
            ax_tmpl.set_title("Template preview", fontsize=9)
        ax_tmpl.set_ylabel("Amplitude", fontsize=8)
        ax_tmpl.set_xlabel("Time (s)", fontsize=8)
        ax_tmpl.grid(True, alpha=0.25, color="#cccccc")
        canvas.draw_idle()

    def draw_all():
        draw_signal()
        draw_preview()

    # ── Click handler ─────────────────────────────────────────────────────────
    def on_canvas_click(event):
        if not sel["clicking"]:
            return
        if event.inaxes is not ax_sig or event.xdata is None:
            return
        if sel["click1"] is None:
            sel["click1"] = event.xdata
            select_btn.setText("Click 2nd point…")
        else:
            x1, x2 = sorted([sel["click1"], event.xdata])
            sel["s"] = max(0, int(x1 * rate))
            sel["e"] = min(len(data) - 1, int(x2 * rate))
            sel["clicking"] = False
            sel["click1"]   = None
            select_btn.setText("Select…")
            # Restore normal cursor
            canvas.setCursor(Qt.ArrowCursor)
            draw_all()

    canvas.mpl_connect("button_press_event", on_canvas_click)

    # ── Matplotlib nav toolbar (zoom/pan built-in) ────────────────────────────
    toolbar = NavToolbar(canvas, dlg)
    toolbar.setStyleSheet(
        "QToolBar{background:#f0f0f0;border:none;spacing:2px;}"
        "QToolButton{color:#1a1a2e;}")

    # Patch toolbar zoom/pan to disable our click handler while active
    _orig_zoom = toolbar.zoom
    _orig_pan  = toolbar.pan
    def _zoom_patched(*a, **kw):
        sel["clicking"] = False
        _orig_zoom(*a, **kw)
    def _pan_patched(*a, **kw):
        sel["clicking"] = False
        _orig_pan(*a, **kw)
    toolbar.zoom = _zoom_patched
    toolbar.pan  = _pan_patched

    root.addWidget(toolbar)
    root.addWidget(canvas, stretch=4)

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl = QHBoxLayout()

    grp_sel = QGroupBox("Template Selection")
    gsl = QVBoxLayout(grp_sel)
    select_btn = QPushButton("Select…")
    select_btn.setMinimumWidth(120)
    gsl.addWidget(select_btn)
    row_s = QHBoxLayout()
    row_s.addWidget(QLabel("Start (s):"))
    start_edit = QLineEdit(f"{t_full[sel['s']]:.4f}")
    start_edit.setFixedWidth(80); row_s.addWidget(start_edit); gsl.addLayout(row_s)
    row_e = QHBoxLayout()
    row_e.addWidget(QLabel("End (s):"))
    end_edit = QLineEdit(f"{t_full[sel['e']]:.4f}")
    end_edit.setFixedWidth(80); row_e.addWidget(end_edit); gsl.addLayout(row_e)

    hint = QLabel("Tip: use Zoom 🔍 or Pan ✥ to zoom in,\nthen click Select... to pick precise points.")
    hint.setStyleSheet("color:#666; font-size:10px;")
    hint.setWordWrap(True)
    gsl.addWidget(hint)
    ctrl.addWidget(grp_sel)

    grp_mode = QGroupBox("Template Mode")
    gml = QVBoxLayout(grp_mode)
    rb_corr = QRadioButton("Correlation (Rxy)")
    rb_diff = QRadioButton("Difference (Dxy)")
    rb_corr.setChecked(filt.mode == 1)
    rb_diff.setChecked(filt.mode == 0)
    gml.addWidget(rb_corr); gml.addWidget(rb_diff)
    ctrl.addWidget(grp_mode)
    ctrl.addStretch()
    root.addLayout(ctrl)

    btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
    btns.accepted.connect(dlg.accept)
    btns.rejected.connect(dlg.reject)
    root.addWidget(btns)

    # ── Wire callbacks ────────────────────────────────────────────────────────
    def on_select_clicked():
        # Deactivate zoom/pan if active
        if toolbar.mode in ("zoom rect", "pan/zoom"):
            toolbar.zoom()   # toggle off
        sel["clicking"] = True
        sel["click1"]   = None
        select_btn.setText("Click 1st point…")
        canvas.setCursor(Qt.CrossCursor)

    def on_time_edited():
        try:
            s = max(0, int(float(start_edit.text()) * rate))
            e = min(len(data) - 1, int(float(end_edit.text()) * rate))
            if e > s:
                sel["s"] = s; sel["e"] = e
                draw_all()
        except ValueError:
            pass

    select_btn.clicked.connect(on_select_clicked)
    start_edit.returnPressed.connect(on_time_edited)
    end_edit.returnPressed.connect(on_time_edited)

    # Initial draw
    draw_all()

    if dlg.exec_() == QDialog.Accepted:
        s, e = sel["s"], sel["e"]
        if e <= s:
            return False
        filt._tmpl_start = s
        filt._tmpl_end   = e
        filt.mode        = 1 if rb_corr.isChecked() else 0
        filt.template    = data[s : e + 1].copy()
        n = e - s
        filt.name = f"Template Matching ({'Corr' if filt.mode else 'Diff'}, {n} samp)"
        return True
    return False


# ---------------------------------------------------------------------------
class ResampleFilter(BaseFilter):
    """Down-sample or up-sample the signal by a rational factor P/Q."""
    name = "Resample"
    passthrough = True
    num_plots   = 1

    def __init__(self):
        super().__init__()
        self.factor_p: int = 1
        self.factor_q: int = 2

    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("Resample Settings")
        layout = QFormLayout(dlg)
        p_edit = QLineEdit(str(self.factor_p))
        q_edit = QLineEdit(str(self.factor_q))
        layout.addRow("Upsample factor P", p_edit)
        layout.addRow("Downsample factor Q", q_edit)
        layout.addRow(QLabel("Output rate = Input rate \u00d7 P / Q"))
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)
        if dlg.exec_() == QDialog.Accepted:
            try:
                self.factor_p = max(1, int(p_edit.text()))
                self.factor_q = max(1, int(q_edit.text()))
            except ValueError:
                pass
            return True
        return False

    def calculate(self, signal) -> FilterResult:
        p, q   = self.factor_p, self.factor_q
        out    = resample_poly(signal.data, p, q)
        new_fs = signal.rate * p / q
        t      = np.arange(len(out)) / new_fs
        return FilterResult(
            data=out, t=t, rate=new_fs,
            output_text=f"Resample: \u00d7{p}/{q}  \u2192  {new_fs:.1f} Hz",
            passthrough=True)

    def plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear()
        ax.plot(result.t, result.data, color=colour, lw=0.9)
        ax.set_title(f"Resample (\u00d7{self.factor_p}/{self.factor_q})", fontsize=9)
        ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
class CompressionFilter(BaseFilter):
    """Turning-point (TP) data compression."""
    name = "ECG Compression (TP)"
    passthrough = False
    num_plots   = 1

    def __init__(self):
        super().__init__()
        self.ratio: float = 2.0

    def configure(self, parent=None) -> bool:
        dlg = QDialog(parent)
        dlg.setWindowTitle("ECG Compression (TP)")
        layout = QFormLayout(dlg)
        r_edit = QLineEdit(str(self.ratio))
        layout.addRow("Target compression ratio", r_edit)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)
        if dlg.exec_() == QDialog.Accepted:
            try:
                self.ratio = max(1.0, float(r_edit.text()))
            except ValueError:
                pass
            return True
        return False

    def calculate(self, signal) -> FilterResult:
        data   = signal.data
        n      = len(data)
        target = max(2, int(n / self.ratio))
        keep   = [0]
        for i in range(1, n - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1]) or \
               (data[i] < data[i-1] and data[i] < data[i+1]):
                keep.append(i)
        keep.append(n - 1)
        keep = np.array(keep)
        if len(keep) > target:
            idx  = np.linspace(0, len(keep)-1, target, dtype=int)
            keep = keep[idx]
        out = data[keep]
        t   = signal.t[keep]
        actual_ratio = n / max(1, len(out))
        return FilterResult(
            data=out, t=t, rate=signal.rate,
            output_text=f"Compression (TP): {n} \u2192 {len(out)} pts  ratio={actual_ratio:.2f}\u00d7",
            passthrough=False)

    def plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear()
        ax.plot(result.t, result.data, color=colour, lw=0.9, marker=".", ms=2)
        ax.set_title("Compression (TP)", fontsize=9)
        ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)
