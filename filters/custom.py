"""
filters/custom.py  –  Custom Filter designer + Custom Python filter.

CustomCoeffFilter  : full FilterGUI equivalent —
    • Mode switcher: Preset / FIR Impulse / FIR Zero-Place / IIR Impulse /
                     IIR Two-Pole / IIR Pole-Zero / Integer
    • 4-panel analysis plots: magnitude, impulse, phase, pole-zero
    • Mouse coordinate readout on every plot
    • Transfer function H(z) rendered as LaTeX fraction
    • Cascade N times button
    • Save/load .fil format

CustomPythonFilter : arbitrary NumPy expression.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, filtfilt, freqz, tf2zpk, zpk2tf
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLineEdit, QDialogButtonBox, QLabel, QTextEdit, QWidget,
    QComboBox, QCheckBox, QGroupBox, QSizePolicy, QPushButton,
    QListWidget, QListWidgetItem, QSpinBox, QSplitter, QFrame,
    QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


# ── Preset library (from FilterGUI.m setupDefaultFilters) ────────────────────
# Pan-Tompkins coefficients matching the original DigiScope .fil files exactly.
# LP  b=[1,0,0,0,0,0,-2,0,0,0,0,0,1]  a=[1,-2,1]   (13 num, 3 den)
_PT_LP_B = [1.0,0.0,0.0,0.0,0.0,0.0,-2.0,0.0,0.0,0.0,0.0,0.0,1.0]
_PT_LP_A = [1.0, -2.0, 1.0]

# HP  b[-1/32, 0*15, 1,-1, 0*14, 1/32]  a=[1,-1]   (33 num, 2 den)
_PT_HP_B = ([-0.031250] + [0.0]*15 + [1.0, -1.0] + [0.0]*14 + [0.031250])
_PT_HP_A = [1.0, -1.0]

# Derivative (5-pt, 0.1 gain)
_PT_DERIV_B = [-1, -2, 0, 2, 1]
_PT_DERIV_A = [10]

PRESETS = {
    "Custom (enter below)":     (None, None),
    # ── Pan-Tompkins stages ──────────────────────────────────────────────────
    "PT Lowpass":               (_PT_LP_B, _PT_LP_A),
    "PT Highpass":              (_PT_HP_B, _PT_HP_A),
    "PT Derivative":            (_PT_DERIV_B, _PT_DERIV_A),
    # ── General ───────────────────────────────────────────────────────────────
    "Hanning":                 ([1,2,1], [4]),
    "Derivative 2nd order":    ([2,1,0,-1,-2], [10]),
    "Derivative 3rd order":    ([3,2,1,0,-1,-2,-3], [28]),
    "Derivative 5th order":    ([5,4,3,2,1,0,-1,-2,-3,-4,-5], [110]),
    "Savitzky-Golay poly2":    ([-3,12,17,12,-3], [35]),
    "Savitzky-Golay poly3":    ([-2,3,6,7,6,3,-2], [21]),
    "Savitzky-Golay poly4":    ([-21,14,39,54,59,54,39,14,-21], [231]),
    "Rect Integration":        ([0,1], [1,-1]),
    "Trap Integration":        ([1,1], [2,-2]),
    "Simpson's Rule":          ([1,4,2], [3,0,-3]),
    "5-pt Moving Average":     ([1,1,1,1,1], [5]),
    "30-pt Moving Average":    (list(np.ones(30, dtype=int)), [30]),
}

DESIGN_MODES = [
    "Preset",
    "FIR: Impulse Response",
    "FIR: Zero-Place",
    "IIR: Impulse Response",
    "IIR: Two-Pole",
    "IIR: Pole-Zero Place",
    "Integer Filter",
]

INTEGER_ZEROS   = ["Positive (1 + z^-N)", "Negative (1 - z^-N)"]
INTEGER_POLES   = ["Pole at 0 deg (1 - z^-1)", "Pole at 180 deg (1 + z^-1)",
                   "Poles at ±60 deg", "Poles at ±90 deg",
                   "Poles at ±120 deg", "No poles (FIR)"]
TWO_POLE_TYPES  = ["Low-Pass", "High-Pass", "Band-Pass", "Band-Stop"]
TWO_POLE_FORMS  = ["Polar (deg)", "Frequency (Hz)", "Normalized (0-0.5)"]


def _style_ax(ax):
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#333355", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#c8ccd8")
    ax.xaxis.label.set_color("#4a5270")
    ax.yaxis.label.set_color("#4a5270")
    ax.title.set_color("#1a1a2e")
    ax.grid(True, color="#eeeef5", linewidth=0.7)


def _parse(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.replace(",", " ").split()], dtype=float)


def _read_fil(path: str) -> tuple:
    """
    Read a DigiScope .fil file, return (b, a) numpy arrays.
    Handles both the original DigiScope format (DigiFilRead.m) and the
    format written by our own _save_filter().
    """
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) < 4:
        raise ValueError("Not a valid .fil file (too few lines)")
    # line 0: num numerator coefficients
    # line 1: 0=FIR / 1=IIR
    # line 2: 0=float / 1=integer (ignored)
    # line 3: tab-separated numerator coefficients
    is_iir = int(lines[1])
    b = np.array([float(x) for x in lines[3].replace(",", " ").split()],
                 dtype=float)
    a = np.array([1.0])
    if is_iir and len(lines) >= 6:
        a = np.array([float(x) for x in lines[5].replace(",", " ").split()],
                     dtype=float)
    return b, a


def _normalise(b: np.ndarray, a: np.ndarray):
    """If a is a single scalar gain, fold it into b."""
    if len(a) == 1 and a[0] != 1.0:
        return b / a[0], np.array([1.0])
    return b, a


def _poly2str(coeffs: np.ndarray) -> str:
    """Format polynomial as H(z) numerator/denominator unicode string."""
    c = np.array(coeffs, dtype=float)
    c[np.abs(c) < 1e-10] = 0.0
    parts = []
    for i, v in enumerate(c):
        if v == 0:
            continue
        sign = "+" if v > 0 else "-"
        av = abs(v)
        av_str = f"{av:.4g}" if av != 1.0 or i == 0 else ""
        if i == 0:
            parts.append(f"{v:.4g}")
        elif i == 1:
            parts.append(f" {sign} {av_str}z⁻¹")
        else:
            parts.append(f" {sign} {av_str}z⁻{i}")
    return "".join(parts) if parts else "0"


def _poly2mathtext(coeffs: np.ndarray) -> str:
    """
    Format polynomial coefficients as a matplotlib mathtext string.
    Returns something like:  1 + 2z^{-1} + z^{-2}
    Uses matplotlib mathtext (dollar-sign math) for pretty rendering.
    """
    c = np.array(coeffs, dtype=float)
    c[np.abs(c) < 1e-10] = 0.0
    parts = []
    for i, v in enumerate(c):
        if v == 0:
            continue
        sign = "+" if v > 0 else "-"
        av = abs(v)
        # Format the coefficient value
        if av == 1.0 and i > 0:
            av_str = ""
        elif av == int(av):
            av_str = str(int(av))
        else:
            av_str = f"{av:.4g}"
        if i == 0:
            # First term — include sign only if negative
            parts.append(f"{v:.4g}" if av != int(av) else str(int(v)))
        elif i == 1:
            parts.append(f" {sign} {av_str}z^{{-1}}")
        else:
            parts.append(f" {sign} {av_str}z^{{-{i}}}")
    return "".join(parts) if parts else "0"


# ── Main filter class ─────────────────────────────────────────────────────────

class CustomCoeffFilter(BaseFilter):
    name = "Custom Filter"
    passthrough = True
    num_plots = 1

    def __init__(self):
        super().__init__()
        self.b: list = [1]
        self.a: list = [1]
        self.zero_phase: bool = True
        self._label: str = "Custom"

    def configure(self, parent=None) -> bool:
        dlg = FilterGUIDialog(self, parent)
        if dlg.exec_() == QDialog.Accepted:
            self.b         = dlg.b()
            self.a         = dlg.a()
            self.zero_phase = dlg.zero_phase()
            self._label    = dlg.filter_label()
            self.name      = f"Custom Filter: {self._label[:30]}"
            return True
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        x = signal.data; fs = signal.rate
        b, a = _normalise(np.array(self.b, dtype=float),
                          np.array(self.a, dtype=float))

        # Check whether poles are stable (all inside unit circle).
        # filtfilt cannot handle poles on or outside the unit circle
        # (e.g. the PT LP has double poles at z=1).
        # For unstable/marginally-stable filters, fall back to lfilter.
        def _is_stable(a_poly):
            if len(a_poly) <= 1:
                return True
            poles = np.roots(a_poly)
            return bool(np.all(np.abs(poles) < 1.0 - 1e-8))

        note = ""
        try:
            if self.zero_phase and _is_stable(a):
                d = filtfilt(b, a, x)
            else:
                if self.zero_phase and not _is_stable(a):
                    note = " (zero-phase skipped: filter has poles on/outside unit circle)"
                d = lfilter(b, a, x)
        except Exception as e:
            # Last-resort fallback: try lfilter
            try:
                d = lfilter(b, a, x)
                note = f" (filtfilt failed: {e} — used lfilter)"
            except Exception as e2:
                d = x.copy()
                return FilterResult(data=d, t=np.arange(len(d))/fs, rate=fs,
                                    output_text=f"Filter error: {e2}",
                                    passthrough=True)
        t = np.arange(len(d)) / fs
        return FilterResult(data=d, t=t, rate=fs,
                            output_text=f"Custom Filter [{self._label}]{note}",
                            passthrough=True, extras={"b": b, "a": a})

    def plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear(); _style_ax(ax)
        ax.plot(result.t, result.data, color=colour, linewidth=0.9)
        ax.set_title(f"Custom Filter: {self._label}", fontsize=9)
        ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")


# ── FilterGUI dialog ──────────────────────────────────────────────────────────

class FilterGUIDialog(QDialog):
    """
    Full port of FilterGUI.m.
    Left panel: filter type selector + mode-specific controls.
    Right panel: 2×2 analysis plots + H(z) label.
    """

    def __init__(self, filt: CustomCoeffFilter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FilterGUI")
        self.resize(1200, 780)
        self._b = np.array(filt.b, dtype=float)
        self._a = np.array(filt.a, dtype=float)
        self._zero_phase = filt.zero_phase
        self._label = filt._label
        self._fs = 200.0
        self._normalize = True
        self._cascades = 0
        self._zero_place_list: list = []   # list of complex numbers for zero-place mode
        self._pole_place_list: list = []

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(340)
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0); ll.setSpacing(6)
        root.addWidget(left)

        # Top row: filter type selector, Save, Create, Return
        top_row = QHBoxLayout()
        self._type_combo = QComboBox()
        self._type_combo.addItem("Custom Filter")
        self._type_combo.setMinimumWidth(160)
        self._btn_save   = QPushButton("Save Filter…")
        self._btn_create = QPushButton("Create Filter")
        self._btn_create.setObjectName("primary")
        self._btn_return = QPushButton("Return")
        self._btn_return.setObjectName("green")
        self._btn_save.clicked.connect(self._save_filter)
        self._btn_create.clicked.connect(self._create_filter)
        self._btn_return.clicked.connect(self.accept)
        top_row.addWidget(self._type_combo, stretch=1)
        top_row.addWidget(self._btn_save)
        top_row.addWidget(self._btn_create)
        top_row.addWidget(self._btn_return)
        ll.addLayout(top_row)

        # Normalize + sampling rate row
        params_row = QHBoxLayout()
        self._norm_chk = QCheckBox("Normalize")
        self._norm_chk.setChecked(True)
        self._norm_chk.stateChanged.connect(self._draw_plots)
        self._show_3db_chk = QCheckBox("-3 dB line")
        self._show_3db_chk.setChecked(True)
        self._show_3db_chk.stateChanged.connect(self._draw_plots)
        lbl_fs = QLabel("Sampling Rate:")
        self._fs_edit = QLineEdit("200")
        self._fs_edit.setMaximumWidth(70)
        self._fs_edit.editingFinished.connect(self._on_fs_changed)
        lbl_casc = QLabel("# Cascades:")
        self._casc_spin = QSpinBox()
        self._casc_spin.setRange(0, 20)
        self._casc_spin.setValue(0)
        self._casc_spin.setMaximumWidth(55)
        params_row.addWidget(self._norm_chk)
        params_row.addWidget(self._show_3db_chk)
        params_row.addStretch()
        params_row.addWidget(lbl_fs)
        params_row.addWidget(self._fs_edit)
        params_row.addWidget(lbl_casc)
        params_row.addWidget(self._casc_spin)
        ll.addLayout(params_row)

        # Name row
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit(filt._label)
        name_row.addWidget(self._name_edit, stretch=1)
        ll.addLayout(name_row)

        # Zero-phase checkbox
        self._zp_chk = QCheckBox("Zero-phase filtering (filtfilt)")
        self._zp_chk.setChecked(filt.zero_phase)
        ll.addWidget(self._zp_chk)

        # Filter Coefficients display (read-only, updated by Create Filter)
        grp_coef = QGroupBox("Filter Coefficients")
        coef_l = QFormLayout(grp_coef)
        self._num_disp = QLineEdit(self._arr2str(self._b))
        self._den_disp = QLineEdit(self._arr2str(self._a))
        self._num_disp.setReadOnly(True)
        self._den_disp.setReadOnly(True)
        self._num_disp.setStyleSheet("background:#f0f0f8;")
        self._den_disp.setStyleSheet("background:#f0f0f8;")
        coef_l.addRow("Num:", self._num_disp)
        coef_l.addRow("Den:", self._den_disp)
        ll.addWidget(grp_coef)

        # Filter Designer group
        grp_design = QGroupBox("Filter Designer")
        design_l = QVBoxLayout(grp_design)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(DESIGN_MODES)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_change)
        design_l.addWidget(self._mode_combo)

        # Mode description label
        self._mode_desc = QLabel("")
        self._mode_desc.setWordWrap(True)
        self._mode_desc.setStyleSheet("color:#4a5270; font-size:10px; padding:2px;")
        design_l.addWidget(self._mode_desc)

        # ── Mode panels (stacked, only one visible at a time) ─────────────────
        self._panel_preset    = self._build_preset_panel()
        self._panel_fir_imp   = self._build_fir_imp_panel()
        self._panel_fir_zero  = self._build_fir_zero_panel()
        self._panel_iir_imp   = self._build_iir_imp_panel()
        self._panel_two_pole  = self._build_two_pole_panel()
        self._panel_pz        = self._build_pz_panel()
        self._panel_integer   = self._build_integer_panel()

        self._mode_panels = [
            self._panel_preset, self._panel_fir_imp, self._panel_fir_zero,
            self._panel_iir_imp, self._panel_two_pole, self._panel_pz,
            self._panel_integer
        ]
        for p in self._mode_panels:
            design_l.addWidget(p)

        ll.addWidget(grp_design)
        ll.addStretch()

        # ── Right panel: plots ────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.setSpacing(4)
        root.addWidget(right, stretch=1)

        # top button bar
        top_plot = QHBoxLayout()
        self._zoom_chk = QCheckBox("Zoom")
        self._measure_chk = QCheckBox("Measure")
        top_plot.addStretch()
        top_plot.addWidget(self._zoom_chk)
        top_plot.addWidget(self._measure_chk)
        rl.addLayout(top_plot)

        # 2×2 figure
        self._fig = Figure(facecolor="#ffffff", tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl.addWidget(self._canvas, stretch=1)

        self._ax_mag  = self._fig.add_subplot(2, 2, 1)
        self._ax_imp  = self._fig.add_subplot(2, 2, 2)
        self._ax_phase= self._fig.add_subplot(2, 2, 3)
        self._ax_pz   = self._fig.add_subplot(2, 2, 4)

        for ax in (self._ax_mag, self._ax_imp, self._ax_phase, self._ax_pz):
            _style_ax(ax)

        # Transfer function — rendered as a proper LaTeX fraction in its own canvas
        self._tf_fig    = Figure(facecolor="#f8f8ff", figsize=(6, 0.9))
        self._tf_canvas = FigureCanvas(self._tf_fig)
        self._tf_canvas.setFixedHeight(75)
        self._tf_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tf_canvas.setStyleSheet(
            "border:1px solid #c8ccd8; border-radius:4px;")
        self._tf_ax = self._tf_fig.add_axes([0, 0, 1, 1])
        self._tf_ax.set_axis_off()
        self._tf_ax.set_facecolor("#f8f8ff")
        rl.addWidget(self._tf_canvas)

        # Mouse coordinate display
        self._coord_label = QLabel("Move mouse over a plot to see coordinates")
        self._coord_label.setAlignment(Qt.AlignCenter)
        self._coord_label.setStyleSheet("color:#4a5270; font-size:10px;")
        rl.addWidget(self._coord_label)

        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        # Stored plot data for snap-to-curve coordinate readout
        self._mag_w   = np.array([])   # frequency array
        self._mag_db  = np.array([])   # magnitude dB array
        self._phase_h = np.array([])   # complex h for phase
        self._imp_y   = np.array([])   # impulse response y values
        self._pz_zeros= np.array([], dtype=complex)
        self._pz_poles= np.array([], dtype=complex)

        # Initialise
        self._on_mode_change(0)
        self._update_coef_display()
        self._draw_plots()

    # ── panel builders ────────────────────────────────────────────────────────

    def _build_preset_panel(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.setSpacing(4)

        # Built-in presets
        l.addWidget(QLabel("Built-in preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(PRESETS.keys()))
        self._preset_combo.currentIndexChanged.connect(self._load_preset)
        l.addWidget(self._preset_combo)

        # Saved .fil filters
        l.addWidget(QLabel("Saved filter (.fil):"))
        self._saved_combo = QComboBox()
        self._saved_combo.addItem("── select a saved filter ──")
        self._saved_combo.currentIndexChanged.connect(self._load_saved_filter)
        l.addWidget(self._saved_combo)

        scan_btn = QPushButton("⟳  Scan folder for .fil files…")
        scan_btn.clicked.connect(self._scan_fil_folder)
        l.addWidget(scan_btn)

        self._fil_paths: dict = {}
        self._auto_scan_fil()   # populate on open
        return w

    def _build_fir_imp_panel(self):
        w = QWidget()
        l = QFormLayout(w); l.setContentsMargins(0,0,0,0)
        self._fir_b_edit = QLineEdit("1")
        self._fir_a_edit = QLineEdit("1")
        l.addRow("Zeros (b):", self._fir_b_edit)
        l.addRow("Pole gain (a):", self._fir_a_edit)
        note = QLabel("Enter space-separated coefficients. a must be a single scalar for FIR.")
        note.setWordWrap(True); note.setStyleSheet("color:#4a5270;font-size:10px;")
        l.addRow(note)
        return w

    def _build_fir_zero_panel(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
        grp = QGroupBox("Zero-Place")
        gl = QVBoxLayout(grp)
        form = QFormLayout()
        self._zp_form = QComboBox()
        self._zp_form.addItems(["Rect (re + im)", "Polar (r, θ°)"])
        form.addRow("Form:", self._zp_form)
        gl.addLayout(form)
        self._zp_list = QListWidget()
        self._zp_list.setMaximumHeight(100)
        gl.addWidget(self._zp_list)
        zp_btns = QHBoxLayout()
        self._zp_add = QPushButton("+"); self._zp_add.setMaximumWidth(40)
        self._zp_rem = QPushButton("-"); self._zp_rem.setMaximumWidth(40)
        self._zp_add.clicked.connect(self._add_zero)
        self._zp_rem.clicked.connect(self._rem_zero)
        zp_btns.addWidget(self._zp_add); zp_btns.addWidget(self._zp_rem)
        zp_btns.addStretch()
        gl.addLayout(zp_btns)
        form2 = QFormLayout()
        self._zp_pole_coef = QLineEdit("1")
        form2.addRow("Pole Coef:", self._zp_pole_coef)
        gl.addLayout(form2)
        l.addWidget(grp)
        return w

    def _build_iir_imp_panel(self):
        w = QWidget()
        l = QFormLayout(w); l.setContentsMargins(0,0,0,0)
        self._iir_b_edit = QLineEdit("1")
        self._iir_a_edit = QLineEdit("1")
        l.addRow("Zeros (b):", self._iir_b_edit)
        l.addRow("Poles (a):", self._iir_a_edit)
        note = QLabel("Full denominator for IIR. Enter all polynomial coefficients.")
        note.setWordWrap(True); note.setStyleSheet("color:#4a5270;font-size:10px;")
        l.addRow(note)
        return w

    def _build_two_pole_panel(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
        grp = QGroupBox("Two-Pole")
        gl = QFormLayout(grp)
        self._tp_type = QComboBox(); self._tp_type.addItems(TWO_POLE_TYPES)
        self._tp_form = QComboBox(); self._tp_form.addItems(TWO_POLE_FORMS)
        self._tp_form.currentIndexChanged.connect(self._on_tp_form_change)
        self._tp_r   = QLineEdit(".7")
        self._tp_freq = QLineEdit("25")
        self._tp_freq_lbl = QLabel("Freq (Hz)")
        gl.addRow("Type:", self._tp_type)
        gl.addRow("Form:", self._tp_form)
        gl.addRow("R:", self._tp_r)
        gl.addRow(self._tp_freq_lbl, self._tp_freq)
        self._tp_update = QPushButton("Update")
        self._tp_update.clicked.connect(self._compute_two_pole)
        gl.addRow(self._tp_update)
        l.addWidget(grp)
        return w

    def _build_pz_panel(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
        grp = QGroupBox("Pole-Zero Place")
        gl = QVBoxLayout(grp)
        # zero place
        z_lbl = QLabel("Zeros:"); z_lbl.setStyleSheet("font-weight:600;")
        gl.addWidget(z_lbl)
        self._pz_zform = QComboBox()
        self._pz_zform.addItems(["Rect (re + im)", "Polar (r, θ°)"])
        gl.addWidget(self._pz_zform)
        self._pz_zlist = QListWidget(); self._pz_zlist.setMaximumHeight(70)
        gl.addWidget(self._pz_zlist)
        z_btns = QHBoxLayout()
        self._pz_zadd = QPushButton("+"); self._pz_zadd.setMaximumWidth(38)
        self._pz_zrem = QPushButton("-"); self._pz_zrem.setMaximumWidth(38)
        self._pz_zadd.clicked.connect(self._pz_add_zero)
        self._pz_zrem.clicked.connect(self._pz_rem_zero)
        z_btns.addWidget(self._pz_zadd); z_btns.addWidget(self._pz_zrem); z_btns.addStretch()
        gl.addLayout(z_btns)
        # pole place
        p_lbl = QLabel("Poles:"); p_lbl.setStyleSheet("font-weight:600;")
        gl.addWidget(p_lbl)
        self._pz_pform = QComboBox()
        self._pz_pform.addItems(["Rect (re + im)", "Polar (r, θ°)"])
        gl.addWidget(self._pz_pform)
        self._pz_plist = QListWidget(); self._pz_plist.setMaximumHeight(70)
        gl.addWidget(self._pz_plist)
        p_btns = QHBoxLayout()
        self._pz_padd = QPushButton("+"); self._pz_padd.setMaximumWidth(38)
        self._pz_prem = QPushButton("-"); self._pz_prem.setMaximumWidth(38)
        self._pz_padd.clicked.connect(self._pz_add_pole)
        self._pz_prem.clicked.connect(self._pz_rem_pole)
        p_btns.addWidget(self._pz_padd); p_btns.addWidget(self._pz_prem); p_btns.addStretch()
        gl.addLayout(p_btns)
        l.addWidget(grp)
        return w

    def _build_integer_panel(self):
        w = QWidget()
        l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0)
        grp = QGroupBox("Integer Filter")
        gl = QFormLayout(grp)
        self._int_order = QSpinBox(); self._int_order.setRange(1, 32); self._int_order.setValue(4)
        self._int_zeros = QComboBox(); self._int_zeros.addItems(INTEGER_ZEROS)
        self._int_poles = QComboBox(); self._int_poles.addItems(INTEGER_POLES)
        gl.addRow("Filter Order:", self._int_order)
        gl.addRow("Zeros:", self._int_zeros)
        gl.addRow("Poles:", self._int_poles)
        self._int_update = QPushButton("Update")
        self._int_update.clicked.connect(self._compute_integer)
        gl.addRow(self._int_update)
        l.addWidget(grp)
        return w

    # ── mode switching ────────────────────────────────────────────────────────

    _MODE_DESC = [
        "Select from built-in preset filter library.",
        "Enter FIR filter as numerator (b) and a single gain denominator.",
        "Place zeros on the z-plane to build the FIR polynomial.",
        "Enter full IIR numerator and denominator coefficient polynomials.",
        "Design a two-pole IIR LP/HP/BP/BS filter by specifying R and frequency.",
        "Place both zeros and poles on the z-plane to design an IIR filter.",
        "Design integer-coefficient filters using exact pole/zero locations.",
    ]

    def _on_mode_change(self, idx: int):
        for i, p in enumerate(self._mode_panels):
            p.setVisible(i == idx)
        self._mode_desc.setText(self._MODE_DESC[idx] if idx < len(self._MODE_DESC) else "")

    # ── saved .fil filter loading ─────────────────────────────────────────────

    def _auto_scan_fil(self):
        """Scan common locations for .fil files on dialog open."""
        import os, glob
        search_dirs = [
            os.getcwd(),
            os.path.join(os.path.expanduser("~"), "digiscope_filters"),
            os.path.join(os.path.expanduser("~"), "Documents"),
        ]
        found = {}
        for d in search_dirs:
            if os.path.isdir(d):
                for fp in glob.glob(os.path.join(d, "*.fil")):
                    found[os.path.basename(fp)] = fp
        self._populate_fil_combo(found)

    def _scan_fil_folder(self):
        import os, glob
        from PyQt5.QtWidgets import QFileDialog as _QFD
        folder = _QFD.getExistingDirectory(self, "Select folder with .fil files")
        if not folder:
            return
        found = {}
        for fp in glob.glob(os.path.join(folder, "*.fil")):
            found[os.path.basename(fp)] = fp
        if not found:
            QMessageBox.information(self, "No Filters Found",
                "No .fil files found in:\n" + folder)
        self._populate_fil_combo(found)

    def _populate_fil_combo(self, found: dict):
        self._fil_paths.update(found)
        self._saved_combo.blockSignals(True)
        self._saved_combo.clear()
        self._saved_combo.addItem("── select a saved filter ──")
        for name in sorted(self._fil_paths.keys()):
            self._saved_combo.addItem(name)
        self._saved_combo.blockSignals(False)

    def _load_saved_filter(self, idx: int):
        if idx <= 0:
            return
        name = self._saved_combo.currentText()
        path = self._fil_paths.get(name)
        if not path:
            return
        try:
            b, a = _read_fil(path)
            self._b = b; self._a = a
            self._update_coef_display()
            self._name_edit.setText(name.replace(".fil", ""))
            self._draw_plots()
        except Exception as e:
            QMessageBox.critical(self, "Load Error",
                "Could not read " + name + ":\n" + str(e))

    # ── preset ────────────────────────────────────────────────────────────────

    def _load_preset(self, idx=None):
        name = self._preset_combo.currentText()
        b, a = PRESETS.get(name, (None, None))
        if b is not None:
            self._b = np.array(b, dtype=float)
            self._a = np.array(a, dtype=float)
            self._update_coef_display()
            self._name_edit.setText(name)

    # ── zero-place helpers ────────────────────────────────────────────────────

    def _parse_complex_input(self, form_combo) -> complex | None:
        from PyQt5.QtWidgets import QInputDialog
        if form_combo.currentIndex() == 0:  # rect
            re_s, ok1 = __import__("PyQt5.QtWidgets", fromlist=["QInputDialog"]).QInputDialog.getText(
                self, "Zero - Real Part", "Real part:")
            if not ok1: return None
            im_s, ok2 = __import__("PyQt5.QtWidgets", fromlist=["QInputDialog"]).QInputDialog.getText(
                self, "Zero - Imaginary Part", "Imaginary part:")
            if not ok2: return None
            return float(re_s) + 1j * float(im_s)
        else:  # polar
            r_s, ok1 = __import__("PyQt5.QtWidgets", fromlist=["QInputDialog"]).QInputDialog.getText(
                self, "Zero - Radius", "Radius r:")
            if not ok1: return None
            t_s, ok2 = __import__("PyQt5.QtWidgets", fromlist=["QInputDialog"]).QInputDialog.getText(
                self, "Zero - Angle", "Angle θ (degrees):")
            if not ok2: return None
            r = float(r_s); th = float(t_s) * np.pi / 180
            return r * np.cos(th) + 1j * r * np.sin(th)

    def _roots_to_poly(self, roots: list) -> np.ndarray:
        if not roots: return np.array([1.0])
        return np.poly(roots).real

    def _add_zero(self):
        z = self._parse_complex_input(self._zp_form)
        if z is None: return
        self._zero_place_list.append(z)
        if z.imag != 0: self._zero_place_list.append(np.conj(z))
        self._refresh_zp_list(self._zp_list, self._zero_place_list)
        self._b = self._roots_to_poly(self._zero_place_list)
        self._a = np.array([float(self._zp_pole_coef.text() or "1")])
        self._update_coef_display()

    def _rem_zero(self):
        row = self._zp_list.currentRow()
        if row < 0 or row >= len(self._zero_place_list): return
        self._zero_place_list.pop(row)
        self._refresh_zp_list(self._zp_list, self._zero_place_list)
        self._b = self._roots_to_poly(self._zero_place_list)
        self._update_coef_display()

    def _pz_add_zero(self):
        z = self._parse_complex_input(self._pz_zform)
        if z is None: return
        self._zero_place_list.append(z)
        if z.imag != 0: self._zero_place_list.append(np.conj(z))
        self._refresh_zp_list(self._pz_zlist, self._zero_place_list)
        self._b = self._roots_to_poly(self._zero_place_list)
        self._update_coef_display()

    def _pz_rem_zero(self):
        row = self._pz_zlist.currentRow()
        if row < 0 or row >= len(self._zero_place_list): return
        self._zero_place_list.pop(row)
        self._refresh_zp_list(self._pz_zlist, self._zero_place_list)
        self._b = self._roots_to_poly(self._zero_place_list)
        self._update_coef_display()

    def _pz_add_pole(self):
        p = self._parse_complex_input(self._pz_pform)
        if p is None: return
        self._pole_place_list.append(p)
        if p.imag != 0: self._pole_place_list.append(np.conj(p))
        self._refresh_zp_list(self._pz_plist, self._pole_place_list)
        self._a = self._roots_to_poly(self._pole_place_list)
        self._update_coef_display()

    def _pz_rem_pole(self):
        row = self._pz_plist.currentRow()
        if row < 0 or row >= len(self._pole_place_list): return
        self._pole_place_list.pop(row)
        self._refresh_zp_list(self._pz_plist, self._pole_place_list)
        self._a = self._roots_to_poly(self._pole_place_list)
        self._update_coef_display()

    def _refresh_zp_list(self, lw: QListWidget, roots: list):
        lw.clear()
        for z in roots:
            if z.imag == 0:
                lw.addItem(f"{z.real:.4g}")
            else:
                sign = "+" if z.imag >= 0 else "-"
                lw.addItem(f"{z.real:.4g} {sign} {abs(z.imag):.4g}i")

    # ── two-pole ──────────────────────────────────────────────────────────────

    def _on_tp_form_change(self, idx):
        labels = ["Theta (deg)", "Freq (Hz)", "Freq (norm)"]
        self._tp_freq_lbl.setText(labels[idx])

    def _compute_two_pole(self):
        try:
            R  = float(self._tp_r.text())
            th = float(self._tp_freq.text())
            form = self._tp_form.currentIndex()
            fs = self._fs
            if form == 1:   th = (th / fs) * 360
            elif form == 2: th = th * 360
            th_rad = th * np.pi / 180
            pole1 = R * (np.cos(th_rad) + 1j * np.sin(th_rad))
            pole2 = np.conj(pole1)
            tp_type = self._tp_type.currentIndex()
            if tp_type == 0:   b = np.array([1, 1])        # LP
            elif tp_type == 1: b = np.array([1, -1])       # HP
            elif tp_type == 2: b = np.array([1, 0, -1])    # BP
            else:              b = np.poly([np.exp(1j*th_rad), np.exp(-1j*th_rad)]).real  # BS
            self._b = b
            self._a = np.poly([pole1, pole2]).real
            self._update_coef_display()
        except Exception as e:
            QMessageBox.warning(self, "Two-Pole Error", str(e))

    # ── integer filter ────────────────────────────────────────────────────────

    def _compute_integer(self):
        order = self._int_order.value()
        z_mode = self._int_zeros.currentIndex()   # 0=pos, 1=neg
        p_mode = self._int_poles.currentIndex()

        b = np.zeros(order + 1); b[0] = 1
        b[-1] = 1 if z_mode == 0 else -1

        pole_map = {
            0: [1, -1],        # 1 - z^-1  → pole at 0°
            1: [1,  1],        # 1 + z^-1  → pole at 180°
            2: [1, -1, 1],     # ±60°
            3: [1,  0, 1],     # ±90°
            4: [1,  1, 1],     # ±120°
            5: [1],            # no poles (FIR)
        }
        self._b = b
        self._a = np.array(pole_map.get(p_mode, [1]), dtype=float)
        self._update_coef_display()

    # ── Create Filter button ──────────────────────────────────────────────────

    def _create_filter(self):
        mode = self._mode_combo.currentIndex()
        try:
            if mode == 0:    # preset
                self._load_preset()
            elif mode == 1:  # FIR impulse
                self._b = _parse(self._fir_b_edit.text())
                self._a = _parse(self._fir_a_edit.text())
            elif mode == 2:  # FIR zero-place — already updated on add/rem
                pass
            elif mode == 3:  # IIR impulse
                self._b = _parse(self._iir_b_edit.text())
                self._a = _parse(self._iir_a_edit.text())
            elif mode == 4:  # two-pole
                self._compute_two_pole()
            elif mode == 5:  # IIR pz
                pass
            elif mode == 6:  # integer
                self._compute_integer()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e)); return

        # Apply cascades
        n = self._casc_spin.value()
        b, a = self._b.copy(), self._a.copy()
        for _ in range(n):
            b = np.convolve(b, self._b)
            a = np.convolve(a, self._a)
        if n > 0:
            self._b, self._a = b, a

        self._update_coef_display()
        self._draw_plots()

    def _on_fs_changed(self):
        try:
            self._fs = float(self._fs_edit.text())
        except ValueError:
            pass
        self._draw_plots()

    def _update_coef_display(self):
        self._num_disp.setText(self._arr2str(self._b))
        self._den_disp.setText(self._arr2str(self._a))

    @staticmethod
    def _arr2str(a: np.ndarray) -> str:
        return "  ".join(f"{v:.4g}" for v in a)

    # ── save filter ───────────────────────────────────────────────────────────

    def _save_filter(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Filter", "", "DigiScope Filter (*.fil)")
        if not path: return
        b, a = self._b, self._a
        is_iir = len(a) > 1
        try:
            with open(path, "w") as f:
                f.write(f"{len(b)}\n")
                f.write(f"{1 if is_iir else 0}\n")
                f.write("0\n")
                f.write("\t".join(f"{v:.8f}" for v in b) + "\n")
                if is_iir:
                    f.write(f"{len(a)}\n")
                    f.write("\t".join(f"{v:.8f}" for v in a) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ── plots ─────────────────────────────────────────────────────────────────

    def _draw_plots(self):
        b, a = _normalise(self._b.copy(), self._a.copy())
        fs = self._fs
        n_fft = 1024

        # ── Magnitude ────────────────────────────────────────────────────────
        ax = self._ax_mag; ax.clear(); _style_ax(ax)
        try:
            w, h = freqz(b, a, worN=n_fft, fs=fs)
            mag = np.abs(h)
            # Use the 95th-percentile magnitude as the normalisation reference
            # (robust to integrating filters whose DC/peak value is huge)
            peak_val = float(np.percentile(mag[mag > 0], 95)) if np.any(mag > 0) else 1.0
            peak_val = max(peak_val, np.max(mag) * 0.01, 1e-12)
            peak_db  = 20 * np.log10(peak_val)

            do_norm = self._norm_chk.isChecked()
            if do_norm:
                mag_db = 20 * np.log10(mag / peak_val + 1e-12)
            else:
                mag_db = 20 * np.log10(mag + 1e-12)

            # Store for snap-to-curve
            self._mag_w   = w
            self._mag_db  = mag_db
            self._phase_h = h
            ax.plot(w, mag_db, color="#1565C0", linewidth=1.2)

            # -3 dB reference line
            if self._show_3db_chk.isChecked():
                ax.axhline(-3, color="#C62828", linewidth=0.8,
                           linestyle="--", label="-3 dB")
                ax.legend(fontsize=6, loc="lower left")

            ax.set_xlabel("Frequency (Hz)", fontsize=7)
            ax.set_ylabel("Magnitude Response (dB)", fontsize=7)

            # Gain annotation: always show peak gain; label changes with norm
            if do_norm:
                gain_label = f"Peak gain = {peak_db:+.2f} dB  (plot normalised to 0 dB)"
            else:
                gain_label = f"Peak gain = {peak_db:+.2f} dB"
            ax.set_title(gain_label, fontsize=7, color="#1565C0", pad=3)
        except Exception as e:
            ax.set_title(f"Mag error: {e}", fontsize=7)

        # ── Impulse response ──────────────────────────────────────────────────
        ax = self._ax_imp; ax.clear(); _style_ax(ax)
        try:
            n_imp = min(max(len(b) * 5, 32), 300)
            imp   = np.zeros(n_imp); imp[0] = 1.0
            h_imp = lfilter(b, a, imp)
            self._imp_y = h_imp   # store for snap
            ax.stem(np.arange(n_imp), h_imp,
                    linefmt="#1565C0", markerfmt="C0o", basefmt="#888888")
            ax.set_xlabel("Samples", fontsize=7)
            ax.set_ylabel("Impulse Response", fontsize=7)
            ax.set_title("", fontsize=8)
        except Exception as e:
            ax.set_title(f"Impulse error: {e}", fontsize=7)

        # ── Phase response ────────────────────────────────────────────────────
        ax = self._ax_phase; ax.clear(); _style_ax(ax)
        try:
            phase = np.angle(h)
            ax.plot(w, phase, color="#1565C0", linewidth=1.2)
            ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
            ax.set_xlabel("Frequency (Hz)", fontsize=7)
            ax.set_ylabel("Phase Response", fontsize=7)
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda v, _: f"{v/np.pi:.1f}π" if v != 0 else "0"))
        except Exception as e:
            ax.set_title(f"Phase error: {e}", fontsize=7)

        # ── Pole-zero ─────────────────────────────────────────────────────────
        ax = self._ax_pz; ax.clear(); _style_ax(ax)
        try:
            zeros, poles, gain = tf2zpk(b, a)
            self._pz_zeros = zeros
            self._pz_poles = poles
            theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(np.cos(theta), np.sin(theta),
                    color="#aaaacc", linewidth=0.8, linestyle="--")
            ax.axhline(0, color="#ccccdd", linewidth=0.5)
            ax.axvline(0, color="#ccccdd", linewidth=0.5)
            if len(zeros):
                ax.scatter(zeros.real, zeros.imag, marker="o", s=60,
                           facecolors="none", edgecolors="#1565C0",
                           linewidth=1.5, zorder=5, label=f"Zeros ({len(zeros)})")
            if len(poles):
                ax.scatter(poles.real, poles.imag, marker="x", s=70,
                           color="#C62828", linewidth=2, zorder=5,
                           label=f"Poles ({len(poles)})")
            ax.set_aspect("equal")
            all_pts = np.concatenate([zeros, poles, [1+0j, -1+0j]])
            lim = max(1.3, np.max(np.abs(all_pts)) * 1.2)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("Real Part", fontsize=7)
            ax.set_ylabel("Imaginary Part", fontsize=7)
            ax.set_title("Pole-Zero Plot", fontsize=8)
            if len(zeros) or len(poles):
                ax.legend(fontsize=6)
        except Exception as e:
            ax.set_title(f"PZ error: {e}", fontsize=7)

        # ── Transfer function — rendered as LaTeX fraction ────────────────────
        try:
            self._render_tf(b, a)
        except Exception:
            pass

        try:
            self._fig.tight_layout(pad=1.2)
        except Exception:
            pass
        self._canvas.draw()

    # ── Transfer function renderer ───────────────────────────────────────────

    def _render_tf(self, b: np.ndarray, a: np.ndarray):
        """
        Render H(z) as a proper typeset fraction in the TF canvas.

        Uses matplotlib mathtext:  $H(z) = dfrac{numerator}{denominator}$

        The fraction bar, super/subscripts, and spacing are all handled by
        matplotlib's built-in math text engine — no external LaTeX required.
        """
        ax = self._tf_ax
        ax.clear()
        ax.set_axis_off()
        ax.set_facecolor("#f8f8ff")

        try:
            num_str = _poly2mathtext(b)
            den_str = _poly2mathtext(a)

            is_fir = (len(a) == 1 and abs(a[0] - 1.0) < 1e-9)

            if is_fir:
                latex = r"$H(z) = " + num_str + r"$"
            else:
                # \dfrac gives a full-size (display-style) fraction
                latex = r"$H(z) = \dfrac{" + num_str + r"}{" + den_str + r"}$"

            ax.text(
                0.5, 0.5, latex,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=13,
                color="#1a1a2e",
                usetex=False,           # use matplotlib mathtext, not system LaTeX
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"H(z) — render error: {e}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888888")

        try:
            self._tf_fig.tight_layout(pad=0)
        except Exception:
            pass
        self._tf_canvas.draw()

    # ── mouse coordinate readout — snaps to nearest curve point ─────────────

    def _on_mouse_move(self, event):
        if event.inaxes is None:
            self._coord_label.setText("Move mouse over a plot to see coordinates")
            return
        ax = event.inaxes
        titles = {
            id(self._ax_mag):   "Magnitude",
            id(self._ax_imp):   "Impulse",
            id(self._ax_phase): "Phase",
            id(self._ax_pz):    "Pole-Zero",
        }
        panel = titles.get(id(ax), "")
        mx, my = event.xdata, event.ydata
        if mx is None or my is None:
            return

        if panel == "Magnitude" and len(self._mag_w):
            # Snap to nearest frequency sample on the magnitude curve
            idx = int(np.argmin(np.abs(self._mag_w - mx)))
            sx, sy = self._mag_w[idx], self._mag_db[idx]
            # Find -3dB crossings if normalised
            crossings = ""
            if self._norm_chk.isChecked() and len(self._mag_db):
                cross_idx = np.where(np.diff(np.sign(self._mag_db + 3)))[0]
                if len(cross_idx):
                    freqs = [f"{self._mag_w[i]:.2f}" for i in cross_idx]
                    crossings = f"    −3dB @ {', '.join(freqs)} Hz"
            self._coord_label.setText(
                f"Magnitude  ─  Freq = {sx:.3f} Hz    "
                f"Magnitude = {sy:.3f} dB{crossings}")

        elif panel == "Impulse" and len(self._imp_y):
            # Snap to nearest integer sample
            idx = int(np.clip(round(mx), 0, len(self._imp_y) - 1))
            sy  = self._imp_y[idx]
            self._coord_label.setText(
                f"Impulse  ─  Sample n = {idx}    h[n] = {sy:.6g}")

        elif panel == "Phase" and len(self._mag_w):
            # Snap to nearest frequency on phase curve
            idx = int(np.argmin(np.abs(self._mag_w - mx)))
            sx  = self._mag_w[idx]
            sy  = float(np.angle(self._phase_h[idx]))
            self._coord_label.setText(
                f"Phase  ─  Freq = {sx:.3f} Hz    "
                f"Phase = {sy:.4f} rad  ({sy*180/np.pi:.2f}°)")

        elif panel == "Pole-Zero":
            # Snap to nearest zero or pole
            best_label = ""
            best_dist  = float("inf")
            for i, z in enumerate(self._pz_zeros):
                d = np.sqrt((z.real - mx)**2 + (z.imag - my)**2)
                if d < best_dist:
                    best_dist = d
                    best_label = (f"Zero {i}  →  {z.real:.4f} + {z.imag:.4f}j    "
                                  f"|z|={abs(z):.4f}  ∠={np.degrees(np.angle(z)):.2f}°")
            for i, p in enumerate(self._pz_poles):
                d = np.sqrt((p.real - mx)**2 + (p.imag - my)**2)
                if d < best_dist:
                    best_dist = d
                    best_label = (f"Pole {i}  →  {p.real:.4f} + {p.imag:.4f}j    "
                                  f"|z|={abs(p):.4f}  ∠={np.degrees(np.angle(p)):.2f}°")
            if best_label:
                self._coord_label.setText(f"Pole-Zero  ─  {best_label}")
            else:
                self._coord_label.setText(
                    f"Pole-Zero  ─  Re={mx:.4f}  Im={my:.4f}  "
                    f"|z|={np.sqrt(mx**2+my**2):.4f}")
        else:
            self._coord_label.setText(f"({mx:.4g}, {my:.4g})")

    # ── public accessors ──────────────────────────────────────────────────────

    def b(self) -> list:
        """Return b coefficients, normalised to unity peak gain if checked."""
        b, a = _normalise(self._b.copy(), self._a.copy())
        if self._norm_chk.isChecked():
            try:
                from scipy.signal import freqz as _freqz
                _, h = _freqz(b, a, worN=1024)
                peak = float(np.max(np.abs(h)))
                if peak > 1e-12:
                    b = b / peak
            except Exception:
                pass
        return list(b)

    def a(self) -> list:
        return list(self._a)

    def zero_phase(self) -> bool:
        return self._zp_chk.isChecked()

    def filter_label(self) -> str:
        return self._name_edit.text() or "Custom"


# ── CustomPythonFilter ────────────────────────────────────────────────────────

class CustomPythonFilter(BaseFilter):
    name = "Custom Python Filter"
    passthrough = True
    num_plots = 1

    def __init__(self):
        super().__init__()
        self.code: str = "# Example: 5-point moving average\ny = np.convolve(x, np.ones(5)/5, mode='same')"

    def configure(self, parent=None) -> bool:
        dlg = _PythonDialog(self, parent)
        if dlg.exec_() == QDialog.Accepted:
            self.code = dlg.code()
            return True
        return False

    def calculate(self, signal: Signal) -> FilterResult:
        x = signal.data.copy(); fs = signal.rate; ns = len(x)
        ns_locals = {"x": x, "fs": fs, "ns": ns, "np": np}
        try:
            exec(self.code, ns_locals)  # noqa: S102
            y = np.asarray(ns_locals.get("y", x), dtype=float)
        except Exception as e:
            return FilterResult(data=x, t=np.arange(ns)/fs, rate=fs,
                                output_text=f"Python filter error: {e}",
                                passthrough=True)
        t = np.arange(len(y)) / fs
        return FilterResult(data=y, t=t, rate=fs,
                            output_text="Custom Python filter: ok",
                            passthrough=True)

    def plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear()
        from ui.app import _style_ax
        _style_ax(ax)
        ax.plot(result.t, result.data, color=colour, linewidth=0.9)
        ax.set_title("Custom Python Filter", fontsize=9)
        ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.4)


class _PythonDialog(QDialog):
    def __init__(self, filt: CustomPythonFilter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Python Filter")
        self.resize(560, 340)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Variables:  x = input array,  fs = sample rate,  ns = num samples\n"
            "Store result in y.  NumPy available as np.")
        note.setStyleSheet("color:#4a5270; font-size:11px;")
        layout.addWidget(note)
        self._editor = QTextEdit()
        self._editor.setFont(QFont("Consolas", 11))
        self._editor.setPlainText(filt.code)
        layout.addWidget(self._editor)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def code(self) -> str:
        return self._editor.toPlainText()
