"""
realtime/rt_window.py  –  Real-Time DigiScope
"""

from __future__ import annotations
import sys, os, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QLabel, QLineEdit, QTextEdit, QComboBox, QGroupBox,
    QSizePolicy, QAbstractItemView, QMessageBox, QDialog,
    QFormLayout, QDialogButtonBox, QDoubleSpinBox,
    QSpinBox, QAction, QTabWidget, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams.update({
    "path.simplify": True, "path.simplify_threshold": 0.2,
    "figure.dpi": 96, "figure.autolayout": False,
    "lines.antialiased": True,
})
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.signal import Signal
from core.filter_chain import FilterChain
from filters import FILTER_REGISTRY
from filters.source import SourceFilter
from realtime.pico_driver import PicoDriver, list_pico_ports
from realtime.rt_buffer import RTBuffer
from realtime.rt_filter_engine import RTFilterEngine, _HISTORY_FILTER_CLASSES

DISPLAY_S    = 5.0
DEFAULT_FONT = 14      # 2pt larger than original 12

# ── Style ─────────────────────────────────────────────────────────────────────
def _make_style(font_size: int) -> str:
    return f"""
QMainWindow,QWidget{{background:#f5f6fa;color:#1a1a2e;
  font-family:'Segoe UI','Helvetica Neue',sans-serif;font-size:{font_size}px;}}
QGroupBox{{border:1px solid #c8ccd8;border-radius:5px;margin-top:12px;
  padding-top:10px;color:#4a5270;font-size:{font_size-1}px;font-weight:600;}}
QGroupBox::title{{subcontrol-origin:margin;left:8px;}}
QListWidget{{background:#fff;border:1px solid #c8ccd8;border-radius:4px;
  color:#1a1a2e;selection-background-color:#1565C0;selection-color:#fff;outline:none;}}
QListWidget::item{{padding:4px 8px;}}
QListWidget::item:hover{{background:#e8eaf6;}}
QPushButton{{background:#fff;border:1px solid #c8ccd8;border-radius:4px;
  color:#1a1a2e;padding:4px 12px;min-height:26px;font-size:{font_size}px;}}
QPushButton:hover{{background:#e8eaf6;border-color:#9fa8da;}}
QPushButton#connect{{background:#2E7D32;border-color:#1B5E20;color:#fff;font-weight:600;}}
QPushButton#connect:hover{{background:#388E3C;}}
QPushButton#disconnect{{background:#C62828;border-color:#b71c1c;color:#fff;font-weight:600;}}
QPushButton#primary{{background:#1565C0;border-color:#0d47a1;color:#fff;font-weight:600;}}
QPushButton#danger{{background:#C62828;border-color:#b71c1c;color:#fff;}}
QPushButton#green{{background:#2E7D32;border-color:#1B5E20;color:#fff;}}
QLineEdit,QComboBox,QDoubleSpinBox,QSpinBox{{background:#fff;
  border:1px solid #c8ccd8;border-radius:4px;color:#1a1a2e;
  padding:3px 6px;font-size:{font_size}px;}}
QTextEdit{{background:#fafafa;border:1px solid #c8ccd8;border-radius:4px;
  color:#1a1a2e;font-family:'Consolas',monospace;font-size:{font_size-1}px;}}
QLabel{{color:#4a5270;font-size:{font_size}px;}}
QSplitter::handle{{background:#c8ccd8;width:2px;height:2px;}}
QMenuBar{{background:#f5f6fa;color:#1a1a2e;border-bottom:1px solid #c8ccd8;
  font-size:{font_size}px;}}
QMenuBar::item:selected{{background:#e8eaf6;}}
QSlider::groove:horizontal{{height:4px;background:#c8ccd8;border-radius:2px;}}
QSlider::handle:horizontal{{background:#1565C0;border-radius:6px;
  width:14px;height:14px;margin:-5px 0;}}
"""

def _style_ax(ax, title=""):
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#333355", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#c8ccd8")
    ax.xaxis.label.set_color("#4a5270")
    ax.yaxis.label.set_color("#4a5270")
    ax.title.set_color("#1a1a2e")
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    if title:
        ax.set_title(title, fontsize=10)


# Filters whose plot() methods draw beat markers / threshold lines / annotations
# rather than just a plain line. These need full plot() calls in RT mode.
_CUSTOM_PLOT_FILTERS = ("BeatDetectorFilter", "PanTompkinsFilter")

def _has_custom_plot(filt, result) -> bool:
    """Return True if this filter has a meaningful custom plot() with annotations."""
    cls_name = type(filt).__name__
    if cls_name in _CUSTOM_PLOT_FILTERS:
        return True
    # Also detect any filter whose result extras contain beat markers
    extras = getattr(result, "extras", {})
    return "qrs" in extras or "beats" in extras


class RTWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DigiScope RT  –  Real-Time ECG / ADC Viewer")
        self.resize(1360, 820)

        self._driver     : PicoDriver | None = None
        self._buffer      = RTBuffer(capacity_s=120.0, rate_hz=360.0)
        self._chain       = FilterChain()
        self._engine      = RTFilterEngine()
        self._engine.results_ready.connect(self._on_engine_ready)
        # output_text connected after _build_ui creates _info
        self._engine.start()
        self._rate_hz     = 360
        self._display_s   = DISPLAY_S
        self._connected   = False
        self._paused      = False
        self._font_size   = DEFAULT_FONT
        self._engine_dirty  = False   # True when engine has new data to display
        self._rt_scroll_off = 0.0    # seconds from end of buffer (0 = latest)

        # Redraw timer ~20 fps
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(50)
        self._redraw_timer.timeout.connect(self._update_plots)

        self._build_menu()
        self._build_ui()
        # Now _info exists, safe to connect output_text
        self._engine.output_text.connect(self._info.setPlainText)
        self._apply_font(DEFAULT_FONT)

    # ── Menu ─────────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        self._act(fm, "Save Buffer as CSV…",  self._save_csv)
        self._act(fm, "Save Buffer as TXT…",  self._save_txt)
        fm.addSeparator()
        self._act(fm, "Exit", self.close)
        vm = mb.addMenu("View")
        self._act(vm, "Increase Font Size  (Ctrl++)", self._font_bigger)
        self._act(vm, "Decrease Font Size  (Ctrl+-)", self._font_smaller)
        self._act(vm, "Reset Font Size",               self._font_reset)

    def _act(self, menu, label, slot):
        a = QAction(label, self)
        a.triggered.connect(slot)
        menu.addAction(a)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(285)
        ll   = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(6)
        splitter.addWidget(left)

        # Status label created FIRST so _refresh_ports can write to it
        self._status_lbl = QLabel("Scanning ports…")
        self._status_lbl.setStyleSheet(
            "color:#4a5270; font-weight:600;")
        self._status_lbl.setWordWrap(True)

        # Connection group
        grp_conn = QGroupBox("Pico Connection")
        cl = QFormLayout(grp_conn)

        self._port_combo = QComboBox()
        self._port_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        ref_btn = QPushButton("⟳")
        ref_btn.setFixedWidth(32)
        ref_btn.setToolTip("Rescan serial ports")
        ref_btn.clicked.connect(self._refresh_ports)

        port_row = QHBoxLayout()
        port_row.addWidget(self._port_combo, stretch=1)
        port_row.addWidget(ref_btn)
        cl.addRow("Port:", port_row)

        self._rate_spin = QSpinBox()
        self._rate_spin.setRange(1, 50000)
        self._rate_spin.setValue(self._rate_hz)
        self._rate_spin.setSuffix(" Hz")
        self._rate_spin.valueChanged.connect(self._on_rate_changed)
        cl.addRow("Pico Sample Rate:", self._rate_spin)

        self._vref_spin = QDoubleSpinBox()
        self._vref_spin.setRange(0.1, 5.0); self._vref_spin.setValue(3.3)
        self._vref_spin.setSingleStep(0.1); self._vref_spin.setSuffix(" V")
        self._vref_spin.valueChanged.connect(self._on_vref_changed)
        cl.addRow("ADC Vref:", self._vref_spin)

        self._bias_spin = QDoubleSpinBox()
        self._bias_spin.setRange(-5.0, 5.0); self._bias_spin.setValue(0.0)
        self._bias_spin.setSingleStep(0.01); self._bias_spin.setSuffix(" V")
        self._bias_spin.setToolTip(
            "Subtract this DC offset. Use 1.65 V if signal is biased to 0–3.3 V.")
        self._bias_spin.valueChanged.connect(self._on_bias_changed)
        cl.addRow("DC Bias:", self._bias_spin)

        self._disp_spin = QDoubleSpinBox()
        self._disp_spin.setRange(0.5, 60.0); self._disp_spin.setValue(self._display_s)
        self._disp_spin.setSingleStep(0.5); self._disp_spin.setSuffix(" s")
        self._disp_spin.valueChanged.connect(lambda v: setattr(self, "_display_s", v))
        cl.addRow("Display Window:", self._disp_spin)

        btn_row = QHBoxLayout()
        self._conn_btn  = QPushButton("▶  Connect")
        self._conn_btn.setObjectName("connect")
        self._conn_btn.clicked.connect(self._toggle_connect)
        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.clicked.connect(self._toggle_pause)
        self._pause_btn.setEnabled(False)
        btn_row.addWidget(self._conn_btn); btn_row.addWidget(self._pause_btn)
        cl.addRow(btn_row)
        ll.addWidget(grp_conn)
        ll.addWidget(self._status_lbl)

        # Font size — plain textbox
        grp_font = QGroupBox("Font Size")
        fl = QHBoxLayout(grp_font)
        fl.addWidget(QLabel("Size:"))
        self._font_edit = QLineEdit(str(DEFAULT_FONT))
        self._font_edit.setFixedWidth(42)
        self._font_edit.setAlignment(Qt.AlignCenter)
        self._font_edit.setToolTip("Font size in pt — press Enter to apply")
        self._font_edit.returnPressed.connect(self._on_font_edit)
        fl.addWidget(self._font_edit)
        fl.addWidget(QLabel("pt"))
        fl.addStretch()
        ll.addWidget(grp_font)

        # Filter chain
        grp_filt = QGroupBox("Filter Chain  (applied to filtered plots)")
        flv = QVBoxLayout(grp_filt)
        self._flist = QListWidget()
        self._flist.setMaximumHeight(170)
        self._flist.setSelectionMode(QAbstractItemView.SingleSelection)
        self._flist.itemSelectionChanged.connect(self._on_rt_sel)
        flv.addWidget(self._flist)

        # Row 1: Add / Del / Config / Apply
        fb1 = QHBoxLayout()
        self._b_add = QPushButton("+ Add");    self._b_add.setObjectName("green")
        self._b_del = QPushButton("✕ Del");    self._b_del.setObjectName("danger")
        self._b_cfg = QPushButton("⚙ Cfg")
        self._b_run = QPushButton("▶ Apply");  self._b_run.setObjectName("primary")
        self._b_add.clicked.connect(self._add_filter)
        self._b_del.clicked.connect(self._del_filter)
        self._b_cfg.clicked.connect(self._cfg_filter)
        self._b_run.clicked.connect(self._rebuild_plots)
        for b in (self._b_add, self._b_del, self._b_cfg, self._b_run):
            fb1.addWidget(b)
        flv.addLayout(fb1)

        # Row 2: Move up / down / show toggle
        fb2 = QHBoxLayout()
        self._b_up   = QPushButton("▲ Up")
        self._b_dn   = QPushButton("▼ Dn")
        self._b_up.clicked.connect(self._move_filter_up)
        self._b_dn.clicked.connect(self._move_filter_dn)
        self._chk_vis = QCheckBox("Show")
        self._chk_vis.setChecked(True)
        self._chk_vis.stateChanged.connect(self._toggle_vis)
        fb2.addWidget(self._b_up); fb2.addWidget(self._b_dn)
        fb2.addStretch(); fb2.addWidget(self._chk_vis)
        flv.addLayout(fb2)
        ll.addWidget(grp_filt)

        # Pause-scroll controls
        grp_scroll = QGroupBox("History Scroll  (while paused)")
        sl = QHBoxLayout(grp_scroll)
        self._b_back = QPushButton("◀◀")
        self._b_back_s = QPushButton("◀")
        self._b_fwd_s  = QPushButton("▶")
        self._b_fwd    = QPushButton("▶▶")
        self._b_latest = QPushButton("Latest")
        for b in (self._b_back, self._b_back_s,
                  self._b_fwd_s, self._b_fwd, self._b_latest):
            b.setFixedWidth(44)
            b.setEnabled(False)
            sl.addWidget(b)
        self._b_back.clicked.connect(lambda: self._rt_scroll(-1.0))
        self._b_back_s.clicked.connect(lambda: self._rt_scroll(-0.1))
        self._b_fwd_s.clicked.connect(lambda: self._rt_scroll(0.1))
        self._b_fwd.clicked.connect(lambda: self._rt_scroll(1.0))
        self._b_latest.clicked.connect(self._rt_scroll_latest)
        ll.addWidget(grp_scroll)

        grp_out = QGroupBox("Output")
        ol = QVBoxLayout(grp_out)
        self._info = QTextEdit(); self._info.setReadOnly(True)
        self._info.setMaximumHeight(100)
        ol.addWidget(self._info)
        ll.addWidget(grp_out)
        ll.addStretch()

        # ── Right panel: stats bar + canvas ───────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        stats = QHBoxLayout()
        self._lbl_rate    = QLabel("Rate: —")
        self._lbl_bpm     = QLabel("BPM: —")
        self._lbl_samples = QLabel("Samples: 0")
        self._lbl_buf     = QLabel("Buffer: 0.0 s")
        for lbl in (self._lbl_rate, self._lbl_bpm,
                    self._lbl_samples, self._lbl_buf):
            lbl.setStyleSheet("color:#1565C0; font-weight:700;")
            stats.addWidget(lbl)
        stats.addStretch()
        rl.addLayout(stats)

        self._fig = Figure(facecolor="#ffffff")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl.addWidget(self._canvas, stretch=1)

        # Build the initial 2-subplot layout (raw + passthrough)
        self._plot_axes  : list = []   # all current axes
        self._plot_lines : list = []   # (ax, filter_index, filter_obj)
        self._build_subplots()

        # Scan ports now that all widgets exist
        self._refresh_ports()

    # ── Subplot management ────────────────────────────────────────────────────
    def _build_subplots(self):
        """
        Rebuild the figure subplots to match:
          row 0       – raw ADC signal  (always present)
          rows 1..N   – one row per visible filter in the chain
                        (same as main DigiScope app)
        """
        self._fig.clear()
        self._plot_axes  = []
        self._plot_lines = []

        # Determine how many rows we need
        filter_rows = [(i, f) for i, f in enumerate(self._chain.filters)
                       if getattr(f, "visible", True)]
        n_rows = 1 + len(filter_rows)

        colours = ["#1565C0", "#C62828", "#2E7D32", "#6A0DAD",
                   "#E65100", "#00838F", "#AD1457"]

        for row in range(n_rows):
            ax = self._fig.add_subplot(n_rows, 1, row + 1)
            _style_ax(ax)
            self._plot_axes.append(ax)

            if row == 0:
                ax.set_title("Raw ADC Signal", fontsize=10)
                ax.set_ylabel("Voltage (V)", fontsize=9)
                self._plot_lines.append((ax, -1, None))   # -1 = raw
            else:
                fi, filt = filter_rows[row - 1]
                colour = getattr(filt, "colour", colours[row % len(colours)])
                ax.set_title(filt.name, fontsize=10)
                ax.set_ylabel("Amplitude", fontsize=9)
                self._plot_lines.append((ax, fi, filt))

            if row == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=9)

        try:
            self._fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self._canvas.draw()

    def _rebuild_plots(self):
        """Called when filter chain changes — rebuilds subplots and reinstalls engine."""
        self._build_subplots()
        self._engine.set_chain(self._chain, float(self._rate_hz))
        self._info.setPlainText("")

    # ── Helpers ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _slice_result_static(result, n_show: int):
        """
        Return a FilterResult showing only the last n_show samples,
        with time axis re-zeroed to start at 0 and all index-based
        extras (qrs, qrs_real, thresh1, thresh2) correctly offset.
        """
        from core.filter_base import FilterResult
        full_n  = len(result.data)
        n_show  = min(n_show, full_n)
        offset  = full_n - n_show

        d = result.data[offset:].copy()
        # Re-zero time axis so display always starts at t=0
        rate = result.rate or 360.0
        t = np.arange(n_show) / rate

        extras = dict(result.extras)

        # Re-index all sample-index arrays
        for key in ("qrs", "qrs_real"):
            if key in extras:
                arr = np.asarray(extras[key])
                mask = arr >= offset
                extras[key] = arr[mask] - offset

        # Slice sample-aligned arrays (same length as data)
        for key in ("thresh1", "thresh2"):
            if key in extras and len(extras[key]) == full_n:
                extras[key] = extras[key][offset:].copy()

        return FilterResult(
            data=d, t=t, rate=rate,
            output_text=result.output_text,
            passthrough=result.passthrough,
            extras=extras)

    # ── Live plotting ─────────────────────────────────────────────────────────
    def _update_plots(self):
        # When paused, allow scrolling through buffered history
        if self._paused and self._rt_scroll_off > 0:
            full = self._buffer.snapshot(
                self._buffer.duration_buffered)
            n_show = int(self._display_s * self._buffer.rate)
            n_off  = int(self._rt_scroll_off * self._buffer.rate)
            end    = max(n_show, len(full) - n_off)
            raw    = full[max(0, end - n_show):end]
        else:
            raw = self._buffer.snapshot(self._display_s)

        if len(raw) < 4:
            return

        rate = self._buffer.rate
        t_raw = np.arange(len(raw)) / rate

        # Run history-dependent filters (beat detectors, PT) on full buffer.
        # All other filters already ran in the engine thread — read their buffers.
        history_results = {}
        if len(self._chain) > 0:
            try:
                history_results = self._engine.run_history_filters(
                    self._buffer, self._buffer.duration_buffered)
            except Exception as e:
                self._info.setPlainText(f"History filter error: {e}")

        # Update each subplot
        for ax, fi, filt in self._plot_lines:
            if ax is None:
                continue

            if fi == -1:
                # Raw ADC signal
                t, d = t_raw, raw
                ax.clear(); _style_ax(ax)
                ax.plot(t, d, color="#1565C0", lw=0.9)
                ax.set_title("Raw ADC Signal", fontsize=10)
                ax.set_ylabel("Voltage (V)", fontsize=9)
                ax.set_xlim(t[0], max(t[-1], t[0] + 0.01))
                ylo, yhi = float(d.min()), float(d.max())
                pad = max((yhi - ylo) * 0.1, 0.02)
                ax.set_ylim(ylo - pad, yhi + pad)
                continue

            if filt is None:
                continue

            cls_name = type(filt).__name__

            if cls_name in _HISTORY_FILTER_CLASSES:
                # Beat detectors / PT: use the full-history result
                result = history_results.get(fi)
                if result is None or len(result.data) == 0:
                    continue
                # Show only the last display_s seconds of the full-history output
                n_show = min(len(result.data),
                             int(self._display_s * (result.rate or rate)))
                d = result.data[-n_show:].astype(np.float32)
                t = np.arange(len(d)) / (result.rate or rate)
                if _has_custom_plot(filt, result):
                    # Slice result to display window for annotations
                    result_slice = self._slice_result_static(result, n_show)
                    result_slice.extras["_colour"] = getattr(
                        filt, "colour", "#C62828")
                    ax.clear(); _style_ax(ax)
                    try:
                        filt.plot([ax], result_slice)
                    except Exception:
                        ax.plot(t, d, lw=0.9)
                    ax.set_xlim(t[0], max(t[-1], t[0] + 0.01))
                else:
                    colour = getattr(filt, "colour", "#C62828")
                    ax.clear(); _style_ax(ax)
                    ax.plot(t, d, color=colour, lw=0.9)
                    ax.set_title(filt.name, fontsize=10)
                    ax.set_ylabel("Amplitude", fontsize=9)
                    ax.set_xlim(t[0], max(t[-1], t[0] + 0.01))
                    ylo, yhi = float(d.min()), float(d.max())
                    pad = max((yhi - ylo) * 0.1, 0.02)
                    ax.set_ylim(ylo - pad, yhi + pad)

            else:
                # Streaming filter: read the display window from the engine buffer
                eng_buf = self._engine.get_filter_buffer(fi)
                if eng_buf is None:
                    continue
                # Apply same scroll offset as raw plot
                if self._paused and self._rt_scroll_off > 0:
                    full_e = eng_buf.snapshot(eng_buf.duration_buffered)
                    n_show = int(self._display_s * rate)
                    n_off  = int(self._rt_scroll_off * rate)
                    end_e  = max(n_show, len(full_e) - n_off)
                    d = full_e[max(0, end_e - n_show):end_e]
                else:
                    d = eng_buf.snapshot(self._display_s)
                if len(d) < 2:
                    continue
                t = np.arange(len(d)) / rate
                colour = getattr(filt, "colour", "#C62828")
                ax.clear(); _style_ax(ax)
                ax.plot(t, d, color=colour, lw=0.9)
                ax.set_title(filt.name, fontsize=10)
                ax.set_ylabel("Amplitude", fontsize=9)
                ax.set_xlim(t[0], max(t[-1], t[0] + 0.01))
                ylo, yhi = float(d.min()), float(d.max())
                pad = max((yhi - ylo) * 0.1, 0.02)
                ax.set_ylim(ylo - pad, yhi + pad)

        # Stats — use beat detector result for BPM if available, else estimate
        bpm = None
        for fi, filt in enumerate(self._chain.filters):
            if type(filt).__name__ in ("BeatDetectorFilter", "PanTompkinsFilter"):
                r = history_results.get(fi)
                if r:
                    bpm = r.extras.get("bpm")
                    break
        if bpm is None:
            bpm = self._estimate_bpm(raw, rate)
        self._lbl_bpm.setText(f"BPM: {bpm:.0f}" if bpm else "BPM: —")
        self._lbl_samples.setText(f"Samples: {self._buffer.total_samples:,}")
        self._lbl_buf.setText(f"Buffer: {self._buffer.duration_buffered:.1f} s")
        self._engine_dirty = False
        self._canvas.draw_idle()

    # ── Port scanning ─────────────────────────────────────────────────────────
    def _refresh_ports(self):
        self._port_combo.clear()
        try:
            ports = list_pico_ports()
        except Exception:
            ports = []

        if ports:
            for p, d in ports:
                self._port_combo.addItem(f"{p}  –  {d}", userData=p)
            self._set_status(
                f"{len(ports)} port(s) found — select your Pico's port", ok=True)
        else:
            self._port_combo.addItem("── No serial ports found ──", userData="")
            self._set_status(
                "No ports found.  pip install pyserial  then click ⟳", ok=False)

    # ── Connection ────────────────────────────────────────────────────────────
    def _toggle_connect(self):
        if self._connected: self._disconnect()
        else:               self._connect()

    def _connect(self):
        port = self._port_combo.currentData()
        if not port:
            QMessageBox.warning(self, "No Port Selected",
                "Select a serial port.\n\n"
                "Tip: on Windows the Pico appears as 'USB Serial Device (COMx)'\n"
                "in Device Manager — look for a COM port and try it.")
            return
        rate = self._rate_spin.value()
        vref = self._vref_spin.value()
        bias = self._bias_spin.value()

        # Do NOT reset buffer here — let update_rate handle it only if rate changed
        self._buffer.update_rate(rate)
        self._driver = PicoDriver(port, rate_hz=rate, vref=vref, dc_bias=bias)
        self._driver.samples_ready.connect(self._on_samples)
        self._driver.status_msg.connect(self._on_status)
        self._driver.error.connect(self._on_error)
        self._driver.rate_confirmed.connect(self._on_rate_confirmed)
        self._driver.start()

        self._connected = True
        self._paused    = False
        self._conn_btn.setText("■  Disconnect")
        self._conn_btn.setObjectName("disconnect")
        self._conn_btn.setStyle(self._conn_btn.style())
        self._pause_btn.setEnabled(True)
        self._engine.set_chain(self._chain, float(rate))
        self._redraw_timer.start()
        self._set_status(f"Connecting to {port} @ {rate} Hz…", ok=True)

    def _disconnect(self):
        self._redraw_timer.stop()
        if self._driver:
            self._driver.stop()
            self._driver.wait(2000)
            self._driver = None
        self._connected = False
        self._paused    = False
        self._conn_btn.setText("▶  Connect")
        self._conn_btn.setObjectName("connect")
        self._conn_btn.setStyle(self._conn_btn.style())
        self._pause_btn.setEnabled(False)
        self._engine_dirty = False
        self._set_status("Disconnected", ok=False)

    def _toggle_pause(self):
        self._paused = not self._paused
        self._pause_btn.setText("▶  Resume" if self._paused else "⏸  Pause")
        if self._driver:
            self._driver.send_command("STOP" if self._paused else "START")
        # Enable scroll buttons only while paused
        scroll_enabled = self._paused
        for b in (self._b_back, self._b_back_s,
                  self._b_fwd_s, self._b_fwd, self._b_latest):
            b.setEnabled(scroll_enabled)
        if not self._paused:
            self._rt_scroll_off = 0.0   # snap back to latest on resume

    # ── Driver callbacks ──────────────────────────────────────────────────────
    def _on_samples(self, arr: np.ndarray):
        if not self._paused:
            self._buffer.push(arr)
            self._engine.push_chunk(arr)

    def _on_engine_ready(self):
        """Engine finished processing a chunk — flag for next display frame."""
        self._engine_dirty = True

    def _on_status(self, msg: str):
        self._set_status(msg, ok=True)

    def _on_error(self, err: str):
        self._set_status(f"Error: {err}", ok=False)
        self._disconnect()
        QMessageBox.critical(self, "Pico Driver Error", err)

    def _on_rate_confirmed(self, hz: int):
        self._rate_hz = hz
        self._buffer.update_rate(hz)
        self._engine.set_rate(float(hz))
        self._lbl_rate.setText(f"Rate: {hz} Hz")
        self._rate_spin.blockSignals(True)
        self._rate_spin.setValue(hz)
        self._rate_spin.blockSignals(False)

    def _on_rate_changed(self, hz: int):
        self._rate_hz = hz
        self._buffer.update_rate(hz)
        self._engine.set_rate(float(hz))
        if self._driver:
            self._driver.set_rate(hz)
            self._update_calib_status()

    def _on_vref_changed(self, vref: float):
        """Update driver calibration scale when Vref changes while connected."""
        if self._driver:
            self._driver.vref = vref
            # Driver loop re-reads self.vref each iteration automatically
            self._update_calib_status()

    def _on_bias_changed(self, bias: float):
        """Update driver DC bias when bias changes while connected."""
        if self._driver:
            self._driver.dc_bias = bias
            self._update_calib_status()

    def _update_calib_status(self):
        """Refresh status label with current vref/bias."""
        hz   = self._rate_hz
        vref = self._vref_spin.value()
        bias = self._bias_spin.value()
        self._set_status(
            f"Streaming  {hz} Hz | Vref={vref:.2f} V | bias={bias:.2f} V",
            ok=True)

    # ── Filter chain ──────────────────────────────────────────────────────────
    def _add_filter(self):
        dlg = _AddFilterDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        cls = dlg.selected_class()
        if cls is None:
            return
        f = cls()
        if type(f).__name__ == "TemplateMatchFilter":
            # Pause and pass full buffer on first add
            was_paused = self._paused
            if not was_paused and self._connected:
                self._paused = True
                self._pause_btn.setText("▶  Resume")
                if self._driver:
                    self._driver.send_command("STOP")
            snap = self._buffer.snapshot(self._buffer.duration_buffered)
            if len(snap) < 4:
                snap = self._buffer.snapshot(self._display_s)
            f.configure(self,
                        signal_data=snap if len(snap) > 4 else None,
                        rate=float(self._rate_hz))
            if not was_paused and self._connected:
                self._paused = False
                self._pause_btn.setText("⏸  Pause")
                if self._driver:
                    self._driver.send_command("START")
        else:
            f.configure(self)
        self._chain.append(f)
        self._refresh_flist()
        self._rebuild_plots()

    def _del_filter(self):
        items = self._flist.selectedItems()
        if not items:
            return
        idx = self._flist.row(items[0])
        if 0 <= idx < len(self._chain):
            self._chain.remove(idx)
            self._refresh_flist()
            self._rebuild_plots()

    def _cfg_filter(self):
        items = self._flist.selectedItems()
        if not items:
            return
        idx = self._flist.row(items[0])
        if 0 <= idx < len(self._chain):
            f = self._chain.filters[idx]
            if type(f).__name__ == "TemplateMatchFilter":
                # Pause streaming, show full buffer for template selection
                was_paused = self._paused
                if not was_paused:
                    self._paused = True
                    self._pause_btn.setText("▶  Resume")
                    if self._driver:
                        self._driver.send_command("STOP")
                # Use full buffered history for template selection
                snap = self._buffer.snapshot(self._buffer.duration_buffered)
                if len(snap) < 4:
                    snap = self._buffer.snapshot(self._display_s)
                changed = f.configure(self,
                                      signal_data=snap if len(snap) > 4 else None,
                                      rate=float(self._rate_hz))
                if not was_paused:
                    self._paused = False
                    self._pause_btn.setText("⏸  Pause")
                    if self._driver:
                        self._driver.send_command("START")
            else:
                f.configure(self)
            self._refresh_flist()

    def _refresh_flist(self):
        self._flist.clear()
        for f in self._chain.filters:
            visible = getattr(f, "visible", True)
            prefix  = "👁 " if visible else "🚫 "
            self._flist.addItem(prefix + f.name)

    # ── Font size ─────────────────────────────────────────────────────────────
    def _apply_font(self, size: int):
        size = max(7, min(32, int(size)))
        self._font_size = size
        QApplication.instance().setStyleSheet(_make_style(size))
        if hasattr(self, "_font_edit"):
            self._font_edit.blockSignals(True)
            self._font_edit.setText(str(size))
            self._font_edit.blockSignals(False)

    def _on_font_edit(self):
        try:
            self._apply_font(int(self._font_edit.text().strip()))
        except ValueError:
            self._font_edit.setText(str(self._font_size))

    def _font_bigger(self):
        self._apply_font(self._font_size + 1)

    def _font_smaller(self):
        self._apply_font(max(7, self._font_size - 1))

    def _font_reset(self):
        self._apply_font(DEFAULT_FONT)

    # ── Stats helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _estimate_bpm(data: np.ndarray, rate: float):
        if len(data) < int(rate * 2):
            return None
        try:
            thresh = data.mean() + 0.5 * data.std()
            above  = (data > thresh).astype(int)
            edges  = np.where(np.diff(above) == 1)[0]
            if len(edges) < 2:
                return None
            rr_med = float(np.median(np.diff(edges)))
            return 60.0 * rate / rr_med if rr_med > 0 else None
        except Exception:
            return None

    # ── File I/O ──────────────────────────────────────────────────────────────
    def _save_csv(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Buffer as CSV", "", "CSV (*.csv)")
        if not path:
            return
        raw  = self._buffer.snapshot(self._buffer.duration_buffered)
        rate = self._buffer.rate
        t    = np.arange(len(raw)) / rate
        np.savetxt(path, np.column_stack([t, raw]),
                   delimiter=",", header="time_s,voltage_V", comments="")

    def _save_txt(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Buffer as TXT", "", "Text Files (*.txt)")
        if not path:
            return
        raw  = self._buffer.snapshot(self._buffer.duration_buffered)
        rate = self._buffer.rate
        t    = np.arange(len(raw)) / rate
        with open(path, "w") as f:
            f.write("# DigiScope RT buffer export\n")
            f.write(f"# Sample rate: {rate:.2f} Hz\n")
            f.write(f"# Samples: {len(raw)}\n")
            f.write("# Columns: time_s  voltage_V\n")
            for ti, vi in zip(t, raw):
                f.write(f"{ti:.6f}\t{vi:.6f}\n")

    # ── Misc ──────────────────────────────────────────────────────────────────
    def _set_status(self, msg: str, ok: bool):
        if hasattr(self, "_status_lbl"):
            colour = "#2E7D32" if ok else "#C62828"
            self._status_lbl.setStyleSheet(
                f"color:{colour}; font-weight:600;")
            self._status_lbl.setText(msg)

    # ── History scroll (while paused) ────────────────────────────────────────────
    def _rt_scroll(self, fraction: float):
        """Scroll the view by fraction * display_s seconds (negative = back in time)."""
        step = fraction * self._display_s
        buf_dur = self._buffer.duration_buffered
        # _rt_scroll_off is seconds from the END of buffer (0 = latest)
        self._rt_scroll_off = max(
            0.0,
            min(buf_dur - self._display_s, self._rt_scroll_off - step))
        self._update_plots()

    def _rt_scroll_latest(self):
        self._rt_scroll_off = 0.0
        self._update_plots()

    # ── RT filter list helpers ───────────────────────────────────────────────────
    def _sel_idx(self) -> int:
        items = self._flist.selectedItems()
        return self._flist.row(items[0]) if items else -1

    def _on_rt_sel(self):
        idx = self._sel_idx()
        if idx < 0 or idx >= len(self._chain.filters):
            return
        vis = getattr(self._chain.filters[idx], "visible", True)
        self._chk_vis.blockSignals(True)
        self._chk_vis.setChecked(vis)
        self._chk_vis.blockSignals(False)

    def _move_filter_up(self):
        idx = self._sel_idx()
        if idx <= 0 or idx >= len(self._chain):
            return
        self._chain.move_up(idx)
        self._refresh_flist()
        self._flist.setCurrentRow(idx - 1)
        self._rebuild_plots()

    def _move_filter_dn(self):
        idx = self._sel_idx()
        if idx < 0 or idx >= len(self._chain) - 1:
            return
        self._chain.move_down(idx)
        self._refresh_flist()
        self._flist.setCurrentRow(idx + 1)
        self._rebuild_plots()

    def _toggle_vis(self, state):
        idx = self._sel_idx()
        if idx < 0 or idx >= len(self._chain.filters):
            return
        # Qt stateChanged passes int: 0=unchecked, 2=checked
        self._chain.filters[idx].visible = (state == Qt.Checked or state == 2)
        self._refresh_flist()
        # Reselect the same row so checkbox stays in sync
        self._flist.setCurrentRow(idx)
        self._rebuild_plots()

    def closeEvent(self, e):
        self._disconnect()
        self._engine.stop()
        self._engine.wait(2000)
        super().closeEvent(e)


# ── Add-filter dialog (tabbed + favorites, mirrors main DigiScope) ────────────
import json as _json, os as _os

_FAV_FILE = _os.path.join(_os.path.expanduser("~"), ".digiscope_favs.json")

def _load_favs() -> set:
    try:
        with open(_FAV_FILE) as f:
            return set(_json.load(f))
    except Exception:
        return set()

def _save_favs(favs: set):
    try:
        with open(_FAV_FILE, "w") as f:
            _json.dump(sorted(favs), f)
    except Exception:
        pass

_FILTER_TABS = {
    "★ Favorites":  [],
    "Common": [
        "Custom Filter Designer", "Squaring",
        "MWI (Moving Window Integrator)", "Template Matching",
        "Resample", "PT Thresholding",
    ],
    "Detection": ["Beat Detector (threshold)", "PT Thresholding"],
    "Signal":    ["Remove Mean", "Add Noise", "ECG Filter (Butterworth/Notch)"],
    "All":       [],
}

class _AddFilterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Processing Step")
        self.setMinimumWidth(340); self.setMinimumHeight(420)

        self._favs = _load_favs()
        layout = QVBoxLayout(self)

        self._tabs  = QTabWidget()
        self._lists : dict = {}
        self._items : dict = {}

        valid = [(n, c) for n, c in FILTER_REGISTRY if "ECG Runner" not in n]
        reg   = {n: c for n, c in valid}

        for tab_name, names in _FILTER_TABS.items():
            if tab_name == "★ Favorites":
                items = [(n, c) for n, c in valid if n in self._favs]
            elif tab_name == "All":
                items = valid
            else:
                items = [(n, reg[n]) for n in names if n in reg]

            lw = QListWidget()
            lw.itemDoubleClicked.connect(self.accept)
            for name, _ in items:
                lw.addItem(("★ " if name in self._favs else "  ") + name)
            if items:
                lw.setCurrentRow(0)

            self._lists[tab_name] = lw
            self._items[tab_name] = items
            self._tabs.addTab(lw, tab_name)

        layout.addWidget(self._tabs)

        bot = QHBoxLayout()
        star = QPushButton("★ Toggle Favorite")
        star.clicked.connect(self._toggle_fav)
        bot.addWidget(star); bot.addStretch()
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        bot.addWidget(btns)
        layout.addLayout(bot)

    def _current_item(self):
        tab   = self._tabs.tabText(self._tabs.currentIndex())
        lw    = self._lists.get(tab)
        items = self._items.get(tab, [])
        if lw is None or lw.currentRow() < 0: return None
        row = lw.currentRow()
        return items[row] if row < len(items) else None

    def _toggle_fav(self):
        item = self._current_item()
        if item is None: return
        name = item[0]
        if name in self._favs: self._favs.discard(name)
        else: self._favs.add(name)
        _save_favs(self._favs)
        tab = self._tabs.tabText(self._tabs.currentIndex())
        lw  = self._lists[tab]; row = lw.currentRow()
        lw.item(row).setText(("★ " if name in self._favs else "  ") + name)

    def selected_class(self):
        item = self._current_item()
        return item[1] if item else None


# ── Entry point ───────────────────────────────────────────────────────────────
def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(_make_style(DEFAULT_FONT))
    win = RTWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
