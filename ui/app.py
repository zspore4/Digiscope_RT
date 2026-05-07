"""
ui/app.py  –  DigiScope main window.
White-theme, per-filter signal colour picker, background colour picker.
"""

from __future__ import annotations
import sys, traceback
from pathlib import Path
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QLineEdit, QTextEdit,
    QFileDialog, QMessageBox, QCheckBox, QDialog,
    QFormLayout, QDialogButtonBox, QComboBox,
    QSizePolicy, QAction, QMenu, QGroupBox,
    QAbstractItemView, QColorDialog, QProgressDialog, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPixmap, QIcon

import matplotlib
matplotlib.use("Qt5Agg")

# ── Performance tuning ────────────────────────────────────────────────────────
# These settings cut redraw time significantly on Windows/macOS:
#   • agg.path.chunksize  breaks long polylines into chunks so the renderer
#     can skip off-screen segments — big win for long ECG recordings
#   • path.simplify + simplify_threshold  merges near-identical line segments
#     before rasterization — reduces GPU/CPU work for dense signals
#   • figure.dpi 96 instead of default 100 — slightly fewer pixels to push
matplotlib.rcParams.update({
    "agg.path.chunksize":       0,      # 0 = auto-chunk
    "path.simplify":            True,
    "path.simplify_threshold":  0.15,   # merge segments < 15% of a pixel apart
    "figure.dpi":               96,
    "figure.autolayout":        False,  # we manage layout manually
    "lines.antialiased":        True,
    "axes.formatter.use_mathtext": False,
    "axes.unicode_minus":       True,
})
# ─────────────────────────────────────────────────────────────────────────────

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from core.signal import Signal
from core.filter_chain import FilterChain
from core.filter_base import FilterResult
from filters.source import SourceFilter
from filters import FILTER_REGISTRY
import utils.data_loader as loader

# ── Default signal colours assigned in order ─────────────────────────────────
_SIG_COLOURS = [
    "#1565C0",  # deep blue      (source)
    "#C62828",  # deep red
    "#2E7D32",  # deep green
    "#6A1B9A",  # purple
    "#E65100",  # deep orange
    "#00695C",  # teal
    "#4527A0",  # indigo
    "#558B2F",  # light green
    "#AD1457",  # pink
    "#0277BD",  # light blue
]

def _next_colour(idx: int) -> str:
    return _SIG_COLOURS[idx % len(_SIG_COLOURS)]

# ── Light stylesheet (font-size parameterised) ────────────────────────────────
_APP_FONT_SIZE = 12   # module-level, updated by _apply_font()

def _make_style(font_size: int) -> str:
    fs  = font_size
    fs1 = max(8, font_size - 1)
    return f"""
QMainWindow,QWidget{{
  background:#f5f6fa; color:#1a1a2e;
  font-family:'Segoe UI','Helvetica Neue',sans-serif; font-size:{fs}px;}}
QGroupBox{{
  border:1px solid #c8ccd8; border-radius:5px;
  margin-top:10px; padding-top:8px;
  color:#4a5270; font-size:{fs1}px; font-weight:600;}}
QGroupBox::title{{subcontrol-origin:margin; left:8px;}}
QListWidget{{
  background:#ffffff; border:1px solid #c8ccd8;
  border-radius:4px; color:#1a1a2e;
  selection-background-color:#1565C0;
  selection-color:#ffffff; outline:none;}}
QListWidget::item{{padding:5px 8px;}}
QListWidget::item:hover{{background:#e8eaf6;}}
QPushButton{{
  background:#ffffff; border:1px solid #c8ccd8;
  border-radius:4px; color:#1a1a2e;
  padding:4px 12px; min-height:24px; font-size:{fs}px;}}
QPushButton:hover{{background:#e8eaf6; border-color:#9fa8da;}}
QPushButton:pressed{{background:#c5cae9;}}
QPushButton#primary{{
  background:#1565C0; border-color:#0d47a1;
  color:#ffffff; font-weight:600;}}
QPushButton#primary:hover{{background:#1976D2;}}
QPushButton#danger{{
  background:#C62828; border-color:#b71c1c;
  color:#ffffff; font-weight:600;}}
QPushButton#danger:hover{{background:#D32F2F;}}
QPushButton#green{{
  background:#2E7D32; border-color:#1B5E20;
  color:#ffffff; font-weight:600;}}
QPushButton#green:hover{{background:#388E3C;}}
QPushButton#colour{{
  border:2px solid #9fa8da; border-radius:4px;
  min-width:28px; max-width:28px; min-height:24px;}}
QLineEdit,QComboBox{{
  background:#ffffff; border:1px solid #c8ccd8;
  border-radius:4px; color:#1a1a2e; padding:3px 8px; font-size:{fs}px;}}
QLineEdit:focus,QComboBox:focus{{border-color:#1565C0;}}
QTextEdit{{
  background:#fafafa; border:1px solid #c8ccd8;
  border-radius:4px; color:#1a1a2e;
  font-family:'Consolas',monospace; font-size:{fs1}px;}}
QComboBox QAbstractItemView{{
  background:#ffffff; color:#1a1a2e;
  selection-background-color:#e8eaf6;}}
QCheckBox{{color:#1a1a2e; spacing:6px; font-size:{fs}px;}}
QCheckBox::indicator{{
  width:15px; height:15px;
  background:#ffffff; border:1px solid #9fa8da; border-radius:3px;}}
QCheckBox::indicator:checked{{
  background:#1565C0; border-color:#0d47a1;}}
QSplitter::handle{{background:#c8ccd8; width:2px; height:2px;}}
QMenuBar{{
  background:#f5f6fa; color:#1a1a2e;
  border-bottom:1px solid #c8ccd8; font-size:{fs}px;}}
QMenuBar::item:selected{{background:#e8eaf6;}}
QMenu{{background:#ffffff; border:1px solid #c8ccd8; color:#1a1a2e; font-size:{fs}px;}}
QMenu::item:selected{{background:#e8eaf6;}}
QLabel{{color:#4a5270; font-size:{fs}px;}}
QScrollBar:vertical{{
  background:#f5f6fa; width:8px; border-radius:4px;}}
QScrollBar::handle:vertical{{
  background:#c8ccd8; border-radius:4px; min-height:20px;}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical{{height:0;}}
QSlider::groove:horizontal{{height:4px;background:#c8ccd8;border-radius:2px;}}
QSlider::handle:horizontal{{background:#1565C0;border-radius:6px;
  width:14px;height:14px;margin:-5px 0;}}
"""

# Keep STYLE as a convenience alias for the default size
STYLE = _make_style(12)


def _style_ax(ax, bg="#ffffff"):
    """Style a matplotlib axes using the current global font size."""
    fs = _APP_FONT_SIZE
    ax.set_facecolor(bg)
    ax.tick_params(colors="#444466", labelsize=max(7, fs - 3))
    for sp in ax.spines.values():
        sp.set_color("#c8ccd8")
    ax.xaxis.label.set_color("#4a5270")
    ax.xaxis.label.set_fontsize(max(8, fs - 2))
    ax.yaxis.label.set_color("#4a5270")
    ax.yaxis.label.set_fontsize(max(8, fs - 2))
    ax.title.set_color("#1a1a2e")
    ax.title.set_fontsize(fs)
    ax.grid(True, color="#e8eaf0", linewidth=0.7)


def _colour_icon(colour_hex: str, w=18, h=18) -> QIcon:
    """Return a solid-colour square icon for the colour button."""
    px = QPixmap(w, h)
    px.fill(QColor(colour_hex))
    return QIcon(px)


class PlotCanvas(FigureCanvas):
    """
    Matplotlib canvas with two rendering paths:

    rebuild() + refresh()  — full redraw (on filter add/remove/run).
                              Recreates axes, redraws all lines.

    scroll_xlim()          — fast scroll/zoom.
                             Only calls ax.set_xlim() + draw_idle() for each
                             visible scrollable axis.  No Python-level line
                             drawing happens at all — matplotlib just pans
                             the existing renderer output.
    """

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#ffffff", tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._bg_colour  = "#ffffff"
        self._scroll_axes: list = []    # axes that respond to xlim changes
        self._n_axes: int = 0

    def set_bg(self, colour: str):
        self._bg_colour = colour
        self.fig.patch.set_facecolor(colour)

    def rebuild(self, n: int) -> list:
        self.fig.clear()
        self._scroll_axes = []
        self._n_axes = n
        axes = []
        for i in range(n):
            ax = self.fig.add_subplot(n, 1, i + 1)
            _style_ax(ax, self._bg_colour)
            axes.append(ax)
        try:
            self.fig.tight_layout(pad=1.5)
        except Exception:
            pass
        return axes

    def register_scroll_axes(self, axes: list):
        """Tell the canvas which axes should move on scroll/zoom."""
        self._scroll_axes = axes

    def scroll_xlim(self, start: float, end: float):
        """
        Fast path: update only x limits and redraw via draw_idle().
        draw_idle() coalesces multiple rapid calls into a single repaint
        so scrolling with the arrow keys stays fluid.
        """
        for ax in self._scroll_axes:
            ax.set_xlim(start, end)
        self.draw_idle()

    def refresh(self):
        try:
            self.fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self.draw()

    def clear(self):
        self.fig.clear()
        self._scroll_axes = []
        self.draw()


# ── main window ───────────────────────────────────────────────────────────────
class DigiScopeApp:
    def __init__(self):
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._app.setStyleSheet(_make_style(12))
        self._win = MainWindow()

    def run(self):
        self._win.show()
        sys.exit(self._app.exec_())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DigiScope  –  ECG Signal Analysis")
        self.resize(1440, 820)
        self._chain      = FilterChain()
        self._start_t    = 0.0
        self._win_len    = 5.0
        self._plot_bg    = "#ffffff"   # user-selectable canvas background
        self._build_menu()
        self._build_ui()
        # Debounce timer: coalesces rapid run() calls (e.g. holding down arrow key)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(30)
        self._debounce.timeout.connect(self._run_chain)
        # Per-subplot Y-axis zoom overrides  { subplot_index: (ylo, yhi) }
        self._ylim_overrides: dict = {}
        self._axes_list: list = []      # kept in sync with _replot()
        self._font_size: int = 12       # current UI + plot font size

    # ── menus ─────────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        self._act(fm, "Load MIT-BIH Record…",       self._load_mitbih)
        self._act(fm, "Open DigiScope .dat File…",  self._load_dat)
        self._act(fm, "Open Text / CSV File…",      self._load_text)
        self._act(fm, "Open NumPy .npy File…",      self._load_npy)
        fm.addSeparator()
        self._act(fm, "Generate Waveform / Bitstream…", self._gen_waveform)
        self._act(fm, "Generate Synthetic ECG…",    self._gen_ecg)
        fm.addSeparator()
        self._act(fm, "Save Signal as .dat…",        self._save_dat)
        self._act(fm, "Save Signal as CSV…",         self._save_csv)
        self._act(fm, "Save Signal as TXT…",         self._save_txt)
        fm.addSeparator()
        self._act(fm, "Export Any Track as .dat…",   self._export_track_dat)
        self._act(fm, "Export Any Track as CSV…",    self._export_track_csv)
        self._act(fm, "Export Any Track as TXT…",    self._export_track_txt)
        fm.addSeparator()
        self._act(fm, "Save Filter Chain…",          self._save_chain)
        self._act(fm, "Load Filter Chain…",          self._load_chain)
        fm.addSeparator()
        self._act(fm, "Exit", self.close)

        tm = mb.addMenu("Tools")
        self._act(tm, "Change Plot Background…",    self._pick_bg)
        self._act(tm, "Export Plot to PNG…",        self._export_png)
        tm.addSeparator()
        self._act(tm, "Open DigiScope RT  (Pico Live Input)…",
                  self._open_rt_window)

    def _act(self, menu, label, slot):
        a = QAction(label, self)
        a.triggered.connect(slot)
        menu.addAction(a)

    # ── UI layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── left panel ──
        left = QWidget(); left.setFixedWidth(256)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(4)
        splitter.addWidget(left)

        grp = QGroupBox("Filter Chain")
        gl = QVBoxLayout(grp)
        gl.setContentsMargins(4, 12, 4, 4); gl.setSpacing(3)

        self._flist = QListWidget()
        self._flist.setSelectionMode(QAbstractItemView.SingleSelection)
        self._flist.itemSelectionChanged.connect(self._on_sel)
        self._flist.itemDoubleClicked.connect(self._cfg_filter)
        gl.addWidget(self._flist)

        r1 = QHBoxLayout()
        self._b_add  = self._btn("+ Add",    self._add_filter, "green")
        self._b_del  = self._btn("✕ Del",    self._del_filter, "danger")
        self._b_cfg  = self._btn("⚙ Config", self._cfg_filter)
        r1.addWidget(self._b_add)
        r1.addWidget(self._b_del)
        r1.addWidget(self._b_cfg)
        gl.addLayout(r1)

        r2 = QHBoxLayout()
        self._b_up   = self._btn("▲",       self._move_up)
        self._b_dn   = self._btn("▼",       self._move_down)
        self._b_run  = self._btn("▶ Run",   self._run_chain, "primary")
        self._b_copy = self._btn("⎘ Copy",  self._copy_filter)
        r2.addWidget(self._b_up)
        r2.addWidget(self._b_dn)
        r2.addWidget(self._b_run)
        r2.addWidget(self._b_copy)
        gl.addLayout(r2)

        # Colour + visibility row
        r3 = QHBoxLayout()
        self._chk_vis = QCheckBox("Show")
        self._chk_vis.setChecked(True)
        self._chk_vis.stateChanged.connect(self._toggle_vis)

        self._colour_btn = QPushButton()
        self._colour_btn.setObjectName("colour")
        self._colour_btn.setToolTip("Change signal colour")
        self._colour_btn.setIcon(_colour_icon(_SIG_COLOURS[0]))
        self._colour_btn.clicked.connect(self._pick_signal_colour)

        r3.addWidget(self._chk_vis)
        r3.addStretch()
        r3.addWidget(QLabel("Colour:"))
        r3.addWidget(self._colour_btn)
        gl.addLayout(r3)

        ll.addWidget(grp, stretch=3)

        grp2 = QGroupBox("Output")
        g2l = QVBoxLayout(grp2)
        g2l.setContentsMargins(4, 12, 4, 4)
        self._info = QTextEdit()
        self._info.setReadOnly(True)
        self._info.setMaximumHeight(180)
        g2l.addWidget(self._info)
        ll.addWidget(grp2, stretch=1)

        # ── right panel ──
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        rl.addLayout(self._build_nav())
        self._canvas = PlotCanvas()
        self._canvas.mpl_connect("button_press_event", self._on_plot_click)
        rl.addWidget(self._canvas, stretch=1)
        self._mpl_tb = NavToolbar(self._canvas, self)
        self._mpl_tb.setStyleSheet(
            "QToolBar{background:#f5f6fa;border:none;}"
            "QToolButton{color:#1a1a2e;}"
        )
        rl.addWidget(self._mpl_tb)

    def _build_nav(self):
        bar = QHBoxLayout(); bar.setSpacing(3)
        def nb(label, slot, w=34):
            b = QPushButton(label); b.setFixedWidth(w)
            b.clicked.connect(slot); return b
        bar.addWidget(nb("|◀", self._go_start))
        bar.addWidget(nb("◀◀", self._go_back_lg))
        bar.addWidget(nb("◀",  self._go_back_sm))
        bar.addWidget(nb("▶",  self._go_fwd_sm))
        bar.addWidget(nb("▶▶", self._go_fwd_lg))
        bar.addWidget(nb("▶|", self._go_end))
        bar.addWidget(nb("🔍+", self._zoom_in,  44))
        bar.addWidget(nb("🔍-", self._zoom_out, 44))
        bar.addWidget(QLabel("Window (s):"))
        self._t_edit = QLineEdit(f"0.00  {self._win_len:.2f}")
        self._t_edit.setFixedWidth(110)
        self._t_edit.returnPressed.connect(self._on_t_edit)
        bar.addWidget(self._t_edit)
        bar.addWidget(QLabel("Y-range:"))
        self._y_edit = QLineEdit("")
        self._y_edit.setFixedWidth(90)
        self._y_edit.setPlaceholderText("auto")
        self._y_edit.returnPressed.connect(self._replot)
        bar.addWidget(self._y_edit)
        bar.addStretch()
        # Font size — plain textbox, press Enter to apply
        bar.addWidget(QLabel("Font:"))
        self._font_edit = QLineEdit("12")
        self._font_edit.setFixedWidth(38)
        self._font_edit.setAlignment(Qt.AlignCenter)
        self._font_edit.setToolTip("Font size in pt — press Enter to apply")
        self._font_edit.returnPressed.connect(self._on_font_edit)
        bar.addWidget(self._font_edit)
        bar.addWidget(QLabel("pt"))
        return bar

    def _btn(self, label, slot, obj=""):
        b = QPushButton(label)
        if obj: b.setObjectName(obj)
        b.clicked.connect(slot)
        return b

    # ── signal loading ─────────────────────────────────────────────────────────
    def _load_mitbih(self):
        dlg = MITBIHDialog(self)
        if dlg.exec_() != QDialog.Accepted: return
        record, ch, db, load_ann = dlg.result()

        # ── Run the download in a background thread so the UI stays alive ──
        self._mitbih_worker = _MITBIHWorker(record, ch, db, load_ann)

        prog = QProgressDialog(
            f"Downloading  {db}/{record} …\n\n"
            "Records are cached after the first download.",
            "Cancel", 0, 0, self)
        prog.setWindowTitle("Loading from PhysioNet")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)       # show immediately
        prog.setMinimumWidth(380)
        prog.setValue(0)                 # start the spinner

        def _on_cancel():
            self._mitbih_worker.requestInterruption()
            self._mitbih_worker.quit()

        def _on_done(sig, ann, err):
            prog.close()
            if err:
                QMessageBox.critical(self, "MIT-BIH Load Error",
                    err + "\n\n(pip install --upgrade wfdb)")
                return
            self._init_chain(sig)
            if load_ann:
                if ann is not None:
                    self._chain.filters[0]._annotations = ann
                    self._chain.run()
                    self._replot()
                    n   = len(ann["sample"])
                    ext = ann.get("ext", "?")
                    self._info.append(
                        f"Annotations loaded: {n} beats  "
                        f"(from .{ext})  "
                        f"Types: {sorted(set(ann['symbol']))}")
                else:
                    QMessageBox.information(self, "Annotations",
                        "Record loaded but no annotation file was found.\n"
                        "The signal has been loaded without annotations.")

        prog.canceled.connect(_on_cancel)
        self._mitbih_worker.finished.connect(_on_done)
        self._mitbih_worker.status.connect(
            lambda msg: prog.setLabelText(msg))
        self._mitbih_worker.finished.connect(
            lambda *_: self._mitbih_worker.deleteLater())
        self._mitbih_worker.start()
        prog.exec_()

    def _load_dat(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open DigiScope .dat", "", "DAT (*.dat);;All (*)")
        if not path: return
        try:
            self._init_chain(loader.load_digiscope_dat(path))
        except Exception as e:
            QMessageBox.critical(self, "DAT Load Error", str(e))

    def _load_text(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Text/CSV Signal", "",
            "Text Files (*.txt *.csv *.dat);;All (*)")
        if not path: return
        dlg = TextLoadDialog(self)
        if dlg.exec_() != QDialog.Accepted: return
        rate, bits, vh, vl, is_int = dlg.params()
        try:
            self._init_chain(loader.load_text(
                path, rate=rate, bits=bits,
                volt_high=vh, volt_low=vl, is_raw_int=is_int))
        except Exception as e:
            QMessageBox.critical(self, "Text Load Error", str(e))

    def _load_npy(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open NumPy", "", "NumPy (*.npy);;All (*)")
        if not path: return
        rate, ok = self._ask_rate()
        if not ok: return
        try:
            self._init_chain(loader.load_npy(path, rate))
        except Exception as e:
            QMessageBox.critical(self, "NPY Load Error", str(e))

    def _ask_rate(self):
        dlg = QDialog(self); dlg.setWindowTitle("Sample Rate")
        l = QFormLayout(dlg)
        e = QLineEdit("200"); l.addRow("Sample Rate (Hz)", e)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        l.addRow(btns)
        if dlg.exec_() == QDialog.Accepted:
            try: return float(e.text()), True
            except ValueError: return 200.0, True
        return 200.0, False

    def _gen_waveform(self):
        dlg = WaveformDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            try: self._init_chain(dlg.generate())
            except Exception as e:
                QMessageBox.critical(self, "Generator Error", str(e))

    def _gen_ecg(self):
        dlg = SynthECGDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            hr, dur, rate, noise = dlg.params()
            try: self._init_chain(loader.generate_ecg_like(hr, dur, rate, noise))
            except Exception as e:
                QMessageBox.critical(self, "Generator Error", str(e))

    def _save_dat(self):
        idx = self._sel_idx()
        r = self._chain.get_result(max(idx, 0))
        if r is None or not len(r.data):
            QMessageBox.warning(self, "No Signal", "No signal data to save."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save .dat", "", "DAT (*.dat)")
        if path:
            try:
                sig = r.as_signal()
                if len(self._chain):
                    src = self._chain.filters[0]._signal
                    sig.volt_high  = getattr(src, "volt_high",  5.0)
                    sig.volt_low   = getattr(src, "volt_low",  -5.0)
                    sig.resolution = getattr(src, "resolution",  12)
                loader.save_digiscope_dat(path, sig)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))

    def _save_csv(self):
        idx = self._sel_idx()
        r = self._chain.get_result(max(idx, 0))
        if r is None or not len(r.data):
            QMessageBox.warning(self, "No Signal", "No signal data to save."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if path:
            np.savetxt(path, r.data, delimiter=",", header="amplitude", comments="")

    def _save_txt(self):
        """Save currently selected (or source) signal as tab-separated TXT."""
        idx = self._sel_idx()
        r = self._chain.get_result(max(idx, 0))
        if r is None or not len(r.data):
            QMessageBox.warning(self, "No Signal", "No data to save."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save TXT", "", "Text (*.txt)")
        if not path: return
        t = r.t if len(r.t) == len(r.data) else np.arange(len(r.data)) / r.rate
        with open(path, "w") as f:
            f.write("# time_s\tamplitude\n")
            for ti, vi in zip(t, r.data):
                f.write(f"{ti:.8f}\t{vi:.8f}\n")

    # ── Export any track ──────────────────────────────────────────────────────
    _EXPORT_SKIP = ("BeatDetectorFilter", "PanTompkinsFilter")

    def _pick_export_track(self):
        """Show a dialog to pick which track to export. Returns FilterResult or None."""
        if not len(self._chain):
            QMessageBox.warning(self, "No Data", "No signal loaded.")
            return None
        # Build list of exportable tracks (skip multi-output detection filters)
        choices, results = [], []
        for i, f in enumerate(self._chain.filters):
            if type(f).__name__ in self._EXPORT_SKIP:
                continue
            r = self._chain.get_result(i)
            if r is not None and len(r.data) > 0:
                n_samples = len(r.data)
                dur = n_samples / r.rate if r.rate else 0
                choices.append(f"[{i}] {f.name}  ({n_samples:,} samples, {dur:.1f} s)")
                results.append(r)
        if not choices:
            QMessageBox.warning(self, "No Exportable Tracks",
                "No exportable tracks found.\n"
                "Note: Beat Detector and PT Thresholding are excluded "
                "because they output annotations rather than a signal.")
            return None
        if len(choices) == 1:
            return results[0]

        # Show a proper track-picker dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Track to Export")
        dlg.setMinimumWidth(420)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Choose which track to export:"))

        from PyQt5.QtWidgets import QListWidget, QAbstractItemView
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.SingleSelection)
        for name in choices:
            lst.addItem(name)
        lst.setCurrentRow(0)
        lst.itemDoubleClicked.connect(dlg.accept)
        lay.addWidget(lst)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec_() != QDialog.Accepted:
            return None
        row = lst.currentRow()
        if row < 0:
            return None
        return results[row]

    def _export_track_dat(self):
        r = self._pick_export_track()
        if r is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export .dat", "", "DAT (*.dat)")
        if not path: return
        try:
            import utils.data_loader as loader
            sig = r.as_signal()
            if len(self._chain):
                src = self._chain.filters[0]._signal
                sig.volt_high  = getattr(src, "volt_high",  5.0)
                sig.volt_low   = getattr(src, "volt_low",  -5.0)
                sig.resolution = getattr(src, "resolution",  12)
            loader.save_digiscope_dat(path, sig)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_track_csv(self):
        r = self._pick_export_track()
        if r is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV (*.csv)")
        if not path: return
        t = r.t if len(r.t) == len(r.data) else np.arange(len(r.data)) / r.rate
        np.savetxt(path, np.column_stack([t, r.data]),
                   delimiter=",", header="time_s,amplitude", comments="")

    def _export_track_txt(self):
        r = self._pick_export_track()
        if r is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export TXT", "", "Text (*.txt)")
        if not path: return
        t = r.t if len(r.t) == len(r.data) else np.arange(len(r.data)) / r.rate
        with open(path, "w") as f:
            f.write("# time_s\tamplitude\n")
            for ti, vi in zip(t, r.data):
                f.write(f"{ti:.8f}\t{vi:.8f}\n")

    # ── Save / load filter chain ──────────────────────────────────────────────
    def _save_chain(self):
        """Save the current filter chain configuration to a JSON file."""
        import json, pickle, base64
        if len(self._chain) <= 1:
            QMessageBox.information(self, "Save Chain",
                "Add some filters to the chain before saving."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Filter Chain", "", "DigiScope Chain (*.dsc)")
        if not path: return
        chain_data = []
        for f in self._chain.filters[1:]:   # skip SourceFilter
            cls_name = type(f).__name__
            entry = {"class": cls_name, "name": f.name}
            # Serialise filter-specific attributes
            attrs = {}
            for attr in ("b", "a", "zero_phase", "_label", "mode",
                         "threshold", "edge", "min_gap",
                         "look_ahead", "signal_update", "noise_update",
                         "signal_sb_update", "rr_high_perc", "rr_low_perc",
                         "rr_miss_lim", "peak_dur_ms", "qrs_update",
                         "factor_p", "factor_q", "ratio",
                         "num_avg", "noise_amp",
                         "target_hz", "window_n", "hop_n", "N",
                         "visible", "colour",
                         "_tmpl_start", "_tmpl_end"):
                v = getattr(f, attr, None)
                if v is None: continue
                if isinstance(v, np.ndarray):
                    attrs[attr] = v.tolist()
                elif isinstance(v, (int, float, bool, str, list)):
                    attrs[attr] = v
            entry["attrs"] = attrs
            chain_data.append(entry)
        try:
            with open(path, "w") as fh:
                json.dump(chain_data, fh, indent=2)
            QMessageBox.information(self, "Saved",
                f"Filter chain saved ({len(chain_data)} filters).")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_chain(self):
        """Load a filter chain from a JSON .dsc file and append to current chain."""
        import json
        from filters import FILTER_REGISTRY
        if not len(self._chain):
            QMessageBox.warning(self, "Load Chain",
                "Load a signal first, then load a chain."); return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Filter Chain", "", "DigiScope Chain (*.dsc)")
        if not path: return
        try:
            with open(path) as fh:
                chain_data = json.load(fh)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return

        reg = {cls.__name__: cls for _, cls in FILTER_REGISTRY}
        added = 0
        for entry in chain_data:
            cls_name = entry.get("class", "")
            if cls_name not in reg:
                continue
            f = reg[cls_name]()
            f.name = entry.get("name", f.name)
            for attr, val in entry.get("attrs", {}).items():
                try:
                    if isinstance(getattr(f, attr, None), np.ndarray):
                        setattr(f, attr, np.array(val))
                    else:
                        setattr(f, attr, val)
                except Exception:
                    pass
            self._chain.append(f)
            added += 1

        if added:
            self._chain.run()
            self._refresh_list()
            self._replot()
            QMessageBox.information(self, "Loaded",
                f"Loaded {added} filter(s) from chain.")
        else:
            QMessageBox.warning(self, "Load Chain",
                "No compatible filters found in file.")

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot", "", "PNG (*.png)")
        if path:
            self._canvas.fig.savefig(path, dpi=150,
                facecolor=self._canvas.fig.get_facecolor())

    def _open_rt_window(self):
        """Launch the Real-Time DigiScope window (Pico live input)."""
        try:
            from realtime.rt_window import RTWindow
            self._rt_win = RTWindow()
            self._rt_win.show()
        except ImportError as e:
            QMessageBox.critical(self, "DigiScope RT",
                f"Could not open Real-Time window:\n{e}\n\n"
                "Make sure pyserial is installed:\n  pip install pyserial")

    def _pick_bg(self):
        c = QColorDialog.getColor(QColor(self._plot_bg), self, "Plot Background")
        if c.isValid():
            self._plot_bg = c.name()
            self._canvas.set_bg(self._plot_bg)
            self._replot()

    # ── chain management ───────────────────────────────────────────────────────
    def _init_chain(self, signal: Signal):
        self._chain   = FilterChain()
        src = SourceFilter(signal)
        src.colour = _next_colour(0)
        self._chain.append(src)
        self._start_t = 0.0
        self._win_len = min(5.0, signal.duration)
        self._ylim_overrides.clear()   # reset Y overrides for new signal
        self._chain.run()
        self._refresh_list()
        self._replot()

    def _add_filter(self):
        if not len(self._chain):
            QMessageBox.information(self, "No Signal",
                "Load or generate a signal first, then add filters.")
            return
        dlg = AddFilterDialog(self)
        if dlg.exec_() != QDialog.Accepted: return
        cls = dlg.selected_class()
        if cls is None: return
        f = cls()
        f.colour = _next_colour(len(self._chain))
        # For TemplateMatchFilter, pass signal data on first add too
        if type(f).__name__ == "TemplateMatchFilter":
            src = self._chain.filters[0]
            raw_sig = getattr(src, "_signal", None)
            if raw_sig is not None and len(raw_sig.data) > 0:
                f.configure(self, signal_data=raw_sig.data, rate=raw_sig.rate)
            else:
                f.configure(self)
        else:
            f.configure(self)
        self._chain.append(f)
        self._chain.run()
        self._refresh_list()
        self._replot()

    def _del_filter(self):
        idx = self._sel_idx()
        if idx <= 0: return
        if QMessageBox.question(self, "Remove", "Remove this filter?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self._chain.remove(idx)
            self._chain.run()
            self._refresh_list()
            self._replot()

    def _cfg_filter(self):
        idx = self._sel_idx()
        if idx < 0: return
        f = self._chain.filters[idx]
        if type(f).__name__ == "TemplateMatchFilter":
            # Read signal directly from SourceFilter (no chain run needed)
            sig_data = None
            sig_rate = 360.0
            if len(self._chain.filters) > 0:
                src = self._chain.filters[0]
                raw_sig = getattr(src, "_signal", None)
                if raw_sig is not None and len(raw_sig.data) > 0:
                    sig_data = raw_sig.data
                    sig_rate = raw_sig.rate
            changed = f.configure(self, signal_data=sig_data, rate=sig_rate)
        else:
            changed = f.configure(self)
        if changed:
            self._chain.run(from_index=idx)
            self._replot()

    def _move_up(self):
        idx = self._sel_idx()
        self._chain.move_up(idx)
        self._chain.run()
        self._refresh_list(); self._replot()
        if idx - 1 >= 0: self._flist.setCurrentRow(idx - 1)

    def _move_down(self):
        idx = self._sel_idx()
        self._chain.move_down(idx)
        self._chain.run()
        self._refresh_list(); self._replot()
        if idx + 1 < len(self._chain): self._flist.setCurrentRow(idx + 1)

    def _copy_filter(self):
        idx = self._sel_idx()
        if idx <= 0: return
        self._chain.copy(idx)
        self._chain.filters[-1].colour = _next_colour(len(self._chain) - 1)
        self._chain.run()
        self._refresh_list(); self._replot()

    def _run_chain(self):
        self._chain.run(); self._replot()

    def _toggle_vis(self, state):
        idx = self._sel_idx()
        if idx < 0: return
        self._chain.filters[idx].visible = bool(state)
        self._replot()

    def _pick_signal_colour(self):
        idx = self._sel_idx()
        if idx < 0: return
        f = self._chain.filters[idx]
        current = QColor(getattr(f, "colour", "#1565C0"))
        c = QColorDialog.getColor(current, self, "Signal Colour")
        if c.isValid():
            f.colour = c.name()
            self._colour_btn.setIcon(_colour_icon(f.colour))
            self._replot()

    def _on_sel(self):
        idx = self._sel_idx()
        if idx < 0: return
        f = self._chain.filters[idx]
        self._chk_vis.blockSignals(True)
        self._chk_vis.setChecked(f.visible)
        self._chk_vis.blockSignals(False)
        colour = getattr(f, "colour", _next_colour(idx))
        self._colour_btn.setIcon(_colour_icon(colour))

    def _sel_idx(self) -> int:
        items = self._flist.selectedItems()
        return self._flist.row(items[0]) if items else -1

    def _refresh_list(self):
        self._flist.clear()
        for i, f in enumerate(self._chain.filters):
            label = ("📊 " if i == 0 else "⚙ ") + f.name
            item = QListWidgetItem(label)
            colour = getattr(f, "colour", _next_colour(i))
            item.setForeground(QColor(colour if f.visible else "#aaaaaa"))
            self._flist.addItem(item)
        self._info.setPlainText(self._chain.all_output_text())

    # ── plotting ──────────────────────────────────────────────────────────────
    def _replot(self):
        visible = [f for f in self._chain.filters if f.visible]
        n_axes  = sum(f.num_plots for f in visible)
        if n_axes == 0:
            self._canvas.clear(); return

        self._canvas.set_bg(self._plot_bg)
        axes = self._canvas.rebuild(n_axes)
        ax_i = 0
        for f in self._chain.filters:
            if not f.visible:
                continue
            ci = self._chain.filters.index(f)
            result = self._chain.get_result(ci)
            if result is None:
                ax_i += f.num_plots; continue
            # inject colour into result so plot() can use it
            result.extras["_colour"] = getattr(f, "colour", _next_colour(ci))
            try:
                ax_slice = axes[ax_i: ax_i + f.num_plots]
                f.plot(ax_slice, result)
                for ax in ax_slice:
                    if f.scroll:
                        ax.set_xlim(self._start_t,
                                    self._start_t + self._win_len)
            except Exception as e:
                axes[ax_i].text(0.5, 0.5, f"Plot error:\n{e}",
                    transform=axes[ax_i].transAxes,
                    ha="center", va="center", color="red", fontsize=8)
            ax_i += f.num_plots

        yr = self._y_edit.text().strip()
        if yr:
            try:
                parts = [float(x) for x in yr.replace(",", " ").split()]
                if len(parts) == 2:
                    for ax in axes:
                        ax.set_ylim(parts[0], parts[1])
            except ValueError:
                pass

        # Keep axes list and apply any stored Y overrides
        self._axes_list = axes
        for i, ax in enumerate(axes):
            if i in self._ylim_overrides:
                ax.set_ylim(*self._ylim_overrides[i])

        # Draw per-plot Y-zoom buttons (+ / - overlaid top-left of each axis)
        for i, ax in enumerate(axes):
            self._add_yzoom_buttons(ax, i)

        # Register the scrollable axes for the fast scroll_xlim() path
        scroll_axes = []
        ax_i2 = 0
        for f in self._chain.filters:
            if not f.visible: continue
            for p in range(f.num_plots):
                if f.scroll and ax_i2 < len(axes):
                    scroll_axes.append(axes[ax_i2])
                ax_i2 += 1
        self._canvas.register_scroll_axes(scroll_axes)

        self._canvas.refresh()
        self._info.setPlainText(self._chain.all_output_text())
        self._t_edit.setText(
            f"{self._start_t:.2f}  {self._start_t + self._win_len:.2f}")

    # ── navigation ────────────────────────────────────────────────────────────
    def _apply_font(self, size: int):
        """Apply font size to Qt stylesheet and trigger a plot redraw."""
        global _APP_FONT_SIZE
        size = max(7, min(32, int(size)))
        _APP_FONT_SIZE = size
        self._font_size = size
        QApplication.instance().setStyleSheet(_make_style(size))
        if hasattr(self, "_font_edit"):
            self._font_edit.blockSignals(True)
            self._font_edit.setText(str(size))
            self._font_edit.blockSignals(False)
        # Redraw plots so axes labels/titles pick up the new size
        if hasattr(self, "_canvas") and len(self._chain):
            self._replot()

    def _on_font_edit(self):
        """Called when user presses Enter in the font size textbox."""
        try:
            size = int(self._font_edit.text().strip())
            self._apply_font(size)
        except ValueError:
            self._font_edit.setText(str(self._font_size))

    def _dur(self):
        if not len(self._chain): return 10.0
        r = self._chain.get_result(0)
        return r.duration if r else 10.0

    # ── per-plot Y-axis zoom ─────────────────────────────────────────────────
    def _add_yzoom_buttons(self, ax, idx: int):
        """
        Overlay two small text annotations acting as +/- Y-zoom buttons
        at the top-left corner of the given axes.
        The annotations store the subplot index so _on_plot_click can identify them.
        """
        # Remove any old yzoom annotations on this axes
        for child in list(ax.get_children()):
            if getattr(child, "_yzoom_tag", None) is not None:
                child.remove()

        kw = dict(
            transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f0fe",
                      edgecolor="#9fa8da", alpha=0.85),
            ha="center", va="center",
            picker=True,   # makes it respond to button_press_event
        )
        btn_plus = ax.text(0.025, 0.93, " + ", **kw)
        btn_plus._yzoom_tag  = ("zoom_in",  idx)
        btn_minus = ax.text(0.075, 0.93, " − ", **kw)
        btn_minus._yzoom_tag = ("zoom_out", idx)
        btn_reset = ax.text(0.125, 0.93, " ↺ ", **kw)
        btn_reset._yzoom_tag = ("zoom_reset", idx)

    def _on_plot_click(self, event):
        """Handle clicks on the Y-zoom overlay buttons."""
        if event.inaxes is None:
            return
        ax = event.inaxes
        # Check if click landed on a yzoom annotation
        for child in ax.get_children():
            tag = getattr(child, "_yzoom_tag", None)
            if tag is None:
                continue
            # Check if click is within the annotation bbox
            try:
                bbox = child.get_window_extent(
                    renderer=self._canvas.get_renderer())
                if bbox.contains(event.x, event.y):
                    action, idx = tag
                    self._yzoom(action, idx)
                    return
            except Exception:
                pass

    def _yzoom(self, action: str, idx: int):
        """Zoom/reset the Y axis of subplot idx."""
        if idx >= len(self._axes_list):
            return
        ax = self._axes_list[idx]
        if action == "zoom_reset":
            self._ylim_overrides.pop(idx, None)
            ax.set_ylim(auto=True)
            ax.autoscale(axis="y")
        else:
            current = ax.get_ylim()
            mid  = (current[0] + current[1]) / 2
            half = (current[1] - current[0]) / 2
            if action == "zoom_in":
                half *= 0.6
            else:
                half *= 1.667
            new_lo = mid - half
            new_hi = mid + half
            self._ylim_overrides[idx] = (new_lo, new_hi)
            ax.set_ylim(new_lo, new_hi)
        try:
            self._canvas.fig.tight_layout(pad=1.5)
        except Exception:
            pass
        self._canvas.draw_idle()

    # ── fast scroll helpers ────────────────────────────────────────────────────
    def _scroll(self):
        """
        Fast path for navigation: update xlim without redrawing plot lines.
        Falls back to full _replot() if no scroll axes are registered yet.
        """
        if self._canvas._scroll_axes:
            self._canvas.scroll_xlim(self._start_t,
                                     self._start_t + self._win_len)
            self._t_edit.setText(
                f"{self._start_t:.2f}  {self._start_t + self._win_len:.2f}")
        else:
            self._replot()

    def _go_start(self):
        self._start_t = 0.0; self._scroll()
    def _go_end(self):
        self._start_t = max(0.0, self._dur() - self._win_len); self._scroll()
    def _go_back_sm(self):
        self._start_t = max(0.0, self._start_t - self._win_len / 10); self._scroll()
    def _go_back_lg(self):
        self._start_t = max(0.0, self._start_t - self._win_len); self._scroll()
    def _go_fwd_sm(self):
        self._start_t = min(self._dur() - self._win_len,
                            self._start_t + self._win_len / 10); self._scroll()
    def _go_fwd_lg(self):
        self._start_t = min(self._dur() - self._win_len,
                            self._start_t + self._win_len); self._scroll()
    def _zoom_in(self):
        self._win_len = max(0.05, self._win_len / 1.5); self._replot()
    def _zoom_out(self):
        self._win_len = min(self._dur(), self._win_len * 1.5); self._replot()

    def _on_t_edit(self):
        try:
            parts = [float(x) for x in self._t_edit.text().split()]
            if len(parts) == 2:
                self._start_t = max(0.0, parts[0])
                self._win_len = parts[1] - parts[0]
                self._replot()   # full redraw when window size changes
        except ValueError:
            pass


# ── Base filter colour support (monkey-patch into BaseFilter.__init__) ────────
# All concrete filter plot() methods should honour result.extras["_colour"].
# We patch it here so every existing filter gets a colour attribute for free.
from core.filter_base import BaseFilter as _BF
_orig_init = _BF.__init__
def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    if not hasattr(self, "colour"):
        self.colour = _SIG_COLOURS[0]
_BF.__init__ = _patched_init


# ── patch all existing filter plot() methods to use _colour ──────────────────
def _patch_plot(cls):
    """Wrap an existing plot() to pick up _colour from result.extras."""
    orig = cls.plot
    def new_plot(self, axes, result):
        colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
        ax = axes[0]; ax.clear()
        _style_ax(ax, ax.get_facecolor() if ax.get_facecolor() != (0,0,0,0) else "#ffffff")
        ax.plot(result.t, result.data, color=colour, linewidth=0.9)
        ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")
        ax.set_title(self.name, fontsize=_APP_FONT_SIZE)
        ax.grid(True, color="#e8eaf0", linewidth=0.7)
    cls.plot = new_plot

from filters.signal_processing import (
    RemoveMeanFilter, AddNoiseFilter, FullWaveRectFilter,
    SquaringFilter, DerivativeDetectFilter
)
from filters.source import SourceFilter as _SF

# Patch generic filters to use colour
for _cls in (RemoveMeanFilter, AddNoiseFilter, FullWaveRectFilter,
             SquaringFilter, DerivativeDetectFilter):
    _patch_plot(_cls)

# Source filter — special: uses colour too
def _src_plot(self, axes, result):
    colour = result.extras.get("_colour", getattr(self, "colour", "#1565C0"))
    ax = axes[0]; ax.clear()
    _style_ax(ax)
    ax.plot(result.t, result.data, color=colour, linewidth=0.9)
    ax.set_ylabel("Amplitude (mV)"); ax.set_xlabel("Time (s)")
    ax.set_title(self._signal.title or "ECG Signal", fontsize=_APP_FONT_SIZE)
    ax.grid(True, color="#e8eaf0", linewidth=0.7)
    # Overlay MIT-BIH annotations if present
    ann = getattr(self, "_annotations", None)
    if ann is not None:
        t_ann = ann["time"]
        sym   = ann["symbol"]
        # Only plot annotations within the current x-range
        xlim = ax.get_xlim()
        y_range = ax.get_ylim()
        y_top = y_range[1]
        y_bot = y_range[0]
        y_span = y_top - y_bot
        for ti, si in zip(t_ann, sym):
            if xlim[0] <= ti <= xlim[1]:
                # Vertical line at beat
                ax.axvline(ti, color="#E65100", linewidth=0.5, alpha=0.5, zorder=2)
                # Label above signal
                ax.text(ti, y_top - y_span * 0.08, si,
                        ha="center", va="top", fontsize=max(5, _APP_FONT_SIZE - 5),
                        color="#E65100", fontweight="bold", zorder=3,
                        clip_on=True)
_SF.plot = _src_plot


# ── Dialogs ───────────────────────────────────────────────────────────────────

# ── Filter tab / favorites config ────────────────────────────────────────────
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

# Tab definitions: tab_name -> list of registry names that belong there
_FILTER_TABS = {
    "★ Favorites":  [],   # populated from saved prefs
    "Common": [
        "Custom Filter Designer",
        "Squaring",
        "MWI (Moving Window Integrator)",
        "Template Matching",
        "Resample",
        "PT Thresholding",
    ],
    "Detection": [
        "Beat Detector (threshold)",
        "PT Thresholding",
    ],
    "Signal": [
        "Remove Mean",
        "Add Noise",
        "ECG Filter (Butterworth/Notch)",
    ],
    "All": [],   # all filters
}


class AddFilterDialog(QDialog):
    """
    Tabbed filter picker with per-tab lists and a favorites star button.
    Favorites are persisted to ~/.digiscope_favs.json.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Processing Step")
        self.setMinimumWidth(340)
        self.setMinimumHeight(420)

        self._favs       = _load_favs()
        self._selected   : tuple | None = None   # (name, cls)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self._tabs = QTabWidget()
        self._lists: dict[str, QListWidget] = {}
        self._tab_items: dict[str, list] = {}  # tab_name -> [(name, cls)]

        # Build registry lookup
        reg = {name: cls for name, cls in FILTER_REGISTRY}
        all_items = list(FILTER_REGISTRY)

        for tab_name, names in _FILTER_TABS.items():
            if tab_name == "★ Favorites":
                items = [(n, c) for n, c in all_items if n in self._favs]
            elif tab_name == "All":
                items = all_items
            else:
                items = [(n, reg[n]) for n in names if n in reg]

            lw = QListWidget()
            lw.itemDoubleClicked.connect(self.accept)
            for name, _ in items:
                item = QListWidgetItem(
                    ("★ " if name in self._favs else "  ") + name)
                lw.addItem(item)
            if items:
                lw.setCurrentRow(0)

            self._lists[tab_name]     = lw
            self._tab_items[tab_name] = items
            self._tabs.addTab(lw, tab_name)

        layout.addWidget(self._tabs)

        # Bottom row: star button + OK/Cancel
        bot = QHBoxLayout()
        self._star_btn = QPushButton("★ Toggle Favorite")
        self._star_btn.clicked.connect(self._toggle_fav)
        bot.addWidget(self._star_btn)
        bot.addStretch()
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        bot.addWidget(btns)
        layout.addLayout(bot)

    def _current_item(self):
        """Return (name, cls) for the currently highlighted item, or None."""
        tab  = self._tabs.tabText(self._tabs.currentIndex())
        lw   = self._lists.get(tab)
        items = self._tab_items.get(tab, [])
        if lw is None or lw.currentRow() < 0:
            return None
        row = lw.currentRow()
        if row >= len(items):
            return None
        return items[row]

    def _toggle_fav(self):
        item = self._current_item()
        if item is None:
            return
        name = item[0]
        if name in self._favs:
            self._favs.discard(name)
        else:
            self._favs.add(name)
        _save_favs(self._favs)
        # Refresh star prefix on the current item
        tab = self._tabs.tabText(self._tabs.currentIndex())
        lw  = self._lists[tab]
        row = lw.currentRow()
        lw.item(row).setText(("★ " if name in self._favs else "  ") + name)
        # Refresh favorites tab
        fav_lw = self._lists.get("★ Favorites")
        fav_items = [(n, c) for n, c in FILTER_REGISTRY if n in self._favs]
        self._tab_items["★ Favorites"] = fav_items
        if fav_lw:
            fav_lw.clear()
            for n, _ in fav_items:
                fav_lw.addItem(("★ " if n in self._favs else "  ") + n)

    def selected_class(self):
        item = self._current_item()
        return item[1] if item else None


# ── Background worker for MIT-BIH / PhysioNet downloads ─────────────────────
class _MITBIHWorker(QThread):
    """
    Runs wfdb.rdrecord (and optional rdann) in a background thread.
    Emits:
      status(str)              – human-readable progress message
      finished(sig, ann, err)  – sig=Signal, ann=dict|None, err=str|None
    """
    status   = pyqtSignal(str)
    finished = pyqtSignal(object, object, object)   # sig, ann, err_str

    def __init__(self, record: str, channel: int, db: str, load_ann: bool):
        super().__init__()
        self._record   = record
        self._channel  = channel
        self._db       = db
        self._load_ann = load_ann

    def run(self):
        try:
            self.status.emit(
                f"Connecting to PhysioNet …\n"
                f"Database: {self._db}   Record: {self._record}\n\n"
                "Records are cached after the first download.")

            if self._load_ann:
                from utils.data_loader import load_mitbih_with_annotations
                sig, ann = load_mitbih_with_annotations(
                    self._record, self._channel, self._db)
                self.finished.emit(sig, ann, None)
            else:
                import utils.data_loader as _loader
                sig = _loader.load_mitbih(self._record, self._channel, self._db)
                self.finished.emit(sig, None, None)

        except Exception as e:
            self.finished.emit(None, None, str(e))


# ── PhysioNet database catalogue ──────────────────────────────────────────────
# Each entry: (display_name, db_id, default_record, channel_hint, record_hint, ann_note)
_PHYSIONET_DBS = [
    ("MIT-BIH Arrhythmia Database",
     "mitdb", "100", "0 = MLII  |  1 = V5",
     "48 records: 100–124, 200–234",
     "Annotations: .atr  (N=Normal, V=PVC, A=PAC, F=Fusion…)"),
    ("MIT-BIH Normal Sinus Rhythm Database",
     "nsrdb", "16265", "0 = ECG1  |  1 = ECG2",
     "18 records: 16265–16795",
     "Annotations: .atr  (N=Normal beats only — healthy subjects)"),
    ("MIT-BIH Atrial Fibrillation Database",
     "afdb",  "04015", "0 = ECG1  |  1 = ECG2",
     "25 records: 04015–08434",
     "Annotations: .atr  (N=Normal, AFIB=Atrial Fibrillation, AFL=Flutter…)"),
    ("MIT-BIH Supraventricular Arrhythmia Database",
     "svdb",  "800",   "0 = ECG1  |  1 = ECG2",
     "78 records: 800–899",
     "Annotations: .atr  (SVT, AT, AVNRT, AVRT and other SVA types)"),
    ("MIT-BIH Noise Stress Test Database",
     "nstdb", "118e00","0 = MLII  |  1 = V5",
     "Records: 118e00–119e24  (noise levels 0–24 dB SNR)",
     "Annotations: .atr  (same beats as record 118/119 with added noise)"),
    ("MIT-BIH Polysomnographic Database",
     "slpdb", "slp01a","0 = ECG",
     "Records: slp01a–slp67x  (sleep study recordings)",
     "Annotations: .ecg  (sleep-stage + arrhythmia labels)"),
    ("European ST-T Database",
     "edb",   "e0103", "0 = ECG1  |  1 = ECG2",
     "90 records: e0103–e1124",
     "Annotations: .atr + .sta (ST changes, T-wave alterations)"),
    ("MIT-BIH Long-Term ECG Database",
     "ltdb",  "14046", "0 = ECG1",
     "7 records: 14046–15814  (up to 24-hour recordings)",
     "Annotations: .atr  (full beat annotations)"),
    ("Fantasia Database",
     "fantasia","f1o01","0 = ECG",
     "40 records: f1o01–f2y20  (young & elderly, rest & fantasy)",
     "Annotations: .atr  (N=Normal only — healthy subjects)"),
    ("BIDMC Congestive Heart Failure Database",
     "chfdb", "chf01", "0 = ECG1  |  1 = ECG2",
     "15 records: chf01–chf15  (severe CHF patients)",
     "Annotations: .atr  (N=Normal, V=PVC, A=PAC)"),
    ("Other (type manually)", "__other__", "", "", "", ""),
]
_DB_NAMES   = [d[0] for d in _PHYSIONET_DBS]
_DB_IDS     = [d[1] for d in _PHYSIONET_DBS]


class MITBIHDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load PhysioNet Record")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Database selector ──────────────────────────────────────────────
        from PyQt5.QtWidgets import QGroupBox
        grp_db = QGroupBox("PhysioNet Database")
        db_l = QFormLayout(grp_db)

        self._db_combo = QComboBox()
        self._db_combo.addItems(_DB_NAMES)
        self._db_combo.currentIndexChanged.connect(self._on_db_change)
        db_l.addRow("Database:", self._db_combo)

        self._db_custom = QLineEdit()
        self._db_custom.setPlaceholderText("e.g.  mitdb  /  afdb  /  nsrdb")
        self._db_custom.setVisible(False)
        db_l.addRow("Database ID:", self._db_custom)

        layout.addWidget(grp_db)

        # ── Record & channel ───────────────────────────────────────────────
        grp_rec = QGroupBox("Record")
        rec_l = QFormLayout(grp_rec)

        self._rec  = QLineEdit("100")
        self._ch   = QLineEdit("0")
        self._rec_hint = QLabel("")
        self._ch_hint  = QLabel("")
        for lbl in (self._rec_hint, self._ch_hint):
            lbl.setStyleSheet("color:#4a7090; font-size:10px; font-style:italic;")
            lbl.setWordWrap(True)

        rec_l.addRow("Record number:", self._rec)
        rec_l.addRow("",               self._rec_hint)
        rec_l.addRow("Channel:",       self._ch)
        rec_l.addRow("",               self._ch_hint)
        layout.addWidget(grp_rec)

        # ── Annotations ────────────────────────────────────────────────────
        grp_ann = QGroupBox("Annotations")
        ann_l = QFormLayout(grp_ann)
        self._ann_chk = QCheckBox("Load beat annotations")
        self._ann_chk.setChecked(True)
        self._ann_note = QLabel("")
        self._ann_note.setStyleSheet("color:#4a7090; font-size:10px;")
        self._ann_note.setWordWrap(True)
        ann_l.addRow(self._ann_chk)
        ann_l.addRow(self._ann_note)
        layout.addWidget(grp_ann)

        # ── General info ───────────────────────────────────────────────────
        self._gen_info = QLabel(
            "Records download automatically on first use and are cached "
            "in ~/physionet-data/  |  Requires internet on first load\n"
            "If loading fails:  pip install --upgrade wfdb"
        )
        self._gen_info.setWordWrap(True)
        self._gen_info.setStyleSheet("color:#7080a0; font-size:10px;")
        layout.addWidget(self._gen_info)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Initialise hints for first entry
        self._on_db_change(0)

    def _on_db_change(self, idx: int):
        entry = _PHYSIONET_DBS[idx]
        _, db_id, def_rec, ch_hint, rec_hint, ann_note = entry

        is_other = (db_id == "__other__")
        self._db_custom.setVisible(is_other)

        if not is_other:
            self._rec.setText(def_rec)

        self._rec_hint.setText(rec_hint)
        self._ch_hint.setText(ch_hint)
        self._ann_note.setText(ann_note)

    def result(self):
        idx = self._db_combo.currentIndex()
        db_id = _PHYSIONET_DBS[idx][1]
        if db_id == "__other__":
            db_id = self._db_custom.text().strip() or "mitdb"
        try:
            ch = int(self._ch.text())
        except ValueError:
            ch = 0
        return (self._rec.text().strip(), ch, db_id, self._ann_chk.isChecked())


class TextLoadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text File Settings")
        layout = QFormLayout(self)
        self._rate = QLineEdit("200")
        self._bits = QLineEdit("12")
        self._vh   = QLineEdit("5.0")
        self._vl   = QLineEdit("-5.0")
        self._raw  = QCheckBox("Values are raw integers (convert using bit depth + range)")
        self._raw.setChecked(False)
        layout.addRow("Sample Rate (Hz)",     self._rate)
        layout.addRow("Bit Depth (ADC bits)", self._bits)
        layout.addRow("Volt High",            self._vh)
        layout.addRow("Volt Low",             self._vl)
        layout.addRow("",                     self._raw)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def params(self):
        try:
            return (float(self._rate.text()), int(self._bits.text()),
                    float(self._vh.text()), float(self._vl.text()),
                    self._raw.isChecked())
        except ValueError:
            return 200.0, 12, 5.0, -5.0, False


class WaveformDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Waveform / Bitstream")
        self.setMinimumWidth(380)
        layout = QFormLayout(self)

        self._type = QComboBox()
        self._type.addItems(["Sine", "Square", "Triangle", "Sawtooth", "Bitstream"])
        self._type.currentIndexChanged.connect(self._on_type_change)
        layout.addRow("Waveform Type", self._type)

        # Standard waveform params
        self._freq = QLineEdit("1.0")
        self._amp  = QLineEdit("1.0")
        self._duty = QLineEdit("0.5")
        self._dur  = QLineEdit("10.0")
        self._rate = QLineEdit("10000.0")
        self._freq_row = layout.addRow("Frequency (Hz)",          self._freq)
        self._amp_row  = layout.addRow("Amplitude (peak)",         self._amp)
        self._duty_row = layout.addRow("Duty Cycle (square only)", self._duty)
        self._dur_row  = layout.addRow("Duration (s)",             self._dur)
        self._rate_row = layout.addRow("Sample Rate (Hz)",         self._rate)

        # Bitstream-specific params (hidden by default)
        self._bits_edit = QLineEdit("10110100")
        self._sym_rate  = QLineEdit("1000")
        self._repeat    = QLineEdit("4")
        self._bits_row  = layout.addRow("Bit pattern (0s and 1s):", self._bits_edit)
        self._symrate_row = layout.addRow("Symbol Rate (sym/s):",   self._sym_rate)
        self._repeat_row  = layout.addRow("Repeat count:",          self._repeat)
        self._bits_note = QLabel(
            "Each bit is one symbol.  The pattern repeats N times.\n"
            "Apply an RRC filter afterward to shape the pulses.")
        self._bits_note.setWordWrap(True)
        self._bits_note.setStyleSheet("color:#4a5270; font-size:10px;")
        layout.addRow(self._bits_note)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

        self._layout = layout
        self._on_type_change(0)  # hide bitstream rows initially

    def _on_type_change(self, idx):
        is_bits = (idx == 4)  # "Bitstream"
        # Standard rows
        for w in (self._freq, self._amp, self._duty, self._dur):
            w.setVisible(not is_bits)
        # Rate shared by both
        # Bitstream rows
        for w in (self._bits_edit, self._sym_rate, self._repeat, self._bits_note):
            w.setVisible(is_bits)
        # Update labels visibility
        layout = self._layout
        # Walk form rows and toggle labels
        for row in range(layout.rowCount()):
            lbl = layout.itemAt(row, QFormLayout.LabelRole)
            fld = layout.itemAt(row, QFormLayout.FieldRole)
            if fld is None or lbl is None:
                continue
            widget = fld.widget()
            if widget in (self._bits_edit, self._sym_rate, self._repeat, self._bits_note):
                if lbl.widget(): lbl.widget().setVisible(is_bits)
            elif widget in (self._freq, self._amp, self._duty, self._dur):
                if lbl.widget(): lbl.widget().setVisible(not is_bits)

    def generate(self) -> Signal:
        wt   = self._type.currentIndex()
        rate = float(self._rate.text())
        if wt == 4:  # Bitstream
            bits    = self._bits_edit.text().strip()
            sym_r   = float(self._sym_rate.text())
            repeat  = int(self._repeat.text())
            amp     = 1.0
            return loader.generate_bitstream(bits, sym_r, rate, repeat, amp)
        freq = float(self._freq.text()); amp  = float(self._amp.text())
        duty = float(self._duty.text()); dur  = float(self._dur.text())
        if wt == 0: return loader.generate_sine(freq, amp, dur, rate)
        if wt == 1: return loader.generate_square(freq, amp, duty, dur, rate)
        if wt == 2: return loader.generate_triangle(freq, amp, dur, rate)
        return loader.generate_sawtooth(freq, amp, dur, rate)


class SynthECGDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Synthetic ECG")
        layout = QFormLayout(self)
        self._hr    = QLineEdit("70")
        self._dur   = QLineEdit("10")
        self._rate  = QLineEdit("360")
        self._noise = QLineEdit("0.0")
        layout.addRow("Heart Rate (BPM)",       self._hr)
        layout.addRow("Duration (s)",           self._dur)
        layout.addRow("Sample Rate (Hz)",       self._rate)
        layout.addRow("Noise Amplitude",        self._noise)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def params(self):
        try:
            return (float(self._hr.text()), float(self._dur.text()),
                    float(self._rate.text()), float(self._noise.text()))
        except ValueError:
            return 70.0, 10.0, 360.0, 0.0
