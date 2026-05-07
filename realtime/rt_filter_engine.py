"""
realtime/rt_filter_engine.py  —  Continuous streaming filter engine

Key design decisions
--------------------
1. Streaming filters (IIR/FIR) are processed chunk-by-chunk with zi state
   carried forward so the signal is equivalent to running lfilter on the
   whole concatenated stream.

2. History filters (beat detectors, PT thresholding) need the full signal
   history to work correctly.  They are NOT run in the engine thread.
   Instead, run_history_filters() runs them on the full raw buffer at
   display time in the GUI thread.

3. The CustomCoeffFilter uses calculate() which internally calls lfilter
   or filtfilt.  For RT we MUST use lfilter with zi carried forward.
   We intercept CustomCoeffFilter specifically to do this correctly.
"""

from __future__ import annotations
import queue, numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.signal import lfilter, lfilter_zi

from core.signal import Signal
from core.filter_chain import FilterChain
from filters.source import SourceFilter


_HISTORY_FILTER_CLASSES = (
    "BeatDetectorFilter",
    "PanTompkinsFilter",
    "GoertzelFilter",
    "PowerSpectrumFilter",
    "SpectrogramFilter",
    "TemplateMatchFilter",
    "AveragingFilter",
    "CompressionFilter",
)

_STATEFUL_FILTER_CLASSES = (
    "ECGFilterFilter",
    "RemoveMeanFilter",
    "AddNoiseFilter",
    "FullWaveRectFilter",
    "SquaringFilter",
    "DerivativeDetectFilter",
    "ResampleFilter",
    "FFTBandFilter",
    "RRCFilter",
    "CustomCoeffFilter",
    "CustomPythonFilter",
    "MWIFilter",
)


class RTFilterEngine(QThread):
    """
    Background thread: processes raw samples through the filter chain as
    they arrive, maintaining filter state (zi) across chunks.
    """

    results_ready = pyqtSignal()
    output_text   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue   : queue.Queue = queue.Queue(maxsize=512)
        self._chain   : FilterChain = FilterChain()
        self._buffers : dict        = {}   # filter_index -> RTBuffer
        self._rate    : float       = 360.0
        self._running : bool        = False
        self._zi      : dict        = {}   # filter_index -> zi array for lfilter

    # ── Public API ────────────────────────────────────────────────────────────

    def set_chain(self, chain: FilterChain, rate: float):
        self._chain   = chain
        self._rate    = rate
        self._zi      = {}
        self._buffers = {}
        for i in range(len(chain.filters)):
            self._buffers[i] = _make_buffer(120.0, rate)

    def set_rate(self, rate: float):
        self._rate = rate
        self._zi   = {}
        for i in self._buffers:
            self._buffers[i].update_rate(rate)

    def push_chunk(self, samples: np.ndarray):
        try:
            self._queue.put_nowait(samples.astype(np.float64))
        except queue.Full:
            pass

    def get_filter_buffer(self, filter_index: int):
        return self._buffers.get(filter_index)

    def stop(self):
        self._running = False
        self.quit()

    # ── Engine thread ─────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        while self._running and not self.isInterruptionRequested():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not len(self._chain.filters):
                continue
            try:
                self._process_chunk(chunk)
                self.results_ready.emit()
            except Exception:
                pass

    def _process_chunk(self, raw: np.ndarray):
        """
        Run one chunk through every streaming filter, carrying zi forward.
        """
        prev_out = raw
        texts    = []

        for i, filt in enumerate(self._chain.filters):
            cls_name = type(filt).__name__

            if cls_name == "SourceFilter":
                if 0 in self._buffers:
                    self._buffers[0].push(raw.astype(np.float32))
                continue

            if cls_name in _HISTORY_FILTER_CLASSES:
                # History filters run at display time — store input as placeholder
                if i in self._buffers:
                    self._buffers[i].push(prev_out.astype(np.float32))
                continue

            # ── Stateful streaming filter ─────────────────────────────────────
            try:
                if cls_name == "CustomCoeffFilter":
                    # Run lfilter directly with zi state — bypasses filtfilt
                    out = self._run_custom_lfilter(i, filt, prev_out)
                else:
                    sig    = Signal.from_array(prev_out.copy(), self._rate)
                    result = filt.calculate(sig)
                    out    = result.data
                    if result.output_text:
                        texts.append(result.output_text)
                    # Align length
                    if len(out) != len(prev_out):
                        if len(out) > len(prev_out):
                            out = out[:len(prev_out)]
                        else:
                            out = np.pad(out, (0, len(prev_out) - len(out)))

                if filt.passthrough:
                    prev_out = out

                if i in self._buffers:
                    self._buffers[i].push(out.astype(np.float32))

            except Exception:
                if i in self._buffers:
                    self._buffers[i].push(prev_out.astype(np.float32))

        if texts:
            self.output_text.emit("\n".join(texts))

    def _run_custom_lfilter(self, idx: int, filt, x: np.ndarray) -> np.ndarray:
        """
        Run CustomCoeffFilter using lfilter with zi state carried forward.
        This avoids filtfilt (which can't run on chunks) and the dropoff
        artefact from resetting zi to zero on every chunk.
        """
        from filters.custom import _normalise
        b = np.array(filt.b, dtype=float)
        a = np.array(filt.a, dtype=float)
        b, a = _normalise(b, a)

        # Initialise zi on first call or after chain reset
        if idx not in self._zi:
            try:
                zi_init = lfilter_zi(b, a)
                self._zi[idx] = zi_init * x[0]
            except Exception:
                self._zi[idx] = np.zeros(max(len(b), len(a)) - 1)

        try:
            out, self._zi[idx] = lfilter(b, a, x, zi=self._zi[idx])
        except Exception:
            out = x.copy()
            self._zi[idx] = np.zeros(max(len(b), len(a)) - 1)

        return out

    # ── History filter runner (called from GUI thread at display time) ────────

    def run_history_filters(self, raw_buf, display_s: float) -> dict:
        """
        Run history-dependent filters on the full raw buffer.
        Returns {filter_index: FilterResult}.
        """
        results = {}
        if not len(self._chain.filters):
            return results

        # Cap history to last 30 s to avoid startup transients polluting
        # the adaptive threshold in PT Thresholding.
        MAX_HISTORY_S = 30.0
        hist_s = min(raw_buf.duration_buffered, MAX_HISTORY_S)
        raw = raw_buf.snapshot(hist_s)
        if len(raw) < 4:
            return results

        rate     = self._rate
        sig      = Signal.from_array(raw.astype(float), rate)
        prev_sig = sig

        for i, filt in enumerate(self._chain.filters):
            cls_name = type(filt).__name__
            if cls_name == "SourceFilter":
                continue
            if cls_name not in _HISTORY_FILTER_CLASSES:
                buf = self._buffers.get(i)
                if buf is not None:
                    out = buf.snapshot(buf.duration_buffered)
                    if len(out) >= 4:
                        prev_sig = Signal.from_array(out.astype(float), rate)
                continue
            try:
                result = filt.calculate(prev_sig)
                results[i] = result
                if filt.passthrough and len(result.data) >= 4:
                    prev_sig = result.as_signal()
            except Exception:
                pass

        return results


def _make_buffer(capacity_s: float, rate: float):
    from realtime.rt_buffer import RTBuffer
    return RTBuffer(capacity_s=capacity_s, rate_hz=rate)
