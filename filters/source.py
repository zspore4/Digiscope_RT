"""
filters/source.py  –  "Source" step: wraps the raw loaded signal.

This is always filters[0] in the chain.  It does no processing;
it simply packages the loaded Signal into a FilterResult so the chain engine
can feed it to subsequent steps.
"""

from __future__ import annotations
import numpy as np
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal


class SourceFilter(BaseFilter):
    name = "Signal Source"
    passthrough = True
    num_plots = 1
    scroll = True

    def __init__(self, signal: Signal):
        super().__init__()
        self._signal = signal

    def set_signal(self, signal: Signal):
        self._signal = signal

    def calculate(self, signal: Signal) -> FilterResult:
        # The "input" here is itself — the raw source
        s = self._signal
        return FilterResult(
            data=s.data.copy(),
            t=s.t.copy(),
            rate=s.rate,
            output_text=f"Source: {s.title}  |  {s.rate:.0f} Hz  |  {s.duration:.2f} s",
            passthrough=True,
        )

    def plot(self, axes: list, result: FilterResult) -> None:
        ax = axes[0]
        ax.clear()
        ax.plot(result.t, result.data, color="#1a1a2e", linewidth=0.8)
        ax.set_ylabel("Amplitude (mV)")
        ax.set_xlabel("Time (s)")
        ax.set_title(self._signal.title or "ECG Signal", fontsize=9)
        ax.grid(True, alpha=0.3)
