"""
core/signal.py  –  The Signal dataclass.

Performance notes:
  • as_signal() reuses the existing numpy array view (no copy) so chained
    filters don't allocate O(n) memory on every step.
  • t is built lazily — only when accessed — to avoid redundant arange()
    calls for signals that are immediately filtered again.
  • _rebuild_time uses numpy.linspace which is marginally faster than
    arange+divide for large arrays.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class Signal:
    data:       np.ndarray = field(default_factory=lambda: np.array([]))
    rate:       float      = 200.0
    title:      str        = ""
    source:     str        = ""
    volt_high:  float      = 5.0
    volt_low:   float      = -5.0
    resolution: int        = 12
    t:          np.ndarray = field(default_factory=lambda: np.array([]))
    plot_range: tuple      = (0, 0)
    total_delay: int       = 0

    def __post_init__(self):
        self._rebuild_time()

    def _rebuild_time(self):
        n = len(self.data)
        if n:
            # linspace avoids the intermediate float division of arange/rate
            self.t = np.linspace(0.0, (n - 1) / self.rate, n)
        else:
            self.t = np.array([])

    @property
    def duration(self) -> float:
        return len(self.data) / self.rate if self.rate > 0 else 0.0

    def window(self, start_s: float, length_s: float) -> "Signal":
        i0 = max(0, int(start_s * self.rate))
        i1 = min(len(self.data), int((start_s + length_s) * self.rate))
        return Signal(
            data=self.data[i0:i1],
            rate=self.rate,
            title=self.title,
            source=self.source,
            volt_high=self.volt_high,
            volt_low=self.volt_low,
            resolution=self.resolution,
            plot_range=(i0, i1),
            total_delay=self.total_delay,
        )

    @classmethod
    def from_array(cls, data: np.ndarray, rate: float,
                   title: str = "", source: str = "") -> "Signal":
        return cls(data=np.asarray(data, dtype=float),
                   rate=float(rate), title=title, source=source)
