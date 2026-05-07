"""
core/filter_base.py  –  Abstract base class for all Digiscope filters.

Every concrete filter (beat detector, noise adder, FFT filter, …) inherits
from BaseFilter and implements:
  • configure(parent_widget)  – show settings dialog
  • calculate(signal)         – run the algorithm, return FilterResult
  • plot(ax_list, result)     – draw on the provided Axes objects

This mirrors the MATLAB switch(varargin{1}) pattern but with proper OOP.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from core.signal import Signal


# ---------------------------------------------------------------------------
@dataclass
class FilterResult:
    """
    Returned by BaseFilter.calculate().

    • data / t / rate  – the pass-through signal for the next filter step
    • output_text      – string shown in the info box (BPM, PRD, etc.)
    • extras           – dict of any additional computed values (qrs, bpm, …)
    • passthrough      – if True the result.data is propagated down the chain
    """
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    t: np.ndarray    = field(default_factory=lambda: np.array([]))
    rate: float      = 200.0
    output_text: str = ""
    extras: dict     = field(default_factory=dict)
    passthrough: bool = True

    @property
    def duration(self) -> float:
        return len(self.data) / self.rate if self.rate > 0 else 0.0

    def as_signal(self) -> Signal:
        """
        Convert result back to a Signal for chaining.
        Uses a *view* of self.data (no copy) — the next filter's calculate()
        should treat the input as read-only and make its own copy if it mutates.
        """
        s = Signal.__new__(Signal)
        s.data        = self.data          # view, not copy
        s.rate        = self.rate
        s.title       = ""
        s.source      = ""
        s.volt_high   = 5.0
        s.volt_low    = -5.0
        s.resolution  = 12
        s.t           = self.t             # view
        s.plot_range  = (0, len(self.data))
        s.total_delay = 0
        return s


# ---------------------------------------------------------------------------
class BaseFilter(ABC):
    """Abstract base for every Digiscope processing step."""

    #: Human-readable name shown in the filter list
    name: str = "Unnamed Filter"

    #: Number of matplotlib Axes subplots this filter needs
    num_plots: int = 1

    #: If True, this filter's output data is fed to the next filter
    passthrough: bool = True

    #: If True, the filter re-runs when the scroll window changes
    scroll: bool = True
    update_on_scroll: bool = False
    update_window: bool = False

    def __init__(self):
        self._result: Optional[FilterResult] = None
        self.visible: bool = True        # controlled by the Plot checkbox

    # ------------------------------------------------------------------
    def configure(self, parent=None) -> bool:
        """
        Show a settings dialog.  Return True if settings changed,
        False if the user cancelled.  Default: no settings.
        """
        return False

    @abstractmethod
    def calculate(self, signal: Signal) -> FilterResult:
        """Run the algorithm.  Must return a FilterResult."""

    @abstractmethod
    def plot(self, axes: list, result: FilterResult) -> None:
        """Draw result on the provided list of matplotlib Axes."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @staticmethod
    def time_vector(data: np.ndarray, rate: float) -> np.ndarray:
        return np.arange(len(data)) / rate

    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}'>"
