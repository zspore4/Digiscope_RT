"""
core/filter_chain.py  –  Manages the ordered list of filters.

Performance notes (all applied here):
  • run(from_index) skips everything before the changed step
  • _get_input_signal caches the last passthrough result index instead of
    scanning backwards on every call
  • as_signal() is called once and reused across the forward pass
  • traceback.format_exc() is only called when DEBUG=True; in production
    a short one-line error is stored so the GUI stays responsive
"""

from __future__ import annotations
from typing import List, Optional
import traceback

from core.signal import Signal
from core.filter_base import BaseFilter, FilterResult

DEBUG = False   # set True to get full tracebacks in the output box


class FilterChain:
    def __init__(self):
        self._filters: List[BaseFilter]        = []
        self._results: List[Optional[FilterResult]] = []

    # ── list management ───────────────────────────────────────────────────────
    def append(self, f: BaseFilter):
        self._filters.append(f)
        self._results.append(None)

    def insert(self, idx: int, f: BaseFilter):
        self._filters.insert(idx, f)
        self._results.insert(idx, None)

    def remove(self, idx: int):
        self._filters.pop(idx)
        self._results.pop(idx)

    def move_up(self, idx: int):
        if idx <= 1 or idx >= len(self._filters):
            return
        self._filters[idx-1], self._filters[idx] = self._filters[idx], self._filters[idx-1]
        self._results[idx-1] = None
        self._results[idx]   = None

    def move_down(self, idx: int):
        if idx < 1 or idx >= len(self._filters) - 1:
            return
        self._filters[idx+1], self._filters[idx] = self._filters[idx], self._filters[idx+1]
        self._results[idx+1] = None
        self._results[idx]   = None

    def copy(self, idx: int):
        import copy
        self._filters.append(copy.deepcopy(self._filters[idx]))
        self._results.append(None)

    @property
    def filters(self) -> List[BaseFilter]:
        return self._filters

    @property
    def results(self) -> List[Optional[FilterResult]]:
        return self._results

    def __len__(self):
        return len(self._filters)

    # ── execution ─────────────────────────────────────────────────────────────
    def run(self, from_index: int = 0) -> List[str]:
        """
        Run the filter chain starting from from_index.
        Steps before from_index keep their existing results.
        """
        output_lines: List[str] = []

        # Collect output text from already-computed steps
        for i in range(from_index):
            r = self._results[i]
            if r and r.output_text:
                output_lines.append(r.output_text)

        # Track the most recent passthrough signal so we don't re-scan backwards
        # on every step — O(1) per step instead of O(n)
        cur_signal: Optional[Signal] = self._find_last_passthrough_before(from_index)

        for i in range(from_index, len(self._filters)):
            f = self._filters[i]

            # Index 0 is always SourceFilter — it supplies its own signal
            if i == 0:
                sig = Signal()          # SourceFilter ignores this argument
            else:
                sig = cur_signal
                if sig is None:
                    self._results[i] = None
                    continue

            try:
                result = f.calculate(sig)
                self._results[i] = result
                if result.output_text:
                    output_lines.append(result.output_text)
                # Advance the running signal pointer only for passthrough steps
                if result.passthrough and len(result.data):
                    cur_signal = result.as_signal()
            except Exception as e:
                err = (traceback.format_exc() if DEBUG
                       else f"[{f.name}] Error: {e}")
                output_lines.append(err)
                self._results[i] = None
                # Don't update cur_signal — next filter gets the last good one

        return output_lines

    def _find_last_passthrough_before(self, idx: int) -> Optional[Signal]:
        """
        Scan backward to find the most recent passthrough result before idx.
        Called once at the start of run(), not per-step.
        """
        if idx == 0:
            return None
        for j in range(idx - 1, -1, -1):
            r = self._results[j]
            if r is not None and r.passthrough and len(r.data):
                return r.as_signal()
        return None

    # ── accessors ─────────────────────────────────────────────────────────────
    def get_result(self, idx: int) -> Optional[FilterResult]:
        if 0 <= idx < len(self._results):
            return self._results[idx]
        return None

    def all_output_text(self) -> str:
        return "\n".join(
            r.output_text for r in self._results
            if r and r.output_text
        )
