"""
realtime/rt_buffer.py  –  Thread-safe circular sample buffer

Key fix: update_rate() only resets the buffer if the rate actually changed.
Previously it always reset, wiping samples that arrived during handshake.
"""

from __future__ import annotations
import numpy as np
import threading


class RTBuffer:
    def __init__(self, capacity_s: float = 60.0, rate_hz: float = 360.0):
        self._rate       = float(rate_hz)
        self._capacity_s = capacity_s
        self._capacity_n = int(capacity_s * rate_hz)
        self._buf        = np.zeros(self._capacity_n, dtype=np.float32)
        self._head       = 0
        self._total      = 0
        self._lock       = threading.Lock()

    def push(self, samples: np.ndarray):
        n = len(samples)
        with self._lock:
            cap = self._capacity_n
            if n >= cap:
                self._buf[:] = samples[-cap:]
                self._head   = 0
                self._total += n
                return
            end = self._head + n
            if end <= cap:
                self._buf[self._head:end] = samples
            else:
                split = cap - self._head
                self._buf[self._head:] = samples[:split]
                self._buf[:n - split]  = samples[split:]
            self._head  = end % cap
            self._total += n

    def update_rate(self, new_rate: float):
        """Update sample rate. Only resets buffer if rate actually changed."""
        new_rate = float(new_rate)
        with self._lock:
            if new_rate == self._rate:
                return          # ← key fix: don't wipe buffer on re-confirmation
            self._rate       = new_rate
            self._capacity_n = int(self._capacity_s * new_rate)
            self._buf        = np.zeros(self._capacity_n, dtype=np.float32)
            self._head       = 0
            self._total      = 0

    def snapshot(self, duration_s: float) -> np.ndarray:
        with self._lock:
            n_want = min(int(duration_s * self._rate), self._capacity_n)
            n_have = min(self._total, self._capacity_n)
            n      = min(n_want, n_have)
            if n == 0:
                return np.array([], dtype=np.float32)
            head  = self._head
            cap   = self._capacity_n
            start = (head - n) % cap
            if start < head:
                return self._buf[start:head].copy()
            else:
                return np.concatenate(
                    [self._buf[start:], self._buf[:head]]).copy()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def total_samples(self) -> int:
        return self._total

    @property
    def duration_buffered(self) -> float:
        return min(self._total, self._capacity_n) / max(self._rate, 1)
