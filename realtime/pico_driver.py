"""
realtime/pico_driver.py  –  PC-side driver for the Pico ADC streamer

Key design decision: once streaming starts the byte stream is ALMOST entirely
binary packets [LSB, MSB, 0xAA, 0x55].  The previous version tried to detect
text lines vs binary packets mid-stream, but this broke because ADC bytes in
the range 0x20-0x7E look like printable ASCII, causing the parser to eat real
samples while looking for a newline.

Solution: text responses (PONG, RATE N, etc.) are only emitted during the
handshake phase BEFORE binary streaming begins.  We drain all text in
_drain_text_responses() first, then enter a pure binary parser that only looks
for the 0xAA 0x55 sync marker.  If the Pico sends a rate-change confirmation
mid-stream it arrives as "RATE 720\\n" whose bytes will be skipped by the sync
scanner (they don't have 0xAA 0x55 at positions [2][3]) — this is fine because
the rate was already set and confirmed during the handshake.
"""

from __future__ import annotations
import struct, time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

SYNC_A         = 0xAA
SYNC_B         = 0x55
PKT_LEN        = 4        # [LSB, MSB, 0xAA, 0x55]
CHUNK          = 256      # samples per emit
BAUD           = 115200   # ignored by USB CDC but required by pyserial
ADC_FULL_SCALE = 65535.0


class PicoDriver(QThread):
    """
    Background thread: reads binary packets from the Pico, converts to
    volts, emits chunks of CHUNK samples at a time.

    Signals
    -------
    samples_ready(np.ndarray)  – float32 voltages
    status_msg(str)            – human-readable info
    error(str)                 – fatal error, driver will stop
    rate_confirmed(int)        – Pico confirmed a sample rate
    """

    samples_ready  = pyqtSignal(object)
    status_msg     = pyqtSignal(str)
    error          = pyqtSignal(str)
    rate_confirmed = pyqtSignal(int)

    def __init__(self, port: str, rate_hz: int = 360,
                 vref: float = 3.3, bits: int = 16,
                 dc_bias: float = 0.0, parent=None):
        super().__init__(parent)
        self.port      = port
        self.rate_hz   = rate_hz
        self.vref      = vref
        self.bits      = bits
        self.dc_bias   = dc_bias
        self._running  = False
        self._ser      = None
        self._pending_rate: int | None = None
        self._pending_cmds: list       = []

    # ── public API ────────────────────────────────────────────────────────────
    def set_rate(self, hz: int):
        self.rate_hz       = hz
        self._pending_rate = hz

    def send_command(self, cmd: str):
        self._pending_cmds.append(cmd.strip() + "\n")

    def stop(self):
        self._running = False
        self.quit()

    # ── thread entry ──────────────────────────────────────────────────────────
    def run(self):
        try:
            import serial
        except ImportError:
            self.error.emit(
                "pyserial is not installed.\n  pip install pyserial")
            return

        self._running      = True
        self._pending_rate = None
        self._pending_cmds = []

        # Open port
        try:
            self._ser = serial.Serial(
                self.port, BAUD, timeout=1.0, write_timeout=1.0)
            self._ser.reset_input_buffer()
        except Exception as e:
            self.error.emit(f"Cannot open {self.port}:\n{e}")
            return

        self.status_msg.emit(f"Opened {self.port}")

        # ── Handshake: send PING + SET_RATE, then drain all text responses ────
        # After this phase the Pico switches to pure binary streaming and text
        # responses are no longer interleaved with packet data.
        time.sleep(0.4)     # let USB CDC enumerate and Pico print greeting
        self._send("PING\n")
        time.sleep(0.05)
        self._send(f"SET_RATE {self.rate_hz}\n")
        time.sleep(0.05)
        self._send("START\n")
        time.sleep(0.3)     # give Pico time to confirm and begin streaming

        # Drain and parse any buffered text before entering binary mode
        self._drain_text()

        self.status_msg.emit(
            f"Streaming  {self.rate_hz} Hz | "
            f"Vref={self.vref} V | bias={self.dc_bias} V")

        # ── Pure binary receive loop ──────────────────────────────────────────
        buf     = bytearray()
        acc     = []
        read_sz = PKT_LEN * CHUNK * 4   # read generously

        while self._running and not self.isInterruptionRequested():

            # Send any queued commands
            self._flush_pending()
            # Recalculate scale each iteration so live vref/bias changes apply
            scale = self.vref / ADC_FULL_SCALE

            # Read available bytes (non-blocking poll)
            try:
                waiting = self._ser.in_waiting
                if waiting == 0:
                    time.sleep(0.003)
                    continue
                buf.extend(self._ser.read(min(waiting, read_sz)))
            except Exception as e:
                self.error.emit(f"Serial read error: {e}")
                break

            # ── Scan for sync-marked packets ──────────────────────────────────
            # We scan every possible 4-byte window looking for 0xAA at [2]
            # and 0x55 at [3].  When found we extract the uint16 from [0:2].
            # Anything that doesn't match is skipped one byte at a time
            # (resync).  This is robust to any garbage, partial packets, or
            # out-of-band text bytes the Pico might send mid-stream.
            i = 0
            while i + PKT_LEN <= len(buf):
                if buf[i + 2] == SYNC_A and buf[i + 3] == SYNC_B:
                    raw     = struct.unpack_from("<H", buf, i)[0]
                    acc.append(raw * scale - self.dc_bias)
                    i += PKT_LEN
                else:
                    i += 1   # resync

            buf = buf[i:]   # keep unprocessed tail

            # Emit completed chunk to GUI thread
            if len(acc) >= CHUNK:
                self.samples_ready.emit(
                    np.array(acc[:CHUNK], dtype=np.float32))
                acc = acc[CHUNK:]

        # Flush remaining samples
        if acc:
            self.samples_ready.emit(np.array(acc, dtype=np.float32))

        # Shutdown
        try:
            if self._ser and self._ser.is_open:
                self._ser.write(b"STOP\n")
                self._ser.close()
        except Exception:
            pass
        self.status_msg.emit("Driver stopped.")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _send(self, text: str):
        try:
            if self._ser and self._ser.is_open:
                self._ser.write(text.encode())
                self._ser.flush()
        except Exception:
            pass

    def _flush_pending(self):
        if self._pending_rate is not None:
            self._send(f"SET_RATE {self._pending_rate}\n")
            self._pending_rate = None
        for cmd in list(self._pending_cmds):
            self._send(cmd)
        self._pending_cmds.clear()

    def _drain_text(self):
        """
        Read and process all buffered text lines from the Pico.
        Stops when no more data arrives for 150 ms — at that point we
        assume the Pico has switched to binary streaming.
        """
        deadline = time.time() + 1.5   # max 1.5 s to drain greeting
        last_data = time.time()
        partial = b""

        while time.time() < deadline:
            try:
                waiting = self._ser.in_waiting
            except Exception:
                break

            if waiting == 0:
                if time.time() - last_data > 0.15:
                    break   # 150 ms silence → streaming has started
                time.sleep(0.02)
                continue

            chunk = self._ser.read(waiting)
            last_data = time.time()
            partial += chunk

            # Process complete lines
            while b"\n" in partial:
                line, partial = partial.split(b"\n", 1)
                text = line.decode("latin-1", errors="replace").strip()
                if text:
                    self._handle_text(text)

        # Put any remaining bytes back in the serial buffer isn't possible,
        # but if partial contains what looks like binary packets, we'll catch
        # them in the main loop since they go into buf there.
        # (partial at this point is at most a few bytes of an incomplete line)

    def _handle_text(self, line: str):
        """Process a text response from the Pico."""
        parts = line.split()
        # Match "RATE 360" or "DIGISCOPE_PICO RATE 360"
        for i, tok in enumerate(parts):
            if tok == "RATE" and i + 1 < len(parts):
                try:
                    hz = int(parts[i + 1])
                    self.rate_hz = hz
                    self.rate_confirmed.emit(hz)
                    self.status_msg.emit(
                        f"Pico confirmed {hz} Hz | "
                        f"Vref={self.vref} V | bias={self.dc_bias} V")
                except ValueError:
                    pass
                return
        if line == "PONG":
            self.status_msg.emit("Pico alive (PONG)")
        elif line.startswith("ERR"):
            self.status_msg.emit(f"Pico error: {line}")
        elif line.startswith("OK"):
            pass   # OK START / OK STOP — ignore
        elif line.startswith("DIGISCOPE_PICO"):
            # Greeting line — already handled by RATE token above
            pass


# ── Port discovery ─────────────────────────────────────────────────────────────
def list_pico_ports() -> list:
    """
    Return ALL available serial ports as [(device, description), ...],
    with likely Pico/CDC ports sorted to the top.
    Always returns everything so the user can manually pick if heuristics fail.
    """
    try:
        import serial.tools.list_ports as lp

        PICO_VID = 0x2E8A   # Raspberry Pi

        likely, other = [], []
        for p in sorted(lp.comports(), key=lambda x: x.device):
            desc = (p.description or "").lower()
            is_likely = (
                (p.vid == PICO_VID) or
                any(x in desc for x in
                    ("pico", "rp2", "cdc", "usb serial device",
                     "usb serial", "usbmodem")) or
                p.device.startswith("/dev/ttyACM") or
                "usbmodem" in p.device.lower()
            )
            label = (p.description or p.device) + (
                f"  [VID:{p.vid:04X}]" if p.vid else "")
            entry = (p.device, label)
            (likely if is_likely else other).append(entry)

        return likely + other

    except ImportError:
        return []


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ports = list_pico_ports()
    if not ports:
        print("No serial ports found.  pip install pyserial")
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Available ports:")
        for p, d in ports:
            print(f"  {p:20s}  {d}")
        print(f"\nUsage:  python -m realtime.pico_driver <port>")
        sys.exit(0)

    from PyQt5.QtCore import QCoreApplication
    app  = QCoreApplication(sys.argv)
    drv  = PicoDriver(sys.argv[1], rate_hz=360)
    t0   = [time.time()]
    tot  = [0]

    def on_samples(arr):
        tot[0] += len(arr)
        elapsed = time.time() - t0[0]
        print(f"\r{tot[0]:8d} samples  {tot[0]/elapsed:6.1f} Hz  "
              f"mean={arr.mean():.3f} V   ", end="", flush=True)
        if elapsed > 5:
            drv.stop(); app.quit()

    drv.samples_ready.connect(on_samples)
    drv.status_msg.connect(lambda m: print(f"\n[status] {m}"))
    drv.error.connect(lambda e: (print(f"\n[error] {e}"), app.quit()))
    drv.start()
    app.exec_()
    print()
