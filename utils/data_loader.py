"""
utils/data_loader.py  –  Signal loading utilities.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from core.signal import Signal


def load_digiscope_dat(path) -> Signal:
    """Read DigiScope .dat binary file (port of DigiDatRead.m)."""
    p = Path(path)
    with open(str(p), "rb") as f:
        raw = f.read()
    sep = raw.find(b"\r\n\r\n")
    newline_len = 4
    if sep == -1:
        sep = raw.find(b"\n\n")
        newline_len = 2
    header_bytes = raw[:sep]
    data_bytes   = raw[sep + newline_len:]
    header_lines = header_bytes.decode("latin-1").splitlines()
    def _get(prefix):
        for line in header_lines:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
        return ""
    volt_high  = float(_get("VoltHigh:") or "5")
    volt_low   = float(_get("VoltLow:")  or "-5")
    resolution = int(_get("Resolution:") or "12")
    rate       = float(_get("Rate:")     or "200")
    title      = _get("Title:")
    n = len(data_bytes) // 2
    raw_int = np.frombuffer(data_bytes[:n * 2], dtype="<i2")
    scale = (volt_high - volt_low) / (2 ** resolution)
    data  = raw_int.astype(float) * scale
    sig = Signal.from_array(data, rate, title=title or p.stem, source="digiscope_dat")
    sig.volt_high = volt_high; sig.volt_low = volt_low; sig.resolution = resolution
    return sig


def save_digiscope_dat(path, signal: Signal, title: str = "") -> None:
    """Write DigiScope .dat file (port of DigiDatWrite.m)."""
    p = Path(path)
    vh = signal.volt_high; vl = signal.volt_low; res = signal.resolution
    n  = len(signal.data)
    t  = title or signal.title or "DigiScope"
    scale_back = (2 ** res) / (vh - vl)
    int_data = np.clip(signal.data * scale_back, -(2**15), 2**15 - 1).astype("<i2")
    with open(str(p), "wb") as f:
        def wl(s): f.write((s + "\r\n").encode("latin-1"))
        wl(f"Title:  {t}"); wl(f"Creator:  DigiScope Python"); wl(f"Source:  Python")
        wl(f"Type:  ECG single channel"); wl(f"VoltHigh:  {vh}"); wl(f"VoltLow:  {vl}")
        wl(f"Step:  0"); wl(f"Compress:  no"); wl(f"Resolution:  {res}")
        wl(f"Rate:  {signal.rate}"); wl(f"Channels:  1"); wl(f"Samples:  {n}"); wl(f"Chan:  1"); wl("")
        f.write(int_data.tobytes())


def load_mitbih(record: str, channel: int = 0, pb_dir: str = "mitdb") -> Signal:
    """
    Load a MIT-BIH record.  Handles both old (3.x) and new (4.x) wfdb APIs.

    Tries in order:
      1. Local path read
      2. wfdb.dl_database() download to ~/physionet-data/, then local read
      3. wfdb.rdrecord(pn_dir=...)  -- wfdb >= 4.x
      4. wfdb.rdrecord(pb_dir=...)  -- wfdb 3.x
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb is not installed.  Fix:  pip install wfdb")

    import os, inspect

    rdrecord_params = set(inspect.signature(wfdb.rdrecord).parameters.keys())
    has_pn_dir = "pn_dir" in rdrecord_params
    has_pb_dir = "pb_dir" in rdrecord_params

    record = str(record).strip()
    errors = []

    def _read(rec_obj):
        data = rec_obj.p_signal[:, channel].astype(float)
        title = f"{rec_obj.record_name}  [{rec_obj.sig_name[channel]}]"
        return Signal.from_array(data, float(rec_obj.fs), title=title, source="mitbih")

    # 1. Local file (works if files already downloaded)
    try:
        rec = wfdb.rdrecord(record)
        return _read(rec)
    except Exception as e:
        errors.append(f"local read: {e}")

    # 2. Download via dl_database then read locally
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), "physionet-data", pb_dir)
        os.makedirs(cache_dir, exist_ok=True)
        wfdb.dl_database(pb_dir, dl_dir=cache_dir, records=[record])
        local_path = os.path.join(cache_dir, record)
        rec = wfdb.rdrecord(local_path)
        return _read(rec)
    except Exception as e:
        errors.append(f"dl_database: {e}")

    # 3. pn_dir keyword (wfdb >= 4.x)
    if has_pn_dir:
        try:
            rec = wfdb.rdrecord(record, pn_dir=pb_dir)
            return _read(rec)
        except Exception as e:
            errors.append(f"pn_dir kwarg: {e}")

    # 4. pb_dir keyword (wfdb 3.x)
    if has_pb_dir:
        try:
            rec = wfdb.rdrecord(record, pb_dir=pb_dir)
            return _read(rec)
        except Exception as e:
            errors.append(f"pb_dir kwarg: {e}")

    wfdb_ver = getattr(wfdb, "__version__", "unknown")
    bullet_errors = "\n".join(f"  * {e}" for e in errors)
    raise RuntimeError(
        f"Could not load MIT-BIH record '{record}' from '{pb_dir}'.\n"
        f"wfdb version: {wfdb_ver}\n\n"
        f"Attempts made:\n{bullet_errors}\n\n"
        "Things to try:\n"
        "  1. Check internet connection (required on first download)\n"
        "  2. Upgrade wfdb:   pip install --upgrade wfdb\n"
        "  3. Valid record numbers for mitdb: 100, 101 ... 234\n"
        "  4. If you have .hea/.dat files, pass the full local path instead"
    )


def list_mitbih_records(pb_dir: str = "mitdb") -> list:
    try:
        import wfdb
        return wfdb.get_record_list(pb_dir)
    except Exception:
        return [str(i) for i in range(100, 110)] + [
            "200","201","202","203","205","207","208","209","210",
            "212","213","214","215","217","219","220","221","222",
            "223","228","230","231","232","233","234"]


def load_text(path, rate: float = 200.0, bits: int = 12,
              volt_high: float = 5.0, volt_low: float = -5.0,
              is_raw_int: bool = False) -> Signal:
    """Load plain text/CSV signal file."""
    p = Path(path)
    raw = p.read_text(errors="replace")
    lines = [ln for ln in raw.splitlines()
             if ln.strip() and not ln.strip().startswith(("#", "%"))]
    values = []
    for line in lines:
        for part in line.replace(",", " ").split():
            try:
                values.append(float(part))
            except ValueError:
                pass
    if not values:
        raise ValueError(f"No numeric data found in {path}")
    data = np.array(values, dtype=float)
    if is_raw_int:
        data = data * (volt_high - volt_low) / (2 ** bits)
    sig = Signal.from_array(data, rate, title=p.stem, source="txt")
    sig.volt_high = volt_high; sig.volt_low = volt_low; sig.resolution = bits
    return sig


def load_npy(path, rate: float = 200.0) -> Signal:
    p = Path(path)
    data = np.load(str(p))
    if data.ndim > 1:
        data = data[:, 0]
    return Signal.from_array(data.astype(float), rate, title=p.stem, source="npy")


def generate_sine(freq=1.0, amp=1.0, duration=10.0, rate=200.0) -> Signal:
    t = np.arange(int(duration * rate)) / rate
    return Signal.from_array(amp * np.sin(2 * np.pi * freq * t), rate,
                             title=f"Sine {freq:.2f}Hz {amp:.2f}V", source="wavegen")


def generate_square(freq=1.0, amp=1.0, duty=0.5, duration=10.0, rate=200.0) -> Signal:
    from scipy.signal import square
    t = np.arange(int(duration * rate)) / rate
    return Signal.from_array(amp * square(2 * np.pi * freq * t, duty=duty), rate,
                             title=f"Square {freq:.2f}Hz {amp:.2f}V duty={duty:.0%}", source="wavegen")


def generate_triangle(freq=1.0, amp=1.0, duration=10.0, rate=200.0) -> Signal:
    from scipy.signal import sawtooth
    t = np.arange(int(duration * rate)) / rate
    return Signal.from_array(amp * sawtooth(2 * np.pi * freq * t, width=0.5), rate,
                             title=f"Triangle {freq:.2f}Hz {amp:.2f}V", source="wavegen")


def generate_sawtooth(freq=1.0, amp=1.0, duration=10.0, rate=200.0) -> Signal:
    from scipy.signal import sawtooth
    t = np.arange(int(duration * rate)) / rate
    return Signal.from_array(amp * sawtooth(2 * np.pi * freq * t, width=1.0), rate,
                             title=f"Sawtooth {freq:.2f}Hz {amp:.2f}V", source="wavegen")


def generate_ecg_like(heart_rate=70.0, duration=10.0, rate=360.0, noise=0.0) -> Signal:
    """Realistic synthetic ECG from Gaussian P/Q/R/S/T components."""
    rr  = int(rate * 60.0 / heart_rate)
    n   = int(duration * rate)
    out = np.zeros(n)
    t_idx = np.arange(n)
    components = [
        (-0.30, 0.030,  0.25),  # P
        (-0.05, 0.008, -0.15),  # Q
        ( 0.00, 0.012,  1.00),  # R
        ( 0.05, 0.008, -0.20),  # S
        ( 0.20, 0.050,  0.30),  # T
    ]
    for beat in range(int(n / rr) + 2):
        r = beat * rr + rr // 4
        for off, sf, amp in components:
            center = r + int(off * rr)
            sigma  = max(1, int(sf * rr))
            out   += amp * np.exp(-0.5 * ((t_idx - center) / sigma) ** 2)
    if noise > 0:
        out += noise * np.random.randn(n)
    return Signal.from_array(out, rate, title=f"ECG {heart_rate:.0f}BPM", source="wavegen")


def generate_bitstream(bits: str, symbol_rate: float = 1000.0,
                       rate: float = 10000.0, repeat: int = 1,
                       amp: float = 1.0) -> Signal:
    """
    Generate a baseband square-wave bitstream signal from a string of 0s and 1s.

    Parameters
    ----------
    bits        : string of '0' and '1' characters, e.g. "10110100"
    symbol_rate : symbols per second (Hz)
    rate        : sample rate (Hz)
    repeat      : how many times to repeat the bit pattern
    amp         : amplitude of a '1' bit ('0' bit = 0 V, '1' bit = amp V)

    The output is NRZ (Non-Return-to-Zero): 0→0V, 1→amp.
    """
    # Clean input — keep only 0s and 1s
    bits_clean = [int(b) for b in bits if b in "01"]
    if not bits_clean:
        raise ValueError("No valid bits found.  Enter a string of 0s and 1s.")

    bits_rep = bits_clean * max(1, repeat)
    sps = rate / symbol_rate        # samples per symbol
    n_samples = int(round(sps * len(bits_rep)))
    data = np.zeros(n_samples)

    for i, bit in enumerate(bits_rep):
        i0 = int(round(i * sps))
        i1 = int(round((i + 1) * sps))
        data[i0:i1] = bit * amp

    title = f"Bitstream  {''.join(str(b) for b in bits_clean)}  ×{repeat}  {symbol_rate:.0f} sym/s"
    return Signal.from_array(data, rate, title=title, source="wavegen")


# All known PhysioNet annotation file extensions, in priority order.
# Different databases use different extensions:
#   .atr  – MIT-BIH Arrhythmia (mitdb), NSR, SVDB, NSTDB
#   .ecg  – MIT-BIH Polysomnographic (slpdb)
#   .atr  – MIT-BIH AF Database (afdb) also uses .atr
#   .qrs  – some noise-stress records
#   .man  – manual annotations
#   .st   – ST annotations
#   .ari  – arrhythmia index annotations
ANN_EXTENSIONS = ("atr", "ecg", "qrs", "man", "st", "ari", "ann",
                  "epi", "stf", "trigger", "pwave", "xws")


def load_mitbih_with_annotations(record: str, channel: int = 0,
                                  pb_dir: str = "mitdb") -> tuple:
    """
    Load a MIT-BIH / PhysioNet record AND its annotation file.

    Tries every known annotation extension so it works with:
      mitdb  (.atr),  afdb (.atr),  slpdb (.ecg),  nstdb (.atr/.qrs),
      nsrdb  (.atr),  svdb (.atr),  and others.

    Returns
    -------
    (signal, annotations)  where annotations is a dict:
        'sample'  : np.ndarray of sample indices
        'symbol'  : list of annotation symbol strings (N, V, A, F, …)
        'time'    : np.ndarray of times in seconds
        'ext'     : the extension that succeeded (e.g. "atr")
    Returns (signal, None) if no annotation file is found.
    """
    sig = load_mitbih(record, channel, pb_dir)

    annotations = None
    try:
        import wfdb, os, inspect
        rdann_params = set(inspect.signature(wfdb.rdann).parameters.keys())
        has_pn_dir = "pn_dir" in rdann_params

        cache_dir  = os.path.join(os.path.expanduser("~"), "physionet-data", pb_dir)
        local_path = os.path.join(cache_dir, str(record))
        atr = None
        found_ext = None

        def _try_rdann(rec_path, ext, **kwargs):
            try:
                return wfdb.rdann(rec_path, ext, **kwargs)
            except Exception:
                return None

        for ext in ANN_EXTENSIONS:
            # 1. Local path as given
            atr = _try_rdann(str(record), ext)
            if atr: found_ext = ext; break

            # 2. Cached local path
            atr = _try_rdann(local_path, ext)
            if atr: found_ext = ext; break

            # 3. Remote via pn_dir (wfdb >= 4)
            if has_pn_dir:
                atr = _try_rdann(str(record), ext, pn_dir=pb_dir)
                if atr: found_ext = ext; break

            # 4. Remote via pb_dir (wfdb 3.x)
            if "pb_dir" in rdann_params:
                atr = _try_rdann(str(record), ext, pb_dir=pb_dir)
                if atr: found_ext = ext; break

        if atr is not None:
            annotations = {
                "sample": np.array(atr.sample),
                "symbol": list(atr.symbol),
                "time":   np.array(atr.sample) / sig.rate,
                "ext":    found_ext,
            }
    except Exception:
        pass

    return sig, annotations
