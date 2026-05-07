# DigiScope Python

A Python/PyQt5 port of the MATLAB DigiScope ECG signal analysis application.

---

## Project Structure

```
digiscope_python/
├── main.py                     # Entry point — run this
├── requirements.txt
│
├── core/
│   ├── signal.py               # Signal dataclass (replaces MATLAB signal struct)
│   ├── filter_base.py          # BaseFilter ABC + FilterResult dataclass
│   └── filter_chain.py         # Ordered filter chain engine
│
├── filters/
│   ├── __init__.py             # FILTER_REGISTRY — add new filters here
│   ├── source.py               # SourceFilter — wraps the raw loaded signal
│   ├── signal_processing.py    # RemoveMean, AddNoise, FullWaveRect,
│   │                           #   Squaring, DerivativeDetect
│   ├── beat_detector.py        # Threshold-based QRS detector
│   ├── pan_tompkins.py         # Pan-Tompkins adaptive QRS detector
│   ├── ecg_filter.py           # IIR/FIR filter (Butterworth, notch, …)
│   ├── spectral.py             # PowerSpectrum, Spectrogram
│   └── advanced.py             # Averaging, TemplateMatch, Resample, Compression
│
├── ui/
│   ├── app.py                  # MainWindow, PlotCanvas, helper dialogs
│   └── __init__.py
│
└── utils/
    ├── data_loader.py          # MIT-BIH (wfdb), text/CSV, npy, synthetic
    └── __init__.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `wfdb` requires an internet connection on first use to download
> MIT-BIH records from PhysioNet. Records are cached locally afterward.

### 2. Run the application

```bash
python main.py
```

---

## Loading Signals

### MIT-BIH Database (PhysioNet)

- **File → Load MIT-BIH Record…**
- Enter a record number (e.g. `100`, `101`, `200`, …)
- Channel 0 = MLII (standard limb lead), Channel 1 = V5
- Records download automatically on first use

### Text / CSV files

- **File → Open Text / CSV File…**
- One sample per line, or comma-separated rows
- You will be prompted for the sample rate

### NumPy arrays (`.npy`)

- Same dialog as Text/CSV — channel 0 is used if multi-channel

### Synthetic ECG

- **File → Generate Synthetic ECG…**
- Set heart rate, duration, and sample rate

---

## Using the Filter Chain

The filter chain works exactly like MATLAB Digiscope:

1. Load a signal — it becomes the first (source) step
2. **+ Add** a processing step from the catalogue
3. **⚙ Config** to change its settings
4. **▶ Run** to re-execute the entire chain
5. Steps can be **reordered** with ▲/▼
6. **Show in Plot** toggles subplot visibility

### Available Filters

| Filter | MATLAB equivalent | Description |
|---|---|---|
| Remove Mean | ECGremoveMean.m | Subtracts DC offset |
| Add Noise | ECGnoise.m | 60Hz / 50Hz / Gaussian noise |
| ECG Filter | ECGFilter.m | Butterworth LP/HP/BP/BS, notch |
| Full-Wave Rectifier | DSFullWave.m | `abs(x)` |
| Squaring | ECGsquare.m | `x²` |
| Derivative Detect | DSderivDetect.m | 1st+2nd derivative energy |
| Resample | ECGresample.m | Downsample to new rate |
| Beat Detector | ECGbeatDetector.m | Simple threshold QRS detect |
| Pan-Tompkins | ECGpantompkins.m | Adaptive threshold QRS detect |
| Power Spectrum | ECGpower.m | FFT magnitude, dB, windowed |
| Spectrogram | ECGspec.m | STFT spectrogram |
| Template Matching | ECGtemplate.m | Cross-correlation / difference |
| ECG Averaging | ECGaveraging.m | Noise averaging, SNR analysis |
| ECG Compression | ECGcompress.m | Turning-point compression, PRD |

---

## Navigation Controls

| Control | Action |
|---|---|
| `\|◀` / `▶\|` | Jump to start / end |
| `◀◀` / `▶▶` | Scroll one full window |
| `◀` / `▶`  | Scroll 1/10 window |
| `🔍+` / `🔍-` | Zoom in / out |
| **Window** field | Type `start_s  end_s` (e.g. `0  5`) |
| **Y-range** field | Type `min max` (e.g. `-2 2`), blank = auto |

---

## Adding a New Filter

1. Create a class in `filters/` that inherits from `BaseFilter`
2. Implement `configure(parent)`, `calculate(signal)`, `plot(axes, result)`
3. Register it in `filters/__init__.py` → `FILTER_REGISTRY`

Minimal template:

```python
from core.filter_base import BaseFilter, FilterResult
from core.signal import Signal
import numpy as np

class MyFilter(BaseFilter):
    name = "My Filter"
    passthrough = True
    num_plots = 1

    def configure(self, parent=None) -> bool:
        return False   # no settings

    def calculate(self, signal: Signal) -> FilterResult:
        d = signal.data * 2          # example: double amplitude
        t = np.arange(len(d)) / signal.rate
        return FilterResult(data=d, t=t, rate=signal.rate,
                            output_text="My Filter: done",
                            passthrough=True)

    def plot(self, axes, result):
        axes[0].clear()
        axes[0].plot(result.t, result.data)
        axes[0].set_title("My Filter")
```

Then in `filters/__init__.py`:
```python
from filters.my_filter import MyFilter
FILTER_REGISTRY.append(("My Filter", MyFilter))
```

---

## Key Design Decisions vs. MATLAB

| MATLAB | Python |
|---|---|
| `signal` struct passed between functions | `Signal` dataclass |
| `filtdat` struct | `FilterResult` dataclass |
| `switch(varargin{1})` dispatch | `BaseFilter` ABC with `calculate()`, `plot()`, `configure()` |
| `.dat` / `.hea` files | `wfdb` library downloads from PhysioNet directly |
| MATLAB `filter()` | `scipy.signal.lfilter()` / `sosfiltfilt()` |
| MATLAB `fft()` | `numpy.fft.fft()` |
| MATLAB `spectrogram()` | `scipy.signal.spectrogram()` |
| MATLAB `butter()` | `scipy.signal.butter()` |
| `inputdlg` / GUIDE GUI | PyQt5 `QDialog` + `QFormLayout` |
| Subplots managed by digiscope | `PlotCanvas.rebuild_axes(n)` |
