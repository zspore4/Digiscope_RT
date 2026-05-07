"""
filters/__init__.py  –  Filter registry.
Add new filters here to make them appear in the Add Filter dialog.
"""

from filters.signal_processing import (
    RemoveMeanFilter, AddNoiseFilter,
    FullWaveRectFilter, SquaringFilter, DerivativeDetectFilter,
)
from filters.beat_detector  import BeatDetectorFilter
from filters.pan_tompkins   import PanTompkinsFilter
from filters.ecg_filter     import ECGFilterFilter
from filters.spectral       import PowerSpectrumFilter, SpectrogramFilter
from filters.advanced       import (
    AveragingFilter, TemplateMatchFilter, ResampleFilter, CompressionFilter,
)
from filters.custom import CustomCoeffFilter, CustomPythonFilter
from filters.rrc_filter import RRCFilter
from filters.fft_filter import FFTBandFilter
from filters.goertzel_mwi import GoertzelFilter, MWIFilter
from filters.ecg_game import ECGGameFilter

FILTER_REGISTRY: list = [
    # ── Signal conditioning ──
    ("Remove Mean",               RemoveMeanFilter),
    ("Add Noise",                 AddNoiseFilter),
    ("ECG Filter (Butterworth/Notch)", ECGFilterFilter),
    ("Custom Filter Designer",  CustomCoeffFilter),
    ("Custom Python Filter",      CustomPythonFilter),
    ("Full-Wave Rectifier",       FullWaveRectFilter),
    ("Squaring",                  SquaringFilter),
    ("Derivative Detect",         DerivativeDetectFilter),
    ("Resample",                  ResampleFilter),
    # ── Beat detection ──
    ("Beat Detector (threshold)", BeatDetectorFilter),
    ("PT Thresholding",            PanTompkinsFilter),
    # ── Analysis ──
    ("Power Spectrum",            PowerSpectrumFilter),
    ("Spectrogram",               SpectrogramFilter),
    ("Template Matching",         TemplateMatchFilter),
    ("ECG Averaging",             AveragingFilter),
    ("ECG Compression (TP)",      CompressionFilter),
    # ── Pulse shaping ──
    ("FFT Band Filter",                 FFTBandFilter),
    ("Goertzel (single-freq power)",    GoertzelFilter),
    ("MWI (Moving Window Integrator)",  MWIFilter),
    ("RRC Filter (Root Raised Cosine)", RRCFilter),
    # ── Easter egg ──
    ("\u2665 ECG Runner",            ECGGameFilter),
]
