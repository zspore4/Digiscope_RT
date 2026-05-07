"""
realtime/  –  Real-Time DigiScope subsystem

Modules
-------
pico_driver.py   –  QThread that reads from the Pico over USB CDC
rt_buffer.py     –  Thread-safe circular ring buffer
rt_window.py     –  PyQt5 main window for real-time display

Entry points
------------
    python -m realtime.rt_window          # launch standalone window
    python -m realtime.pico_driver        # list ports / test driver
"""
from realtime.rt_window import launch, RTWindow
from realtime.pico_driver import PicoDriver, list_pico_ports
from realtime.rt_buffer import RTBuffer
