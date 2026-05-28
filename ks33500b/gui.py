"""
ks33500b/gui.py

Standalone PyQt5 GUI for the Keysight 33500B series waveform generator.

Launch directly:
    python -m ks33500b.gui

Tabs:
    Connection  — VISA string, scan, mode, connect/disconnect, *IDN?
    Channel 1   — function, freq, amp, offset, phase, load, output on/off
    Channel 2   — same as Channel 1, for CH2
    Burst       — N-cycle / gated burst (per channel)
    Sweep       — frequency sweep (per channel) with frequency-vs-time preview
    Arbitrary   — generate / load / upload arbitrary waveforms
"""

import sys
import time
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox,
    QDoubleSpinBox, QTextEdit, QTabWidget, QGridLayout, QFileDialog,
    QRadioButton, QButtonGroup,
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .controller import KS33500BController
from .driver import DEFAULT_VISA, FUNCTIONS, ARB_MAX_POINTS, ARB_MIN_POINTS
from .arbitrary import WaveformGenerator, compute_optimal_points


# ---------------------------------------------------------------------------
# Worker signals
# ---------------------------------------------------------------------------

class _Signals(QObject):
    status    = pyqtSignal(str)
    connected = pyqtSignal(bool, str)
    op_done   = pyqtSignal(str)
    scan_done = pyqtSignal(list)


class _ConnectWorker(QThread):
    def __init__(self, ctrl, signals):
        super().__init__()
        self._ctrl = ctrl; self._signals = signals

    def run(self):
        try:
            self._ctrl.connect()
            self._signals.connected.emit(True, self._ctrl.identify())
        except Exception as e:
            self._signals.connected.emit(False, str(e))


class _CallWorker(QThread):
    """Run a single (fn, args, kwargs) on the controller off the GUI thread."""

    def __init__(self, fn, args, kwargs, signals, label):
        super().__init__()
        self._fn = fn; self._args = args; self._kwargs = kwargs
        self._signals = signals; self._label = label

    def run(self):
        try:
            self._fn(*self._args, **self._kwargs)
            self._signals.op_done.emit(self._label)
        except Exception as e:
            self._signals.status.emit(f"{self._label} error: {e}")


class _ScanWorker(QThread):
    def __init__(self, signals):
        super().__init__()
        self._signals = signals

    def run(self):
        try:
            devices = KS33500BController.discover(filter_keysight=True)
        except Exception:
            devices = []
        self._signals.scan_done.emit(devices)


# ---------------------------------------------------------------------------
# Reusable channel control panel
# ---------------------------------------------------------------------------

class _ChannelPanel(QWidget):
    """Function/frequency/amplitude/offset/phase/output controls for one channel."""

    def __init__(self, channel: int, get_ctrl, signals, log_fn, parent=None):
        super().__init__(parent)
        self._channel  = channel
        self._get_ctrl = get_ctrl
        self._signals  = signals
        self._log_fn   = log_fn
        self._worker   = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)

        wbox = QGroupBox(f"CH{self._channel} Waveform")
        wg   = QGridLayout(wbox)

        wg.addWidget(QLabel("Function:"), 0, 0)
        self._fn_combo = QComboBox()
        self._fn_combo.addItems(["SIN", "SQU", "RAMP", "PULS", "NOIS", "DC", "ARB"])
        wg.addWidget(self._fn_combo, 0, 1)

        wg.addWidget(QLabel("Frequency (Hz):"), 1, 0)
        self._freq = QDoubleSpinBox()
        self._freq.setRange(1e-6, 30e6); self._freq.setDecimals(6)
        self._freq.setValue(1000.0)
        wg.addWidget(self._freq, 1, 1)

        wg.addWidget(QLabel("Amplitude (Vpp):"), 2, 0)
        self._amp = QDoubleSpinBox()
        self._amp.setRange(0.001, 20.0); self._amp.setDecimals(4)
        self._amp.setValue(0.1)
        wg.addWidget(self._amp, 2, 1)

        wg.addWidget(QLabel("Offset (V):"), 3, 0)
        self._offs = QDoubleSpinBox()
        self._offs.setRange(-10.0, 10.0); self._offs.setDecimals(4)
        self._offs.setValue(0.0)
        wg.addWidget(self._offs, 3, 1)

        wg.addWidget(QLabel("Phase (deg):"), 4, 0)
        self._phase = QDoubleSpinBox()
        self._phase.setRange(-360.0, 360.0); self._phase.setDecimals(2)
        self._phase.setValue(0.0)
        wg.addWidget(self._phase, 4, 1)

        wg.addWidget(QLabel("Output load:"), 5, 0)
        self._load_combo = QComboBox()
        self._load_combo.addItems(["50", "INF (High-Z)"])
        wg.addWidget(self._load_combo, 5, 1)

        lay.addWidget(wbox)

        # Pulse / square / ramp shape parameters
        pbox = QGroupBox(f"CH{self._channel} Shape Parameters")
        pg   = QGridLayout(pbox)

        pg.addWidget(QLabel("Square duty cycle (%):"), 0, 0)
        self._sq_dc = QDoubleSpinBox()
        self._sq_dc.setRange(0.01, 99.99); self._sq_dc.setDecimals(2)
        self._sq_dc.setValue(50.0)
        pg.addWidget(self._sq_dc, 0, 1)

        pg.addWidget(QLabel("Ramp symmetry (%):"), 1, 0)
        self._ramp_sym = QDoubleSpinBox()
        self._ramp_sym.setRange(0.0, 100.0); self._ramp_sym.setDecimals(2)
        self._ramp_sym.setValue(100.0)
        pg.addWidget(self._ramp_sym, 1, 1)

        pg.addWidget(QLabel("Pulse period (s):"), 2, 0)
        self._pper = QDoubleSpinBox()
        self._pper.setRange(1e-9, 1000.0); self._pper.setDecimals(9)
        self._pper.setValue(1e-3)
        pg.addWidget(self._pper, 2, 1)

        pg.addWidget(QLabel("Pulse width (s):"), 3, 0)
        self._pwid = QDoubleSpinBox()
        self._pwid.setRange(1e-9, 1000.0); self._pwid.setDecimals(9)
        self._pwid.setValue(5e-4)
        pg.addWidget(self._pwid, 3, 1)

        lay.addWidget(pbox)

        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply Settings")
        self._on_btn    = QPushButton("Output ON")
        self._off_btn   = QPushButton("Output OFF")
        for b in (self._apply_btn, self._on_btn, self._off_btn):
            b.setEnabled(False)
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

        self._status = QLabel(f"CH{self._channel} Output: OFF")
        self._status.setStyleSheet("color: red;")
        lay.addWidget(self._status)
        lay.addStretch()

        self._apply_btn.clicked.connect(self._on_apply)
        self._on_btn.clicked.connect(self._on_output_on)
        self._off_btn.clicked.connect(self._on_output_off)

    # --- API ---

    def set_enabled(self, enabled: bool):
        for b in (self._apply_btn, self._on_btn, self._off_btn):
            b.setEnabled(enabled)
        if not enabled:
            self._status.setText(f"CH{self._channel} Output: OFF")
            self._status.setStyleSheet("color: red;")

    # --- Slots ---

    def _do(self, fn, args=(), kwargs=None, label: str = ""):
        if kwargs is None:
            kwargs = {}
        w = _CallWorker(fn, args, kwargs, self._signals, label)
        w.start()
        self._worker = w

    def _on_apply(self):
        ctrl = self._get_ctrl()
        if ctrl is None: return
        ch  = self._channel
        fn  = self._fn_combo.currentText()
        f   = self._freq.value()
        a   = self._amp.value()
        o   = self._offs.value()
        ph  = self._phase.value()
        load = self._load_combo.currentText()
        load_arg: float | str = "INF" if load.startswith("INF") else 50.0
        sq_dc   = self._sq_dc.value()
        ramp_sym = self._ramp_sym.value()
        pper = self._pper.value()
        pwid = self._pwid.value()

        def _go():
            ctrl.set_load(load_arg, channel=ch)
            if fn == "SIN":
                ctrl.apply_sine(f, a, o, ph, channel=ch)
            elif fn == "SQU":
                ctrl.apply_square(f, a, o, ph, duty_cycle=sq_dc, channel=ch)
            elif fn == "RAMP":
                ctrl.apply_ramp(f, a, o, ph, symmetry=ramp_sym, channel=ch)
            elif fn == "PULS":
                ctrl.apply_pulse(f, a, o, ph, channel=ch)
                ctrl.configure_pulse(period_s=pper, width_s=pwid, channel=ch)
            elif fn == "NOIS":
                ctrl.apply_noise(a, o, channel=ch)
            elif fn == "DC":
                ctrl.apply_dc(o, channel=ch)
            elif fn == "ARB":
                ctrl.apply_arbitrary(f, a, o, channel=ch)

        self._do(_go, label=f"CH{ch} apply")

    def _on_output_on(self):
        ctrl = self._get_ctrl()
        if ctrl is None: return
        ch = self._channel
        self._do(lambda: ctrl.output_on(ch), label=f"CH{ch} output ON")
        self._status.setText(f"CH{ch} Output: ON")
        self._status.setStyleSheet("color: green;")

    def _on_output_off(self):
        ctrl = self._get_ctrl()
        if ctrl is None: return
        ch = self._channel
        self._do(lambda: ctrl.output_off(ch), label=f"CH{ch} output OFF")
        self._status.setText(f"CH{ch} Output: OFF")
        self._status.setStyleSheet("color: red;")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class KS33500BWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keysight 33500B AWG Control")
        self.resize(960, 760)

        self._ctrl:    KS33500BController | None = None
        self._signals = _Signals()
        self._worker  = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        tabs = QTabWidget()
        tabs.addTab(self._build_connection_tab(), "Connection")
        self._ch1_panel = _ChannelPanel(1, lambda: self._ctrl, self._signals, self._log_msg)
        self._ch2_panel = _ChannelPanel(2, lambda: self._ctrl, self._signals, self._log_msg)
        tabs.addTab(self._ch1_panel, "Channel 1")
        tabs.addTab(self._ch2_panel, "Channel 2")
        tabs.addTab(self._build_burst_tab(),  "Burst")
        tabs.addTab(self._build_sweep_tab(),  "Sweep")
        tabs.addTab(self._build_arb_tab(),    "Arbitrary")

        lay.addWidget(tabs)
        lay.addWidget(self._build_log())

    def _build_connection_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox("Instrument Connection")
        g   = QGridLayout(box)

        # Scan / manual mode
        self._mode_scan = QRadioButton("Scan for devices")
        self._mode_man  = QRadioButton("Manual VISA address")
        self._mode_scan.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._mode_scan)
        self._mode_group.addButton(self._mode_man)
        g.addWidget(self._mode_scan, 0, 0)
        g.addWidget(self._mode_man,  0, 1)

        g.addWidget(QLabel("Device:"), 1, 0)
        self._device_combo = QComboBox()
        self._device_combo.setEditable(False)
        self._device_combo.setMinimumWidth(420)
        g.addWidget(self._device_combo, 1, 1)
        self._scan_btn = QPushButton("Scan")
        g.addWidget(self._scan_btn, 1, 2)

        g.addWidget(QLabel("VISA Resource:"), 2, 0)
        self._visa_edit = QLineEdit(DEFAULT_VISA)
        g.addWidget(self._visa_edit, 2, 1, 1, 2)

        g.addWidget(QLabel("Mode:"), 3, 0)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["simulation", "hardware"])
        g.addWidget(self._mode_combo, 3, 1)

        btn_row = QHBoxLayout()
        self._connect_btn    = QPushButton("Connect")
        self._disconnect_btn = QPushButton("Disconnect")
        self._test_btn       = QPushButton("Test")
        self._reset_btn      = QPushButton("*RST")
        self._beep_btn       = QPushButton("Beep")
        self._disconnect_btn.setEnabled(False)
        self._reset_btn.setEnabled(False)
        self._beep_btn.setEnabled(False)
        btn_row.addWidget(self._connect_btn)
        btn_row.addWidget(self._disconnect_btn)
        btn_row.addWidget(self._test_btn)
        btn_row.addWidget(self._reset_btn)
        btn_row.addWidget(self._beep_btn)
        g.addLayout(btn_row, 4, 0, 1, 3)

        self._conn_label = QLabel("Not connected")
        self._conn_label.setStyleSheet("color: red; font-weight: bold;")
        g.addWidget(self._conn_label, 5, 0, 1, 3)

        lay.addWidget(box)
        lay.addStretch()

        self._scan_btn.clicked.connect(self._on_scan)
        self._mode_scan.toggled.connect(self._on_mode_changed)
        self._device_combo.currentIndexChanged.connect(self._on_device_picked)
        return w

    def _build_burst_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox("Burst Mode")
        g   = QGridLayout(box)

        g.addWidget(QLabel("Channel:"), 0, 0)
        self._burst_ch = QComboBox(); self._burst_ch.addItems(["1", "2"])
        g.addWidget(self._burst_ch, 0, 1)

        g.addWidget(QLabel("Mode:"), 1, 0)
        self._burst_mode = QComboBox()
        self._burst_mode.addItems(["TRIG", "GAT"])
        g.addWidget(self._burst_mode, 1, 1)

        g.addWidget(QLabel("N cycles:"), 2, 0)
        self._burst_n = QSpinBox()
        self._burst_n.setRange(1, 1_000_000); self._burst_n.setValue(1)
        g.addWidget(self._burst_n, 2, 1)

        g.addWidget(QLabel("Initial phase (deg):"), 3, 0)
        self._burst_ph = QDoubleSpinBox()
        self._burst_ph.setRange(-360.0, 360.0); self._burst_ph.setValue(0.0)
        g.addWidget(self._burst_ph, 3, 1)

        g.addWidget(QLabel("Trigger source:"), 4, 0)
        self._burst_trig = QComboBox()
        self._burst_trig.addItems(["IMM", "EXT", "TIM", "BUS"])
        g.addWidget(self._burst_trig, 4, 1)

        btn_row = QHBoxLayout()
        self._burst_apply = QPushButton("Enable Burst")
        self._burst_off   = QPushButton("Disable Burst")
        self._burst_trgbtn = QPushButton("Trigger Now")
        self._burst_busbtn = QPushButton("*TRG (BUS)")
        for b in (self._burst_apply, self._burst_off,
                  self._burst_trgbtn, self._burst_busbtn):
            b.setEnabled(False)
            btn_row.addWidget(b)
        g.addLayout(btn_row, 5, 0, 1, 2)
        lay.addWidget(box)
        lay.addStretch()

        self._burst_apply.clicked.connect(self._on_burst_apply)
        self._burst_off.clicked.connect(self._on_burst_off)
        self._burst_trgbtn.clicked.connect(self._on_burst_trigger)
        self._burst_busbtn.clicked.connect(self._on_burst_bustrigger)
        return w

    def _build_sweep_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox("Frequency Sweep")
        g   = QGridLayout(box)

        g.addWidget(QLabel("Channel:"), 0, 0)
        self._sw_ch = QComboBox(); self._sw_ch.addItems(["1", "2"])
        g.addWidget(self._sw_ch, 0, 1)

        g.addWidget(QLabel("Start (Hz):"), 1, 0)
        self._sw_start = QDoubleSpinBox()
        self._sw_start.setRange(1e-6, 30e6); self._sw_start.setDecimals(3)
        self._sw_start.setValue(100e3)
        g.addWidget(self._sw_start, 1, 1)

        g.addWidget(QLabel("Stop (Hz):"), 2, 0)
        self._sw_stop = QDoubleSpinBox()
        self._sw_stop.setRange(1e-6, 30e6); self._sw_stop.setDecimals(3)
        self._sw_stop.setValue(700e3)
        g.addWidget(self._sw_stop, 2, 1)

        g.addWidget(QLabel("Time (s):"), 3, 0)
        self._sw_time = QDoubleSpinBox()
        self._sw_time.setRange(1e-3, 8000.0); self._sw_time.setDecimals(4)
        self._sw_time.setValue(0.1)
        g.addWidget(self._sw_time, 3, 1)

        g.addWidget(QLabel("Return time (s):"), 4, 0)
        self._sw_rtime = QDoubleSpinBox()
        self._sw_rtime.setRange(0.0, 8000.0); self._sw_rtime.setDecimals(4)
        self._sw_rtime.setValue(0.0)
        g.addWidget(self._sw_rtime, 4, 1)

        g.addWidget(QLabel("Hold start (s):"), 5, 0)
        self._sw_hstart = QDoubleSpinBox()
        self._sw_hstart.setRange(0.0, 8000.0); self._sw_hstart.setDecimals(4)
        g.addWidget(self._sw_hstart, 5, 1)

        g.addWidget(QLabel("Hold stop (s):"), 6, 0)
        self._sw_hstop = QDoubleSpinBox()
        self._sw_hstop.setRange(0.0, 8000.0); self._sw_hstop.setDecimals(4)
        g.addWidget(self._sw_hstop, 6, 1)

        g.addWidget(QLabel("Spacing:"), 7, 0)
        self._sw_spc = QComboBox(); self._sw_spc.addItems(["LIN", "LOG"])
        g.addWidget(self._sw_spc, 7, 1)

        g.addWidget(QLabel("Trigger source:"), 8, 0)
        self._sw_trig = QComboBox()
        self._sw_trig.addItems(["IMM", "EXT", "TIM", "BUS"])
        g.addWidget(self._sw_trig, 8, 1)

        btn_row = QHBoxLayout()
        self._sw_apply   = QPushButton("Enable Sweep")
        self._sw_off     = QPushButton("Disable Sweep")
        self._sw_preview = QPushButton("Preview")
        for b in (self._sw_apply, self._sw_off, self._sw_preview):
            b.setEnabled(False)
            btn_row.addWidget(b)
        self._sw_preview.setEnabled(True)   # preview is offline
        g.addLayout(btn_row, 9, 0, 1, 2)
        lay.addWidget(box)

        if HAS_MPL:
            self._sw_fig    = Figure(figsize=(7, 2.5))
            self._sw_canvas = FigureCanvas(self._sw_fig)
            self._sw_ax     = self._sw_fig.add_subplot(111)
            self._sw_ax.set_xlabel("Time (s)")
            self._sw_ax.set_ylabel("Frequency (Hz)")
            self._sw_ax.set_title("Sweep preview")
            self._sw_ax.grid(True, alpha=0.3)
            lay.addWidget(self._sw_canvas)

        lay.addStretch()

        self._sw_apply.clicked.connect(self._on_sweep_apply)
        self._sw_off.clicked.connect(self._on_sweep_off)
        self._sw_preview.clicked.connect(self._on_sweep_preview)
        return w

    def _build_arb_tab(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        box = QGroupBox("Arbitrary Waveform")
        g   = QGridLayout(box)

        g.addWidget(QLabel("Generator:"), 0, 0)
        self._arb_kind = QComboBox()
        self._arb_kind.addItems([
            "Sine", "Square", "Triangle", "Pulse",
            "Gaussian", "Sinc", "Exponential decay",
            "Frequency comb",
        ])
        g.addWidget(self._arb_kind, 0, 1)

        g.addWidget(QLabel("Name (≤12 chars):"), 1, 0)
        self._arb_name = QLineEdit("ARB1")
        self._arb_name.setMaxLength(12)
        g.addWidget(self._arb_name, 1, 1)

        g.addWidget(QLabel("Output frequency (Hz):"), 2, 0)
        self._arb_freq = QDoubleSpinBox()
        self._arb_freq.setRange(1e-6, 30e6); self._arb_freq.setDecimals(3)
        self._arb_freq.setValue(1000.0)
        g.addWidget(self._arb_freq, 2, 1)

        g.addWidget(QLabel("Comb frequencies (CSV Hz, comb only):"), 3, 0)
        self._arb_comb = QLineEdit("100, 200, 300, 400, 500")
        g.addWidget(self._arb_comb, 3, 1)

        g.addWidget(QLabel("Apply to channel:"), 4, 0)
        self._arb_ch = QComboBox(); self._arb_ch.addItems(["1", "2"])
        g.addWidget(self._arb_ch, 4, 1)

        btn_row = QHBoxLayout()
        self._arb_gen_btn   = QPushButton("Generate + Upload")
        self._arb_file_btn  = QPushButton("Load from .csv/.npy")
        self._arb_play_btn  = QPushButton("Apply ARB (current)")
        for b in (self._arb_gen_btn, self._arb_file_btn, self._arb_play_btn):
            b.setEnabled(False)
            btn_row.addWidget(b)
        g.addLayout(btn_row, 5, 0, 1, 2)
        lay.addWidget(box)

        if HAS_MPL:
            self._arb_fig    = Figure(figsize=(7, 2.5))
            self._arb_canvas = FigureCanvas(self._arb_fig)
            self._arb_ax     = self._arb_fig.add_subplot(111)
            self._arb_ax.set_xlabel("Sample"); self._arb_ax.set_ylabel("Amplitude")
            self._arb_ax.set_title("Arbitrary waveform preview")
            self._arb_ax.grid(True, alpha=0.3)
            lay.addWidget(self._arb_canvas)

        self._arb_gen_btn.clicked.connect(self._on_arb_generate)
        self._arb_file_btn.clicked.connect(self._on_arb_file)
        self._arb_play_btn.clicked.connect(self._on_arb_play)

        return w

    def _build_log(self) -> QWidget:
        box = QGroupBox("Status Log")
        lay = QVBoxLayout(box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        self._log.setFont(QFont("Courier", 9))
        lay.addWidget(self._log)
        return box

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._connect_btn.clicked.connect(self._on_connect)
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._test_btn.clicked.connect(self._on_test)
        self._reset_btn.clicked.connect(self._on_reset)
        self._beep_btn.clicked.connect(self._on_beep)
        self._signals.connected.connect(self._on_connect_result)
        self._signals.status.connect(self._log_msg)
        self._signals.op_done.connect(lambda label: self._log_msg(f"OK: {label}"))
        self._signals.scan_done.connect(self._on_scan_result)

    # ------------------------------------------------------------------
    # Connection slots
    # ------------------------------------------------------------------

    def _set_action_buttons_enabled(self, enabled: bool):
        self._ch1_panel.set_enabled(enabled)
        self._ch2_panel.set_enabled(enabled)
        for b in (self._burst_apply, self._burst_off,
                  self._burst_trgbtn, self._burst_busbtn,
                  self._sw_apply, self._sw_off,
                  self._arb_gen_btn, self._arb_file_btn, self._arb_play_btn,
                  self._reset_btn, self._beep_btn):
            b.setEnabled(enabled)

    def _on_mode_changed(self):
        scan = self._mode_scan.isChecked()
        self._device_combo.setEnabled(scan)
        self._scan_btn.setEnabled(scan)
        self._visa_edit.setEnabled(not scan)

    def _on_scan(self):
        if self._mode_combo.currentText() == "simulation":
            self._log_msg("Scan skipped (simulation mode).")
            return
        self._scan_btn.setEnabled(False)
        self._log_msg("Scanning VISA bus for Keysight 33500B...")
        w = _ScanWorker(self._signals)
        w.start(); self._worker = w

    def _on_scan_result(self, devices: list):
        self._scan_btn.setEnabled(True)
        self._device_combo.clear()
        if not devices:
            self._log_msg("No 33500B devices found.")
            return
        for res, idn in devices:
            self._device_combo.addItem(f"{idn}  [{res}]", userData=res)
        self._log_msg(f"Found {len(devices)} device(s).")
        # Auto-pick the first
        if self._device_combo.count() > 0:
            self._device_combo.setCurrentIndex(0)

    def _on_device_picked(self, idx: int):
        if idx < 0:
            return
        res = self._device_combo.itemData(idx)
        if res:
            self._visa_edit.setText(res)

    def _on_connect(self):
        if self._mode_scan.isChecked():
            idx = self._device_combo.currentIndex()
            if idx >= 0:
                self._visa_edit.setText(self._device_combo.itemData(idx) or self._visa_edit.text())
        self._ctrl = KS33500BController(
            visa=self._visa_edit.text().strip(),
            mode=self._mode_combo.currentText(),
        )
        self._log_msg("Connecting...")
        self._connect_btn.setEnabled(False)
        w = _ConnectWorker(self._ctrl, self._signals)
        w.start(); self._worker = w

    def _on_connect_result(self, ok: bool, msg: str):
        self._connect_btn.setEnabled(True)
        if ok:
            self._conn_label.setText(f"Connected: {msg}")
            self._conn_label.setStyleSheet("color: green; font-weight: bold;")
            self._disconnect_btn.setEnabled(True)
            self._set_action_buttons_enabled(True)
            self._log_msg(f"Connected: {msg}")
        else:
            self._conn_label.setText("Failed")
            self._conn_label.setStyleSheet("color: red; font-weight: bold;")
            self._ctrl = None
            self._log_msg(f"FAILED: {msg}")

    def _on_disconnect(self):
        if self._ctrl:
            try:
                self._ctrl.disconnect()
            except Exception as e:
                self._log_msg(f"Disconnect: {e}")
            self._ctrl = None
        self._conn_label.setText("Not connected")
        self._conn_label.setStyleSheet("color: red; font-weight: bold;")
        self._disconnect_btn.setEnabled(False)
        self._set_action_buttons_enabled(False)
        self._log_msg("Disconnected.")

    def _on_test(self):
        config = {"visa": self._visa_edit.text().strip(),
                  "mode": self._mode_combo.currentText()}

        class _T(QThread):
            done = pyqtSignal(bool, str)
            def run(self_):
                ok, msg = KS33500BController.test(config)
                self_.done.emit(ok, msg)
        t = _T(self)
        t.done.connect(lambda ok, m: self._log_msg(f"Test {'OK' if ok else 'FAILED'}: {m}"))
        t.start(); self._worker = t

    def _on_reset(self):
        if self._ctrl is None: return
        try:
            self._ctrl.reset()
            self._log_msg("Reset (*RST) done.")
        except Exception as e:
            self._log_msg(f"Reset error: {e}")

    def _on_beep(self):
        if self._ctrl is None: return
        try:
            self._ctrl.beep()
            self._log_msg("Beep.")
        except Exception as e:
            self._log_msg(f"Beep error: {e}")

    # ------------------------------------------------------------------
    # Burst slots
    # ------------------------------------------------------------------

    def _on_burst_apply(self):
        if self._ctrl is None: return
        ch = int(self._burst_ch.currentText())
        kwargs = dict(
            ncycles  = self._burst_n.value(),
            mode     = self._burst_mode.currentText(),
            phase_deg= self._burst_ph.value(),
            trigger  = self._burst_trig.currentText(),
            channel  = ch,
        )
        w = _CallWorker(self._ctrl.enable_burst, (), kwargs,
                        self._signals, f"CH{ch} burst enable")
        w.start(); self._worker = w

    def _on_burst_off(self):
        if self._ctrl is None: return
        ch = int(self._burst_ch.currentText())
        w = _CallWorker(self._ctrl.disable_burst, (ch,), {},
                        self._signals, f"CH{ch} burst disable")
        w.start(); self._worker = w

    def _on_burst_trigger(self):
        if self._ctrl is None: return
        ch = int(self._burst_ch.currentText())
        try:
            self._ctrl.trigger(channel=ch)
            self._log_msg(f"CH{ch} TRIG:IMM sent.")
        except Exception as e:
            self._log_msg(f"Trigger error: {e}")

    def _on_burst_bustrigger(self):
        if self._ctrl is None: return
        try:
            self._ctrl.bus_trigger()
            self._log_msg("*TRG sent.")
        except Exception as e:
            self._log_msg(f"*TRG error: {e}")

    # ------------------------------------------------------------------
    # Sweep slots
    # ------------------------------------------------------------------

    def _on_sweep_apply(self):
        if self._ctrl is None: return
        ch = int(self._sw_ch.currentText())
        kwargs = dict(
            start_hz    = self._sw_start.value(),
            stop_hz     = self._sw_stop.value(),
            time_s      = self._sw_time.value(),
            spacing     = self._sw_spc.currentText(),
            return_time = self._sw_rtime.value(),
            hold_start  = self._sw_hstart.value(),
            hold_stop   = self._sw_hstop.value(),
            trigger     = self._sw_trig.currentText(),
            channel     = ch,
        )
        w = _CallWorker(self._ctrl.enable_sweep, (), kwargs,
                        self._signals, f"CH{ch} sweep enable")
        w.start(); self._worker = w

    def _on_sweep_off(self):
        if self._ctrl is None: return
        ch = int(self._sw_ch.currentText())
        w = _CallWorker(self._ctrl.disable_sweep, (ch,), {},
                        self._signals, f"CH{ch} sweep disable")
        w.start(); self._worker = w

    def _on_sweep_preview(self):
        if not HAS_MPL:
            self._log_msg("matplotlib not available — preview disabled.")
            return
        start = self._sw_start.value()
        stop  = self._sw_stop.value()
        tsw   = max(1e-6, self._sw_time.value())
        rtime = self._sw_rtime.value()
        hs    = self._sw_hstart.value()
        hp    = self._sw_hstop.value()
        spc   = self._sw_spc.currentText()

        t_segs, f_segs = [], []
        t0 = 0.0
        if hs > 0:
            t_segs.append(np.array([t0, t0 + hs]))
            f_segs.append(np.array([start, start]))
            t0 += hs
        n = 300
        ts = np.linspace(t0, t0 + tsw, n)
        frac = (ts - t0) / tsw
        if spc.upper().startswith("LOG") and start > 0 and stop > 0:
            fs = start * (stop / start) ** frac
        else:
            fs = start + (stop - start) * frac
        t_segs.append(ts); f_segs.append(fs)
        t0 += tsw
        if hp > 0:
            t_segs.append(np.array([t0, t0 + hp]))
            f_segs.append(np.array([stop, stop]))
            t0 += hp
        if rtime > 0:
            tr = np.linspace(t0, t0 + rtime, n)
            frac_r = (tr - t0) / rtime
            if spc.upper().startswith("LOG") and start > 0 and stop > 0:
                fr = stop * (start / stop) ** frac_r
            else:
                fr = stop + (start - stop) * frac_r
            t_segs.append(tr); f_segs.append(fr)

        t_all = np.concatenate(t_segs)
        f_all = np.concatenate(f_segs)

        self._sw_ax.clear()
        self._sw_ax.plot(t_all, f_all, color="#1f77b4", linewidth=1.5)
        self._sw_ax.set_xlabel("Time (s)")
        self._sw_ax.set_ylabel("Frequency (Hz)")
        self._sw_ax.set_title("Sweep preview")
        self._sw_ax.grid(True, alpha=0.3)
        if spc.upper().startswith("LOG"):
            try:
                self._sw_ax.set_yscale("log")
            except Exception:
                pass
        self._sw_fig.tight_layout()
        self._sw_canvas.draw()

    # ------------------------------------------------------------------
    # Arbitrary slots
    # ------------------------------------------------------------------

    def _on_arb_generate(self):
        if self._ctrl is None: return
        kind = self._arb_kind.currentText()
        name = self._arb_name.text().strip() or "ARB1"
        freq = self._arb_freq.value()
        ch   = int(self._arb_ch.currentText())

        try:
            if kind == "Sine":
                wf = WaveformGenerator.sine(freq, name=name)
            elif kind == "Square":
                wf = WaveformGenerator.square(freq, 50.0, name=name)
            elif kind == "Triangle":
                wf = WaveformGenerator.ramp(freq, symmetry=50.0, name=name)
            elif kind == "Pulse":
                wf = WaveformGenerator.pulse(freq, 50.0, 0.01, 0.01, name=name)
            elif kind == "Gaussian":
                wf = WaveformGenerator.gaussian(freq, 0.15, name=name)
            elif kind == "Sinc":
                wf = WaveformGenerator.sinc(freq, 4, name=name)
            elif kind == "Exponential decay":
                wf = WaveformGenerator.exponential(freq, 0.2, decay=True, name=name)
            elif kind == "Frequency comb":
                tones = [float(t.strip()) for t in self._arb_comb.text().split(",")
                         if t.strip()]
                if not tones:
                    raise ValueError("Provide ≥1 comb frequency")
                wf = WaveformGenerator.frequency_comb(
                    tones, monte_carlo_iter=500, name=name)
                freq = wf.frequency
            else:
                raise ValueError(f"Unknown generator: {kind}")
        except Exception as e:
            self._log_msg(f"Generator error: {e}")
            return

        try:
            self._ctrl.load_arbitrary(
                wf.name, wf.data, channel=ch, binary=True, select=True,
                sample_rate=wf.sample_rate if wf.sample_rate > 0 else None)
            self._log_msg(
                f"Uploaded {wf.num_points}-pt {kind!r} as {wf.name!r} on CH{ch} "
                f"(f0={wf.frequency:.4g} Hz, sr={wf.sample_rate/1e6:.3g} MSa/s).")
            self._plot_arb(wf.data, f"{kind}: {wf.name}")
        except Exception as e:
            self._log_msg(f"Upload error: {e}")

    def _on_arb_file(self):
        if self._ctrl is None: return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load arbitrary waveform", "",
            "Waveform (*.csv *.npy *.txt);;All files (*)"
        )
        if not path: return
        try:
            if path.lower().endswith(".npy"):
                samples = np.load(path)
            else:
                samples = np.loadtxt(path, delimiter=",")
            samples = np.asarray(samples).flatten()
            n = samples.size
            if n < ARB_MIN_POINTS or n > ARB_MAX_POINTS:
                raise ValueError(
                    f"Length must be {ARB_MIN_POINTS}..{ARB_MAX_POINTS}, got {n}")
            name = self._arb_name.text().strip() or "ARB1"
            ch   = int(self._arb_ch.currentText())
            self._ctrl.load_arbitrary(name, samples, channel=ch,
                                      binary=True, select=True)
            self._log_msg(
                f"Loaded {n}-pt waveform from {path!r} as {name!r} on CH{ch}.")
            self._plot_arb(samples, f"From file: {name}")
        except Exception as e:
            self._log_msg(f"File load error: {e}")

    def _on_arb_play(self):
        if self._ctrl is None: return
        ch   = int(self._arb_ch.currentText())
        freq = self._arb_freq.value()
        try:
            self._ctrl.apply_arbitrary(freq, 0.1, 0.0, channel=ch)
            self._log_msg(f"CH{ch}: APPLy:ARB at {freq:g} Hz.")
        except Exception as e:
            self._log_msg(f"Apply ARB error: {e}")

    def _plot_arb(self, samples: np.ndarray, title: str):
        if not HAS_MPL:
            return
        self._arb_ax.clear()
        self._arb_ax.plot(samples, lw=1)
        self._arb_ax.set_xlabel("Sample"); self._arb_ax.set_ylabel("Amplitude")
        self._arb_ax.set_title(title)
        self._arb_ax.grid(True, alpha=0.3)
        self._arb_fig.tight_layout()
        self._arb_canvas.draw()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _log_msg(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        if self._ctrl:
            try: self._ctrl.disconnect()
            except Exception: pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    win = KS33500BWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
