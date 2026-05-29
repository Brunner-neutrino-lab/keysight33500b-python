"""
ks33500b/driver.py

Low-level SCPI interface to the Keysight 33500B family of function/arbitrary
waveform generators (33511B, 33512B, 33521B, 33522B and the A variants).

Handles only instrument communication — no experiment logic, no Qt.

Two modes:
  "hardware"   — connects via pyvisa (USB or TCPIP)
  "simulation" — tracks state in memory for development without hardware

SCPI commands follow the 33500B Programming Reference. The 33500B uses the
:SOURce<n>: subtree to address per-channel parameters; one- and two-channel
models are both supported (channel 2 calls are silent no-ops on a 1-channel
instrument, just as the box itself ignores them).

Channel convention:
    channel = 1 -> :SOURce1:...
    channel = 2 -> :SOURce2:...
"""

import time
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VISA = "USB0::0x0957::0x2C07::MY00000000::INSTR"
TIMEOUT_MS   = 10_000

# Substrings expected in the *IDN? response.
IDN_VENDORS = ("Keysight Technologies", "Agilent Technologies")
IDN_MODELS  = ("33511", "33512", "33521", "33522", "335")

# Function names (as used by the SCPI FUNCtion command).
FUNCTIONS = ("SIN", "SQU", "RAMP", "PULS", "NOIS", "DC", "ARB")

# Trigger sources for burst/sweep.
TRIG_SOURCES = ("IMM", "EXT", "TIM", "BUS")

# Burst modes.
BURST_MODES = ("TRIG", "GAT")

# Sweep spacing.
SWEEP_SPACINGS = ("LIN", "LOG")

# Hardware limits (33500B standard memory, 14-bit DAC).
ARB_MIN_POINTS = 8
ARB_MAX_POINTS = 65_536
ARB_DAC_MIN    = 0
ARB_DAC_MAX    = 16_383          # 14-bit unsigned
ARB_MIN_RATE   = 1               # Sa/s
ARB_MAX_RATE   = 250_000_000     # 250 MSa/s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FUNC_SCPI_LONG = {
    "SIN":  "SINusoid",
    "SQU":  "SQUare",
    "RAMP": "RAMP",
    "PULS": "PULSe",
    "NOIS": "NOISe",
    "DC":   "DC",
    "ARB":  "ARB",
}


def _check_channel(channel: int) -> int:
    if channel not in (1, 2):
        raise ValueError(f"channel must be 1 or 2, got {channel}")
    return channel


def _norm_func(function: str) -> str:
    fn = function.upper()
    # accept long-form aliases
    aliases = {"SINUSOID": "SIN", "SINE": "SIN",
               "SQUARE":   "SQU",
               "PULSE":    "PULS",
               "NOISE":    "NOIS",
               "USER":     "ARB", "ARBITRARY": "ARB"}
    fn = aliases.get(fn, fn)
    if fn not in FUNCTIONS:
        raise ValueError(f"function must be one of {FUNCTIONS}, got {function!r}")
    return fn


def _norm_trigger(source: str) -> str:
    s = source.upper()
    s = {"IMMEDIATE": "IMM", "EXTERNAL": "EXT", "TIMER": "TIM"}.get(s, s)
    if s not in TRIG_SOURCES:
        raise ValueError(f"trigger source must be IMM/EXT/TIM/BUS, got {source!r}")
    return s


def _norm_burst_mode(mode: str) -> str:
    m = mode.upper()
    m = {"TRIGGERED": "TRIG", "GATED": "GAT"}.get(m, m)
    if m not in BURST_MODES:
        raise ValueError(f"burst mode must be TRIG or GAT, got {mode!r}")
    return m


def _norm_spacing(spacing: str) -> str:
    s = spacing.upper()
    s = {"LINEAR": "LIN", "LOGARITHMIC": "LOG"}.get(s, s)
    if s not in SWEEP_SPACINGS:
        raise ValueError(f"spacing must be LIN or LOG, got {spacing!r}")
    return s


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class KS33500BDriver:
    """
    Low-level SCPI driver for Keysight 33500B series waveform generators.

    Parameters
    ----------
    visa : str
        VISA resource string (USB or TCPIP).
    mode : str
        "hardware" or "simulation".
    timeout_ms : int
        VISA timeout in milliseconds (hardware mode only).
    """

    def __init__(self, visa: str = DEFAULT_VISA, mode: str = "simulation",
                 timeout_ms: int = TIMEOUT_MS):
        if mode not in ("hardware", "simulation"):
            raise ValueError(f"mode must be 'hardware' or 'simulation', got {mode!r}")
        self._visa_str   = visa
        self._mode       = mode
        self._timeout_ms = timeout_ms
        self._inst       = None
        self._rm         = None
        self._connected  = False

        # Per-channel simulated state.
        self._sim_state: dict[int, dict] = {
            ch: {
                "function":      "SIN",
                "frequency":     1000.0,
                "amplitude":     0.1,
                "offset":        0.0,
                "phase":         0.0,
                "output":        False,
                "load":          50.0,        # ohm or float('inf') for high-Z
                "polarity":      "NORM",
                "sync_output":   True,
                "duty_cycle":    50.0,        # square
                "symmetry":      100.0,       # ramp (rising)
                "pulse_width":   100e-6,
                "pulse_period":  1e-3,
                "pulse_dcyc":    50.0,
                "pulse_rise":    8.4e-9,
                "pulse_fall":    8.4e-9,
                "arb_name":      "",          # selected arb waveform
                "arb_srate":     1.0e6,
                "trig_source":   "IMM",
                # Burst
                "burst_state":   False,
                "burst_mode":    "TRIG",
                "burst_ncycles": 1,
                "burst_phase":   0.0,
                # Sweep
                "sweep_state":   False,
                "sweep_spacing": "LIN",
                "sweep_time":    1.0,
                "sweep_rtime":   0.0,
                "sweep_hstart":  0.0,
                "sweep_hstop":   0.0,
                "freq_start":    100.0,
                "freq_stop":     1000.0,
            } for ch in (1, 2)
        }

        # Volatile arbitrary waveform buffers, keyed by name (shared catalog).
        self._sim_arb: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if self._connected:
            return
        if self._mode == "hardware":
            self._connect_hardware()
        self._connected = True

    def disconnect(self):
        if not self._connected:
            return
        if self._mode == "hardware":
            self._disconnect_hardware()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def visa_resource(self) -> str:
        return self._visa_str

    def identify(self) -> str:
        if self._mode == "hardware":
            return self._inst.query("*IDN?").strip()
        return "Keysight Technologies,33522B,MY[sim]0000001,3.03-1.19-2.00-58-00"

    def reset(self):
        if self._mode == "hardware":
            self._inst.write("*RST")
            self._inst.query("*OPC?")
        else:
            visa, mode, tmo = self._visa_str, self._mode, self._timeout_ms
            self.__init__(visa=visa, mode=mode, timeout_ms=tmo)
            self._connected = True

    def clear_status(self):
        if self._mode == "hardware":
            self._inst.write("*CLS")

    def get_error(self) -> str:
        if self._mode == "hardware":
            return self._inst.query(":SYSTem:ERRor?").strip()
        return '+0,"No error"'

    def beep(self):
        if self._mode == "hardware":
            self._inst.write(":SYSTem:BEEPer:IMMediate")

    def wait_complete(self) -> bool:
        if self._mode == "hardware":
            return self._inst.query("*OPC?").strip() == "1"
        return True

    # ------------------------------------------------------------------
    # Raw I/O
    # ------------------------------------------------------------------

    def write(self, cmd: str):
        if self._mode == "hardware":
            self._inst.write(cmd)

    def query(self, cmd: str) -> str:
        if self._mode == "hardware":
            return self._inst.query(cmd).strip()
        return ""

    def write_raw(self, data: bytes):
        if self._mode == "hardware":
            self._inst.write_raw(data)

    # ------------------------------------------------------------------
    # APPLy — combined function/freq/amp/offset/phase
    # ------------------------------------------------------------------

    def apply(self,
              function:  str,
              frequency: float = 1000.0,
              amplitude: float = 0.1,
              offset:    float = 0.0,
              phase:     float = 0.0,
              channel:   int   = 1):
        """Issue an APPLy:<FUNC> command on the given channel."""
        _check_channel(channel)
        fn = _norm_func(function)
        scpi = _FUNC_SCPI_LONG[fn]

        # APPLy takes at most freq,amplitude,offset — it has no phase
        # argument. Passing a 4th value triggers -108 "Parameter not
        # allowed" and the instrument discards the whole command, so the
        # amplitude never updates. Phase is set separately via :PHASe,
        # and only for the functions that have a phase reference (sine,
        # square, ramp, arb per the 33500 user's guide) — pulse/noise/DC
        # have none and would reject :PHASe.
        if fn in ("NOIS", "DC"):
            cmd = (f":SOURce{channel}:APPLy:{scpi} "
                   f"DEFault,{amplitude:.6f},{offset:.6f}")
        else:
            cmd = (f":SOURce{channel}:APPLy:{scpi} "
                   f"{frequency:.6f},{amplitude:.6f},{offset:.6f}")

        has_phase = fn in ("SIN", "SQU", "RAMP", "ARB")
        if self._mode == "hardware":
            self._inst.write(cmd)
            if has_phase:
                self._inst.write(f":SOURce{channel}:PHASe {phase:.4f}")
        st = self._sim_state[channel]
        st["function"]  = fn
        st["frequency"] = float(frequency)
        st["amplitude"] = float(amplitude)
        st["offset"]    = float(offset)
        if has_phase:
            st["phase"] = float(phase)
        if fn == "SQU":
            st["duty_cycle"] = 50.0   # APPLy resets duty to 50%
        elif fn == "RAMP":
            st["symmetry"] = 100.0    # APPLy resets symmetry to 100%

    def query_apply(self, channel: int = 1) -> str:
        _check_channel(channel)
        if self._mode == "hardware":
            return self._inst.query(f":SOURce{channel}:APPLy?").strip()
        st = self._sim_state[channel]
        return (f'"{st["function"]} {st["frequency"]:.6e},'
                f'{st["amplitude"]:.6e},{st["offset"]:.6e}"')

    # ------------------------------------------------------------------
    # FUNCtion — set function only
    # ------------------------------------------------------------------

    def set_function(self, function: str, channel: int = 1):
        _check_channel(channel)
        fn = _norm_func(function)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:FUNCtion {_FUNC_SCPI_LONG[fn]}")
        self._sim_state[channel]["function"] = fn

    def get_function(self, channel: int = 1) -> str:
        _check_channel(channel)
        if self._mode == "hardware":
            return self._inst.query(f":SOURce{channel}:FUNCtion?").strip()
        return self._sim_state[channel]["function"]

    def set_square_duty_cycle(self, percent: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(
                f":SOURce{channel}:FUNCtion:SQUare:DCYCle {percent:.4f}")
        self._sim_state[channel]["duty_cycle"] = float(percent)

    def get_square_duty_cycle(self, channel: int = 1) -> float:
        _check_channel(channel)
        if self._mode == "hardware":
            return float(self._inst.query(
                f":SOURce{channel}:FUNCtion:SQUare:DCYCle?"))
        return self._sim_state[channel]["duty_cycle"]

    def set_ramp_symmetry(self, percent: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(
                f":SOURce{channel}:FUNCtion:RAMP:SYMMetry {percent:.4f}")
        self._sim_state[channel]["symmetry"] = float(percent)

    def get_ramp_symmetry(self, channel: int = 1) -> float:
        _check_channel(channel)
        if self._mode == "hardware":
            return float(self._inst.query(
                f":SOURce{channel}:FUNCtion:RAMP:SYMMetry?"))
        return self._sim_state[channel]["symmetry"]

    # ------------------------------------------------------------------
    # FREQuency / VOLTage / PHASe
    # ------------------------------------------------------------------

    def set_frequency(self, frequency: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:FREQuency {frequency:.6f}")
        self._sim_state[channel]["frequency"] = float(frequency)

    def get_frequency(self, channel: int = 1) -> float:
        _check_channel(channel)
        if self._mode == "hardware":
            return float(self._inst.query(f":SOURce{channel}:FREQuency?"))
        return self._sim_state[channel]["frequency"]

    def set_amplitude(self, amplitude: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:VOLTage {amplitude:.6f}")
        self._sim_state[channel]["amplitude"] = float(amplitude)

    def get_amplitude(self, channel: int = 1) -> float:
        _check_channel(channel)
        if self._mode == "hardware":
            return float(self._inst.query(f":SOURce{channel}:VOLTage?"))
        return self._sim_state[channel]["amplitude"]

    def set_offset(self, offset: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:VOLTage:OFFSet {offset:.6f}")
        self._sim_state[channel]["offset"] = float(offset)

    def get_offset(self, channel: int = 1) -> float:
        _check_channel(channel)
        if self._mode == "hardware":
            return float(self._inst.query(f":SOURce{channel}:VOLTage:OFFSet?"))
        return self._sim_state[channel]["offset"]

    def set_high_level(self, high_v: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:VOLTage:HIGH {high_v:.6f}")
        st = self._sim_state[channel]
        low = st["offset"] - st["amplitude"] / 2
        st["amplitude"] = high_v - low
        st["offset"]    = (high_v + low) / 2

    def set_low_level(self, low_v: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:VOLTage:LOW {low_v:.6f}")
        st = self._sim_state[channel]
        high = st["offset"] + st["amplitude"] / 2
        st["amplitude"] = high - low_v
        st["offset"]    = (high + low_v) / 2

    def set_phase(self, degrees: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:PHASe {degrees:.4f}")
        self._sim_state[channel]["phase"] = float(degrees)

    def get_phase(self, channel: int = 1) -> float:
        _check_channel(channel)
        if self._mode == "hardware":
            return float(self._inst.query(f":SOURce{channel}:PHASe?"))
        return self._sim_state[channel]["phase"]

    def sync_phases(self):
        """Synchronise the phase reference of CH1 and CH2."""
        if self._mode == "hardware":
            self._inst.write(":PHASe:SYNChronize")

    # ------------------------------------------------------------------
    # OUTPut
    # ------------------------------------------------------------------

    def output_on(self, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":OUTPut{channel} ON")
        self._sim_state[channel]["output"] = True

    def output_off(self, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":OUTPut{channel} OFF")
        self._sim_state[channel]["output"] = False

    def get_output_state(self, channel: int = 1) -> bool:
        _check_channel(channel)
        if self._mode == "hardware":
            resp = self._inst.query(f":OUTPut{channel}?").strip().upper()
            return resp in ("1", "ON")
        return self._sim_state[channel]["output"]

    def set_load(self, load_ohms: float | str, channel: int = 1):
        """Set output load. Use 'INF' or float('inf') for high-Z."""
        _check_channel(channel)
        if isinstance(load_ohms, str) and load_ohms.upper().startswith("INF"):
            arg = "INFinity"
            sim_val = float("inf")
        elif np.isinf(load_ohms):
            arg = "INFinity"
            sim_val = float("inf")
        else:
            arg = f"{float(load_ohms):.2f}"
            sim_val = float(load_ohms)
        if self._mode == "hardware":
            self._inst.write(f":OUTPut{channel}:LOAD {arg}")
        self._sim_state[channel]["load"] = sim_val

    def get_load(self, channel: int = 1) -> float:
        _check_channel(channel)
        if self._mode == "hardware":
            val = float(self._inst.query(f":OUTPut{channel}:LOAD?"))
            return float("inf") if val > 1e30 else val
        return self._sim_state[channel]["load"]

    def set_polarity(self, polarity: str, channel: int = 1):
        _check_channel(channel)
        p = polarity.upper()
        p = "NORM" if p.startswith("NORM") else "INV" if p.startswith("INV") else None
        if p is None:
            raise ValueError(f"polarity must be NORM or INV, got {polarity!r}")
        long = {"NORM": "NORMal", "INV": "INVerted"}[p]
        if self._mode == "hardware":
            self._inst.write(f":OUTPut{channel}:POLarity {long}")
        self._sim_state[channel]["polarity"] = p

    def set_sync_output(self, enable: bool, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":OUTPut{channel}:SYNC {'ON' if enable else 'OFF'}")
        self._sim_state[channel]["sync_output"] = bool(enable)

    # ------------------------------------------------------------------
    # PULSe
    # ------------------------------------------------------------------

    def set_pulse_period(self, period_s: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:FUNCtion:PULSe:PERiod {period_s:.9f}")
        self._sim_state[channel]["pulse_period"] = float(period_s)

    def set_pulse_width(self, width_s: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:FUNCtion:PULSe:WIDTh {width_s:.9f}")
        self._sim_state[channel]["pulse_width"] = float(width_s)

    def set_pulse_duty_cycle(self, percent: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(
                f":SOURce{channel}:FUNCtion:PULSe:DCYCle {percent:.4f}")
        self._sim_state[channel]["pulse_dcyc"] = float(percent)

    def set_pulse_leading_edge(self, time_s: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(
                f":SOURce{channel}:FUNCtion:PULSe:TRANsition:LEADing {time_s:.9f}")
        self._sim_state[channel]["pulse_rise"] = float(time_s)

    def set_pulse_trailing_edge(self, time_s: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(
                f":SOURce{channel}:FUNCtion:PULSe:TRANsition:TRAiling {time_s:.9f}")
        self._sim_state[channel]["pulse_fall"] = float(time_s)

    # ------------------------------------------------------------------
    # BURSt
    # ------------------------------------------------------------------

    def burst_enable(self, enable: bool, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:BURSt:STATe {'ON' if enable else 'OFF'}")
        self._sim_state[channel]["burst_state"] = bool(enable)

    def get_burst_state(self, channel: int = 1) -> bool:
        _check_channel(channel)
        if self._mode == "hardware":
            return self._inst.query(
                f":SOURce{channel}:BURSt:STATe?").strip().upper() in ("1", "ON")
        return self._sim_state[channel]["burst_state"]

    def set_burst_mode(self, mode: str, channel: int = 1):
        _check_channel(channel)
        m = _norm_burst_mode(mode)
        long = {"TRIG": "TRIGgered", "GAT": "GATed"}[m]
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:BURSt:MODe {long}")
        self._sim_state[channel]["burst_mode"] = m

    def set_burst_ncycles(self, ncycles: int | str, channel: int = 1):
        _check_channel(channel)
        if isinstance(ncycles, str) and ncycles.upper().startswith("INF"):
            arg = "INFinity"; sim_val = "INF"
        else:
            arg = f"{int(ncycles)}"; sim_val = int(ncycles)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:BURSt:NCYCles {arg}")
        self._sim_state[channel]["burst_ncycles"] = sim_val

    def set_burst_phase(self, degrees: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:BURSt:PHASe {degrees:.4f}")
        self._sim_state[channel]["burst_phase"] = float(degrees)

    # ------------------------------------------------------------------
    # SWEep
    # ------------------------------------------------------------------

    def sweep_enable(self, enable: bool, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:SWEep:STATe {'ON' if enable else 'OFF'}")
        self._sim_state[channel]["sweep_state"] = bool(enable)

    def get_sweep_state(self, channel: int = 1) -> bool:
        _check_channel(channel)
        if self._mode == "hardware":
            return self._inst.query(
                f":SOURce{channel}:SWEep:STATe?").strip().upper() in ("1", "ON")
        return self._sim_state[channel]["sweep_state"]

    def set_sweep_spacing(self, spacing: str, channel: int = 1):
        _check_channel(channel)
        s = _norm_spacing(spacing)
        long = {"LIN": "LINear", "LOG": "LOGarithmic"}[s]
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:SWEep:SPACing {long}")
        self._sim_state[channel]["sweep_spacing"] = s

    def set_sweep_time(self, seconds: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:SWEep:TIME {seconds:.6f}")
        self._sim_state[channel]["sweep_time"] = float(seconds)

    def set_sweep_return_time(self, seconds: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:SWEep:RTIMe {seconds:.6f}")
        self._sim_state[channel]["sweep_rtime"] = float(seconds)

    def set_sweep_hold_start(self, seconds: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:SWEep:HTIMe:STARt {seconds:.6f}")
        self._sim_state[channel]["sweep_hstart"] = float(seconds)

    def set_sweep_hold_stop(self, seconds: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:SWEep:HTIMe:STOP {seconds:.6f}")
        self._sim_state[channel]["sweep_hstop"] = float(seconds)

    def set_frequency_start(self, frequency: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:FREQuency:STARt {frequency:.6f}")
        self._sim_state[channel]["freq_start"] = float(frequency)

    def set_frequency_stop(self, frequency: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":SOURce{channel}:FREQuency:STOP {frequency:.6f}")
        self._sim_state[channel]["freq_stop"] = float(frequency)

    # ------------------------------------------------------------------
    # TRIGger
    # ------------------------------------------------------------------

    def set_trigger_source(self, source: str, channel: int = 1):
        _check_channel(channel)
        s = _norm_trigger(source)
        long = {"IMM": "IMMediate", "EXT": "EXTernal",
                "TIM": "TIMer",     "BUS": "BUS"}[s]
        if self._mode == "hardware":
            self._inst.write(f":TRIGger{channel}:SOURce {long}")
        self._sim_state[channel]["trig_source"] = s

    def trigger(self, channel: int = 1):
        """Issue an immediate software trigger on the channel."""
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f":TRIGger{channel}:IMMediate")

    def bus_trigger(self):
        """Issue a *TRG (only effective when trigger source is BUS)."""
        if self._mode == "hardware":
            self._inst.write("*TRG")

    # ------------------------------------------------------------------
    # DATA — arbitrary waveform upload
    # ------------------------------------------------------------------

    def download_arbitrary(self,
                           name: str,
                           samples: np.ndarray,
                           channel: int = 1,
                           binary: bool = True):
        """
        Upload an arbitrary waveform to volatile memory under ``name``.

        Parameters
        ----------
        name : str
            Waveform name (alphanumeric, max 12 chars, first char a letter).
        samples : np.ndarray
            1-D array. Values are normalised to [-1, +1]; out-of-range
            values are clipped after scaling by the max absolute value.
        channel : int
            Channel context for the upload (DATA catalog is shared).
        binary : bool
            If True (default), use the binary DATA:ARBitrary:DAC IEEE 488.2
            block transfer with 14-bit DAC codes. Otherwise use ASCII.
        """
        _check_channel(channel)
        if not name or len(name) > 12 or not name[0].isalpha():
            raise ValueError(
                f"name must be 1..12 chars and start with a letter, got {name!r}")

        arr = np.asarray(samples, dtype=np.float64).flatten()
        n = arr.size
        if n < ARB_MIN_POINTS or n > ARB_MAX_POINTS:
            raise ValueError(
                f"length must be {ARB_MIN_POINTS}..{ARB_MAX_POINTS}, got {n}")

        max_abs = float(np.max(np.abs(arr))) if n else 0.0
        norm = arr / max_abs if max_abs > 1.0 else arr.copy()
        norm = np.clip(norm, -1.0, 1.0)

        if binary:
            # Map [-1,+1] -> [0, 16383] unsigned 14-bit DAC.
            dac = np.round((norm + 1.0) / 2.0 * ARB_DAC_MAX).astype(np.uint16)
            dac = np.clip(dac, ARB_DAC_MIN, ARB_DAC_MAX)
            payload = struct.pack(f"<{n}H", *dac.tolist())
            byte_count = len(payload)
            block_hdr = f"#{len(str(byte_count))}{byte_count}"
            header = f":SOURce{channel}:DATA:ARBitrary:DAC {name},"
            cmd_bytes = (header.encode("ascii")
                         + block_hdr.encode("ascii")
                         + payload + b"\n")
            if self._mode == "hardware":
                self._inst.write_raw(cmd_bytes)
                time.sleep(max(0.05, n / 200_000.0))
        else:
            values = ",".join(f"{v:.6f}" for v in norm)
            cmd = f":SOURce{channel}:DATA:ARBitrary {name},{values}"
            if self._mode == "hardware":
                self._inst.write(cmd)
                time.sleep(max(0.05, n / 50_000.0))

        # Update sim state.
        self._sim_arb[name] = norm
        self._sim_state[channel]["arb_name"] = name

    def select_arbitrary(self, name: str, channel: int = 1):
        """Select a previously uploaded arbitrary waveform for output."""
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(f':SOURce{channel}:FUNCtion:ARBitrary "{name}"')
            self._inst.write(f":SOURce{channel}:FUNCtion ARB")
        self._sim_state[channel]["arb_name"] = name
        self._sim_state[channel]["function"] = "ARB"

    def set_arb_sample_rate(self, rate_sps: float, channel: int = 1):
        _check_channel(channel)
        if self._mode == "hardware":
            self._inst.write(
                f":SOURce{channel}:FUNCtion:ARBitrary:SRATe {rate_sps:.6f}")
        self._sim_state[channel]["arb_srate"] = float(rate_sps)

    def get_arb_sample_rate(self, channel: int = 1) -> float:
        _check_channel(channel)
        if self._mode == "hardware":
            return float(self._inst.query(
                f":SOURce{channel}:FUNCtion:ARBitrary:SRATe?"))
        return self._sim_state[channel]["arb_srate"]

    def get_catalog(self) -> str:
        """Query the volatile arbitrary waveform catalog."""
        if self._mode == "hardware":
            return self._inst.query(":SOURce1:DATA:VOLatile:CATalog?").strip()
        return ",".join(f'"{n}"' for n in self._sim_arb.keys())

    def delete_arbitrary(self, name: str):
        if self._mode == "hardware":
            self._inst.write(f":SOURce1:DATA:VOLatile:CLEar")  # use CLEar:name? -> use DELete
            self._inst.write(f":SOURce1:DATA:DELete {name}")
        self._sim_arb.pop(name, None)

    def delete_all_arbitrary(self):
        if self._mode == "hardware":
            self._inst.write(":SOURce1:DATA:VOLatile:CLEar")
        self._sim_arb.clear()

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def get_channel_state(self, channel: int = 1) -> dict:
        _check_channel(channel)
        return dict(self._sim_state[channel])

    # ------------------------------------------------------------------
    # Hardware internals
    # ------------------------------------------------------------------

    def _connect_hardware(self):
        try:
            import pyvisa
        except ImportError as e:
            raise ImportError(
                "pyvisa not installed. Run: pip install pyvisa pyvisa-py"
            ) from e

        self._rm = pyvisa.ResourceManager()
        last_err = None
        for _ in range(3):
            try:
                self._inst = self._rm.open_resource(self._visa_str)
                break
            except Exception as e:
                last_err = e
                time.sleep(1)
        else:
            raise RuntimeError(
                f"Could not open VISA resource {self._visa_str!r} after 3 attempts: "
                f"{last_err}")

        self._inst.timeout = self._timeout_ms
        try:
            self._inst.write_termination = "\n"
            self._inst.read_termination  = "\n"
            self._inst.clear()
        except Exception:
            pass

        idn = self._inst.query("*IDN?")
        ok_vendor = any(v.upper() in idn.upper() for v in IDN_VENDORS)
        ok_model  = any(m in idn for m in IDN_MODELS)
        if not (ok_vendor and ok_model):
            self._inst.close()
            raise RuntimeError(
                f"IDN mismatch. Expected Keysight/Agilent 33500-series, got {idn!r}\n"
                f"Check VISA string: {self._visa_str!r}")

    def _disconnect_hardware(self):
        if self._inst is not None:
            try:
                self.output_off(1)
                self.output_off(2)
                self._inst.close()
            except Exception:
                pass
            self._inst = None
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass
            self._rm = None

    # ------------------------------------------------------------------
    # VISA discovery (hardware mode only)
    # ------------------------------------------------------------------

    @staticmethod
    def discover(filter_keysight: bool = True,
                 query: str = "?*::INSTR") -> list[tuple[str, str]]:
        """
        Discover available VISA instruments by querying *IDN? on each.

        Returns
        -------
        list of (resource, idn) tuples. If ``filter_keysight`` is True,
        only entries whose IDN names a Keysight/Agilent 33500-series box
        are returned.
        """
        try:
            import pyvisa
        except ImportError:
            return []
        try:
            rm = pyvisa.ResourceManager()
        except Exception:
            return []
        found: list[tuple[str, str]] = []
        try:
            resources = [r for r in rm.list_resources(query)
                         if not r.startswith("ASRL")]
        except Exception:
            resources = []
        for res in resources:
            try:
                inst = rm.open_resource(res)
                inst.timeout = 2000
                inst.write_termination = "\n"
                inst.read_termination  = "\n"
                idn = inst.query("*IDN?").strip()
                inst.close()
            except Exception:
                continue
            if filter_keysight:
                ok_vendor = any(v.upper() in idn.upper() for v in IDN_VENDORS)
                ok_model  = any(m in idn for m in IDN_MODELS)
                if not (ok_vendor and ok_model):
                    continue
            found.append((res, idn))
        try:
            rm.close()
        except Exception:
            pass
        return found

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
