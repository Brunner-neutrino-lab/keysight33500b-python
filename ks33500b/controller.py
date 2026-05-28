"""
ks33500b/controller.py

High-level controller for the Keysight 33500B series function/arbitrary
waveform generators (33511B, 33512B, 33521B, 33522B; A variants supported).

Operating modes (any combination on either channel):
  1. Continuous waveform   — sine / square / ramp / pulse / noise / DC
  2. Pulse                 — period, width, duty, rise/fall times
  3. Arbitrary waveform    — upload samples to volatile memory and play
  4. Burst                 — N cycles per trigger
  5. Frequency sweep       — linear or logarithmic, optional return/hold

Usage (headless):

    from ks33500b import KS33500BController

    with KS33500BController(mode="simulation") as awg:

        # 1. Continuous sine on CH1
        awg.apply_sine(1000.0, 0.1, 0.0, channel=1)
        awg.output_on(1)

        # 2. Pulse on CH2
        awg.apply_pulse(frequency=10e3, amplitude=3.3, offset=1.65, channel=2)
        awg.configure_pulse(period_s=100e-6, width_s=20e-6, channel=2)
        awg.output_on(2)

        # 3. Burst — 5 cycles per software trigger on CH1
        awg.apply_sine(1e3, 0.1, 0.0, channel=1)
        awg.enable_burst(ncycles=5, mode="TRIG", trigger="BUS", channel=1)
        awg.bus_trigger()

        # 4. Arbitrary on CH1
        import numpy as np
        samples = np.sin(2*np.pi*np.linspace(0, 1, 1024))**3
        awg.load_arbitrary("ARBSC", samples, channel=1)
        awg.apply_arbitrary(1e3, 0.1, 0.0, channel=1)
        awg.output_on(1)
"""

import numpy as np

from .driver import (
    KS33500BDriver, DEFAULT_VISA, FUNCTIONS, TRIG_SOURCES,
    BURST_MODES, SWEEP_SPACINGS,
    ARB_MIN_POINTS, ARB_MAX_POINTS, ARB_MIN_RATE, ARB_MAX_RATE,
)


class KS33500BController:
    """
    High-level controller for the Keysight 33500B.

    Parameters
    ----------
    visa : str
        VISA resource string.
    mode : str
        "hardware" or "simulation".
    timeout_ms : int
        VISA timeout in milliseconds (hardware mode).
    """

    # ------------------------------------------------------------------
    # Plugin interface (for ETS DAQ discovery)
    # ------------------------------------------------------------------
    MODULE_NAME = "KS33500B"
    DEVICE_NAME = "Keysight 33500B Function/Arbitrary Waveform Generator"
    CONFIG_FIELDS = [
        {"key": "visa", "label": "VISA Resource", "type": "str",    "default": DEFAULT_VISA},
        {"key": "mode", "label": "Mode",          "type": "choice", "default": "simulation",
         "choices": ["simulation", "hardware"]},
    ]
    DEFAULTS = {"visa": DEFAULT_VISA, "mode": "simulation"}

    @staticmethod
    def test(config: dict) -> tuple[bool, str]:
        try:
            ctrl = KS33500BController(
                visa=config.get("visa", DEFAULT_VISA),
                mode=config.get("mode", "simulation"),
            )
            ctrl.connect()
            idn = ctrl.identify()
            ctrl.disconnect()
            return True, f"OK — {idn}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    @staticmethod
    def read(config: dict) -> dict:
        return {
            "visa": config.get("visa", ""),
            "mode": config.get("mode", "simulation"),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, visa: str = DEFAULT_VISA, mode: str = "simulation",
                 timeout_ms: int = 5000):
        self._driver = KS33500BDriver(visa=visa, mode=mode, timeout_ms=timeout_ms)

    def connect(self):
        self._driver.connect()

    def disconnect(self):
        # Safety: turn off outputs before disconnecting hardware.
        try:
            self._driver.output_off(1)
            self._driver.output_off(2)
        except Exception:
            pass
        self._driver.disconnect()

    def identify(self) -> str:
        return self._driver.identify()

    def reset(self):
        self._driver.reset()

    def beep(self):
        self._driver.beep()

    def get_error(self) -> str:
        return self._driver.get_error()

    @property
    def driver(self) -> KS33500BDriver:
        return self._driver

    @property
    def is_connected(self) -> bool:
        return self._driver.is_connected

    # Convenience for VISA discovery without instantiating a controller.
    @staticmethod
    def discover(filter_keysight: bool = True) -> list[tuple[str, str]]:
        return KS33500BDriver.discover(filter_keysight=filter_keysight)

    # ------------------------------------------------------------------
    # 1. Continuous waveform — APPLy convenience wrappers
    # ------------------------------------------------------------------

    def apply_sine(self, frequency: float, amplitude: float = 0.1,
                   offset: float = 0.0, phase: float = 0.0,
                   channel: int = 1):
        """Sine wave (Hz, Vpp, V, deg)."""
        self._driver.apply("SIN", frequency, amplitude, offset, phase, channel)

    def apply_square(self, frequency: float, amplitude: float = 0.1,
                     offset: float = 0.0, phase: float = 0.0,
                     duty_cycle: float | None = None,
                     channel: int = 1):
        """Square wave. APPLy resets duty to 50%; pass duty_cycle to override after."""
        self._driver.apply("SQU", frequency, amplitude, offset, phase, channel)
        if duty_cycle is not None:
            self._driver.set_square_duty_cycle(duty_cycle, channel)

    def apply_ramp(self, frequency: float, amplitude: float = 0.1,
                   offset: float = 0.0, phase: float = 0.0,
                   symmetry: float | None = None,
                   channel: int = 1):
        """Ramp wave. APPLy resets symmetry to 100%; pass symmetry to override after."""
        self._driver.apply("RAMP", frequency, amplitude, offset, phase, channel)
        if symmetry is not None:
            self._driver.set_ramp_symmetry(symmetry, channel)

    def apply_pulse(self, frequency: float, amplitude: float = 0.1,
                    offset: float = 0.0, phase: float = 0.0,
                    channel: int = 1):
        """Pulse. Period/width/duty/edges configured via configure_pulse()."""
        self._driver.apply("PULS", frequency, amplitude, offset, phase, channel)

    def apply_noise(self, amplitude: float = 0.1, offset: float = 0.0,
                    channel: int = 1):
        """Gaussian noise."""
        self._driver.apply("NOIS", 0.0, amplitude, offset, 0.0, channel)

    def apply_dc(self, offset: float, channel: int = 1):
        """DC level only."""
        self._driver.apply("DC", 0.0, 0.0, offset, 0.0, channel)

    def apply_arbitrary(self, frequency: float, amplitude: float = 0.1,
                        offset: float = 0.0, channel: int = 1):
        """
        Play the currently selected arbitrary waveform.

        Upload + select first with :meth:`load_arbitrary`.
        """
        self._driver.apply("ARB", frequency, amplitude, offset, 0.0, channel)

    # ------------------------------------------------------------------
    # 2. Pulse parameters
    # ------------------------------------------------------------------

    def configure_pulse(self,
                        period_s:    float | None = None,
                        width_s:     float | None = None,
                        duty_pct:    float | None = None,
                        rise_time_s: float | None = None,
                        fall_time_s: float | None = None,
                        channel:     int = 1):
        """Set pulse parameters on a channel. Specify any subset."""
        if period_s is not None:
            self._driver.set_pulse_period(period_s, channel)
        if width_s is not None:
            self._driver.set_pulse_width(width_s, channel)
        if duty_pct is not None:
            self._driver.set_pulse_duty_cycle(duty_pct, channel)
        if rise_time_s is not None:
            self._driver.set_pulse_leading_edge(rise_time_s, channel)
        if fall_time_s is not None:
            self._driver.set_pulse_trailing_edge(fall_time_s, channel)

    # ------------------------------------------------------------------
    # 3. Voltage / output configuration
    # ------------------------------------------------------------------

    def set_amplitude(self, amplitude: float, channel: int = 1):
        self._driver.set_amplitude(amplitude, channel)

    def set_offset(self, offset: float, channel: int = 1):
        self._driver.set_offset(offset, channel)

    def set_frequency(self, frequency: float, channel: int = 1):
        self._driver.set_frequency(frequency, channel)

    def set_high_low(self, high_v: float, low_v: float, channel: int = 1):
        """Set amplitude/offset via high and low levels."""
        self._driver.set_high_level(high_v, channel)
        self._driver.set_low_level(low_v, channel)

    def set_phase(self, degrees: float, channel: int = 1):
        self._driver.set_phase(degrees, channel)

    def sync_phases(self):
        """Synchronise CH1/CH2 phase references (PHASe:SYNChronize)."""
        self._driver.sync_phases()

    def set_load(self, load_ohms: float | str, channel: int = 1):
        """Set output load — ohm value or 'INF'/float('inf') for high-Z."""
        self._driver.set_load(load_ohms, channel)

    def set_polarity(self, polarity: str, channel: int = 1):
        """polarity in {'NORM', 'INV'}."""
        self._driver.set_polarity(polarity, channel)

    def enable_sync_output(self, enable: bool = True, channel: int = 1):
        self._driver.set_sync_output(enable, channel)

    def output_on(self, channel: int = 1):
        self._driver.output_on(channel)

    def output_off(self, channel: int = 1):
        self._driver.output_off(channel)

    def all_outputs_off(self):
        self._driver.output_off(1)
        self._driver.output_off(2)

    def get_output_state(self, channel: int = 1) -> bool:
        return self._driver.get_output_state(channel)

    # ------------------------------------------------------------------
    # 4. Burst
    # ------------------------------------------------------------------

    def enable_burst(self,
                     ncycles:   int | str = 1,
                     mode:      str = "TRIG",
                     phase_deg: float = 0.0,
                     trigger:   str = "IMM",
                     channel:   int = 1):
        """
        Configure and enable burst mode on a channel.

        Parameters
        ----------
        ncycles : int or 'INF'
            Cycles per trigger (TRIG mode).
        mode : str
            'TRIG' — N cycles on each trigger.
            'GAT'  — output gated by external level.
        phase_deg : float
            Initial phase in degrees.
        trigger : str
            'IMM', 'EXT', 'TIM', or 'BUS'.
        channel : int
            1 or 2.
        """
        self._driver.set_burst_mode(mode, channel)
        self._driver.set_burst_ncycles(ncycles, channel)
        self._driver.set_burst_phase(phase_deg, channel)
        self._driver.set_trigger_source(trigger, channel)
        self._driver.burst_enable(True, channel)

    def disable_burst(self, channel: int = 1):
        self._driver.burst_enable(False, channel)

    def trigger(self, channel: int = 1):
        """Issue an immediate trigger on the channel (:TRIGger<n>:IMMediate)."""
        self._driver.trigger(channel)

    def bus_trigger(self):
        """*TRG — software trigger, valid when source is BUS on either channel."""
        self._driver.bus_trigger()

    # ------------------------------------------------------------------
    # 5. Frequency sweep
    # ------------------------------------------------------------------

    def enable_sweep(self,
                     start_hz:    float,
                     stop_hz:     float,
                     time_s:      float = 1.0,
                     spacing:     str   = "LIN",
                     return_time: float = 0.0,
                     hold_start:  float = 0.0,
                     hold_stop:   float = 0.0,
                     trigger:     str   = "IMM",
                     channel:     int   = 1):
        """
        Configure and enable a frequency sweep on a channel.

        The carrier waveform (sine/square/ramp) must be selected first
        via apply_sine/apply_square/apply_ramp.
        """
        self._driver.set_frequency_start(start_hz, channel)
        self._driver.set_frequency_stop(stop_hz, channel)
        self._driver.set_sweep_time(time_s, channel)
        self._driver.set_sweep_spacing(spacing, channel)
        self._driver.set_sweep_return_time(return_time, channel)
        self._driver.set_sweep_hold_start(hold_start, channel)
        self._driver.set_sweep_hold_stop(hold_stop, channel)
        self._driver.set_trigger_source(trigger, channel)
        self._driver.sweep_enable(True, channel)

    def disable_sweep(self, channel: int = 1):
        self._driver.sweep_enable(False, channel)

    # ------------------------------------------------------------------
    # 6. Arbitrary waveform
    # ------------------------------------------------------------------

    def load_arbitrary(self,
                       name:    str,
                       samples: np.ndarray,
                       channel: int = 1,
                       binary:  bool = True,
                       select:  bool = True,
                       sample_rate: float | None = None):
        """
        Upload an arbitrary waveform to volatile memory and (optionally)
        select it for output on the given channel.

        Parameters
        ----------
        name : str
            Waveform name (alphanumeric, max 12 chars, first char a letter).
        samples : np.ndarray
            1-D array; values are normalised to [-1, +1] before transfer.
            Length must be {ARB_MIN_POINTS}..{ARB_MAX_POINTS}.
        channel : int
            Channel context (1 or 2).
        binary : bool
            Use binary block transfer (default, faster) or ASCII.
        select : bool
            If True, set the channel's function to ARB and pick this waveform.
        sample_rate : float, optional
            If given, set the arbitrary sample rate (Sa/s).
        """
        self._driver.download_arbitrary(name, samples, channel=channel, binary=binary)
        if select:
            self._driver.select_arbitrary(name, channel)
        if sample_rate is not None:
            self._driver.set_arb_sample_rate(sample_rate, channel)

    def select_arbitrary(self, name: str, channel: int = 1):
        """Select a previously uploaded arbitrary waveform by name."""
        self._driver.select_arbitrary(name, channel)

    def set_arb_sample_rate(self, rate_sps: float, channel: int = 1):
        self._driver.set_arb_sample_rate(rate_sps, channel)

    def list_waves(self) -> list[str]:
        """Return waveform names available in volatile memory."""
        raw = self._driver.get_catalog()
        return [s.strip().strip('"') for s in raw.split(",") if s.strip()]

    def delete_arbitrary(self, name: str):
        self._driver.delete_arbitrary(name)

    def delete_all_arbitrary(self):
        self._driver.delete_all_arbitrary()

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    def get_channel_state(self, channel: int = 1) -> dict:
        """Snapshot of cached/simulated channel parameters (for status display)."""
        return self._driver.get_channel_state(channel)

    def get_status(self) -> dict:
        return {
            "connected": self._driver.is_connected,
            "idn":       self.identify() if self._driver.is_connected else None,
            "channels":  {ch: self.get_channel_state(ch) for ch in (1, 2)},
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()
