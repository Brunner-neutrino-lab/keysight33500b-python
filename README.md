# ks33500b

Python driver and GUI for the Keysight 33500B family of function/arbitrary
waveform generators (33511B, 33512B, 33521B, 33522B and the A variants).

Same architecture as the
[rigoldg1022-python](https://github.com/) driver: low-level SCPI driver,
high-level controller, standalone PyQt5 GUI, and a simulation mode so the
API exercises without hardware attached.

## Operating modes

| Mode | Method | Use case |
|------|--------|----------|
| **Continuous waveform** | `apply_sine` / `apply_square` / `apply_ramp` / `apply_pulse` / `apply_noise` / `apply_dc` | Free-running output on CH1/CH2 |
| **Pulse** | `apply_pulse` + `configure_pulse(period_s, width_s, ...)` | TTL pulses, gates |
| **Arbitrary** | `load_arbitrary(name, samples)` + `apply_arbitrary(...)` | User waveforms downloaded to volatile memory |
| **Burst** | `enable_burst(ncycles, mode, trigger, channel)` + `trigger()` / `bus_trigger()` | N-cycle bursts per trigger |
| **Frequency sweep** | `enable_sweep(start_hz, stop_hz, time_s, spacing, ...)` | Linear / log sweeps with optional return + hold |

## Quick start

```bash
pip install -r requirements.txt

python -m ks33500b.gui              # standalone GUI
python examples/basic_usage.py      # headless example (simulation mode)
```

## API

```python
from ks33500b import KS33500BController, WaveformGenerator
import numpy as np

with KS33500BController(visa="USB0::0x0957::...", mode="simulation") as awg:

    # 1. Continuous sine on CH1
    awg.apply_sine(frequency=1_000, amplitude=0.1, offset=0.0, channel=1)
    awg.output_on(1)

    # 2. Pulse on CH2
    awg.apply_pulse(frequency=10_000, amplitude=3.3, offset=1.65, channel=2)
    awg.configure_pulse(period_s=100e-6, width_s=20e-6, channel=2)
    awg.output_on(2)

    # 3. Burst — 5 cycles per software trigger on CH1
    awg.apply_sine(1_000, 0.1, 0.0, channel=1)
    awg.enable_burst(ncycles=5, mode="TRIG", trigger="BUS", channel=1)
    awg.bus_trigger()

    # 4. Linear frequency sweep on CH1
    awg.enable_sweep(start_hz=100e3, stop_hz=700e3, time_s=0.1,
                     spacing="LIN", channel=1)

    # 5. Arbitrary waveform — sine cubed
    samples = np.sin(2 * np.pi * np.linspace(0, 1, 1024, endpoint=False)) ** 3
    awg.load_arbitrary("SINCUBED", samples, channel=1, binary=True)
    awg.apply_arbitrary(frequency=500, amplitude=0.5, offset=0.0, channel=1)
    awg.output_on(1)

    # 6. Frequency comb (Monte Carlo phase optimisation for flat envelope)
    comb = WaveformGenerator.frequency_comb([100, 200, 300, 400, 500],
                                            monte_carlo_iter=500, name="COMB1")
    awg.load_arbitrary(comb.name, comb.data, channel=2,
                       sample_rate=comb.sample_rate)
    awg.apply_arbitrary(frequency=comb.frequency, amplitude=1.0, channel=2)
```

## Channel convention

`channel=1` maps to `:SOURce1:...` (CH1).
`channel=2` maps to `:SOURce2:...` (CH2).

Two-channel commands (sweep, burst, trigger) take the channel as a keyword
on every call — pick `channel=1` or `channel=2` per call. Calls to channel 2
on a single-channel model are silently ignored by the instrument.

## Notes on the SCPI mapping

The driver follows the Keysight 33500B Programming Reference. Key choices:

- `:SOURce<n>:APPLy:<func> f,a,o,phase` is used as the primary
  set-everything-at-once command (`NOIS`/`DC` use `DEFault` for f).
- After `:APPLy:SQUare`, the duty cycle is forced to 50 %; pass
  `duty_cycle=` to `apply_square` to override after.
- `:OUTPut<n>:LOAD INFinity` (or `set_load("INF")`) selects high-Z.
- `:SOURce<n>:DATA:ARBitrary:DAC <name>,<block>` is used for fast binary
  uploads (14-bit DAC, 8..65 536 samples). Pass `binary=False` to use
  the ASCII `:DATA:ARBitrary` path instead.
- `:TRIGger<n>:IMMediate` issues a per-channel software trigger;
  `*TRG` issues a global BUS trigger (only effective when the source is BUS).
- Simulation mode never opens VISA; it caches per-channel state plus an
  arbitrary-waveform catalog so the full API can be exercised offline
  (used by the GUI tabs and the example).

## Files

```
ks33500b/
  __init__.py
  driver.py       # low-level SCPI + sim
  controller.py   # high-level API
  arbitrary.py    # pure-numpy waveform helpers (shapes + frequency comb)
  gui.py          # PyQt5 standalone GUI
examples/
  basic_usage.py
```
