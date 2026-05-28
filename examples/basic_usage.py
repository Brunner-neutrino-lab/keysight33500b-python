"""
KS33500B basic usage — simulation mode.

Demonstrates the main operating modes:
  1. Continuous sine on CH1
  2. Continuous pulse on CH2 with explicit period/width
  3. Burst (5 cycles per *TRG) on CH1
  4. Linear frequency sweep on CH1 with hold + return
  5. Arbitrary waveform (sine^3) upload on CH1
  6. Frequency comb upload on CH2
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ks33500b import KS33500BController, WaveformGenerator


def main():
    with KS33500BController(mode="simulation") as awg:
        print("IDN:", awg.identify())

        # --- 1. Continuous sine on CH1 ---
        print("\n1. CH1: sine 1 kHz, 100 mVpp, 0 VDC")
        awg.set_load("INF", channel=1)
        awg.apply_sine(frequency=1_000, amplitude=0.1, offset=0.0, channel=1)
        awg.output_on(1)
        print("   state:", awg.get_channel_state(1))
        awg.output_off(1)

        # --- 2. Continuous pulse on CH2 ---
        print("\n2. CH2: 10 kHz pulse, 3.3 Vpp, 1.65 V offset, 20 us width")
        awg.apply_pulse(frequency=10_000, amplitude=3.3, offset=1.65, channel=2)
        awg.configure_pulse(period_s=100e-6, width_s=20e-6,
                            rise_time_s=8.4e-9, fall_time_s=8.4e-9,
                            channel=2)
        awg.output_on(2)
        print("   state:", awg.get_channel_state(2))
        awg.output_off(2)

        # --- 3. Burst on CH1 ---
        print("\n3. CH1: 5-cycle burst per *TRG, sine 1 kHz")
        awg.apply_sine(1_000, 0.1, 0.0, channel=1)
        awg.enable_burst(ncycles=5, mode="TRIG", trigger="BUS", channel=1)
        awg.bus_trigger()
        print("   burst armed; *TRG sent")
        awg.disable_burst(channel=1)

        # --- 4. Linear frequency sweep on CH1 ---
        print("\n4. CH1: linear sweep 100 kHz -> 700 kHz over 0.1 s "
              "(0.01 s hold each end, 0.05 s return)")
        awg.apply_sine(400_000, 0.1, 0.0, channel=1)
        awg.enable_sweep(start_hz=100e3, stop_hz=700e3, time_s=0.1,
                         spacing="LIN",
                         hold_start=0.01, hold_stop=0.01, return_time=0.05,
                         trigger="IMM", channel=1)
        print("   sweep enabled")
        awg.disable_sweep(channel=1)

        # --- 5. Arbitrary waveform: sine cubed ---
        print("\n5. CH1: arbitrary waveform (sine^3, 1024 pts)")
        t = np.linspace(0, 1, 1024, endpoint=False)
        samples = np.sin(2 * np.pi * t) ** 3
        awg.load_arbitrary("SINCUBED", samples, channel=1, binary=True)
        awg.apply_arbitrary(frequency=500, amplitude=0.5, offset=0.0, channel=1)
        awg.output_on(1)
        print("   selected ARB on CH1:", awg.get_channel_state(1)["arb_name"])

        # --- 6. Frequency comb on CH2 ---
        print("\n6. CH2: 10-tone frequency comb (100 Hz .. 1 kHz)")
        comb = WaveformGenerator.frequency_comb(
            [100 * k for k in range(1, 11)],
            monte_carlo_iter=200, name="COMB1", seed=0)
        print(f"   {comb.description}")
        awg.load_arbitrary(comb.name, comb.data, channel=2,
                           binary=True, sample_rate=comb.sample_rate)
        awg.apply_arbitrary(frequency=comb.frequency, amplitude=1.0,
                            offset=0.0, channel=2)

        awg.all_outputs_off()

    print("\nDone.")


if __name__ == "__main__":
    main()
