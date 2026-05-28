"""
ks33500b/arbitrary.py

Pure-numpy helpers for building arbitrary-waveform sample arrays that the
KS33500BController can upload via load_arbitrary().

Nothing here touches VISA or the instrument — these are convenience
generators for common shapes (sine/square/ramp/pulse/Gaussian/sinc/
exponential) and a frequency-comb builder with Monte Carlo phase
optimisation for maximum RMS.

Hardware constants (33500B standard memory):
  Min points  :   8
  Max points  : 65 536
  Min rate    :   1 Sa/s
  Max rate    : 250 MSa/s
  Vertical res: 14 bits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np

from .driver import (
    ARB_MIN_POINTS, ARB_MAX_POINTS,
    ARB_MIN_RATE,   ARB_MAX_RATE,
)

# Smoothness target — number of samples per cycle of the highest-frequency
# component before we consider a comb under-sampled.
MIN_POINTS_PER_CYCLE = 10


# ---------------------------------------------------------------------------
# Optimal point-count calculators
# ---------------------------------------------------------------------------

def compute_optimal_points(frequency: float) -> int:
    """
    Compute the optimal number of waveform points for ``frequency`` (Hz),
    maximising smoothness within the 33500B hardware limits.

    sample_rate = N * frequency must satisfy ARB_MAX_RATE,
    and N must be in [ARB_MIN_POINTS, ARB_MAX_POINTS].
    """
    if frequency <= 0:
        return ARB_MAX_POINTS
    max_pts = int(ARB_MAX_RATE / frequency)
    return max(ARB_MIN_POINTS, min(max_pts, ARB_MAX_POINTS))


def _approximate_gcd(freqs: np.ndarray) -> float:
    """Approximate GCD of an array of frequencies."""
    min_f = float(np.min(freqs))
    if min_f <= 0:
        return min_f
    ratios = np.round(freqs / min_f).astype(int)
    g = reduce(gcd, ratios.tolist())
    return float(min_f / g) if g != 0 else min_f


def compute_optimal_points_for_comb(frequencies: list[float]
                                    ) -> tuple[int, dict]:
    """
    Compute the optimal point count for a frequency-comb waveform.

    The buffer represents one period of the fundamental (GCD of all tones).
    The highest tone constrains smoothness.

    Returns
    -------
    (num_points, info)
        info contains: fundamental, f_max, harmonics, min_points_needed,
        max_points_available, num_points, sample_rate, warning.
    """
    freqs = np.array(frequencies, dtype=float)
    if freqs.size == 0:
        raise ValueError("No frequencies specified")
    if np.any(freqs <= 0):
        raise ValueError("All frequencies must be positive")

    fundamental = _approximate_gcd(freqs)
    f_max = float(np.max(freqs))
    harmonics = f_max / fundamental

    min_pts_needed = int(np.ceil(harmonics * MIN_POINTS_PER_CYCLE))
    max_pts_available = min(int(ARB_MAX_RATE / fundamental), ARB_MAX_POINTS)
    optimal = max(ARB_MIN_POINTS, max_pts_available)

    warning: Optional[str] = None
    if min_pts_needed > ARB_MAX_POINTS:
        warning = (
            f"Frequency span too wide for smooth output. "
            f"Highest tone ({f_max:.4g} Hz) needs ~{min_pts_needed} points "
            f"within a {fundamental:.4g} Hz fundamental period, "
            f"but the 33500B maximum is {ARB_MAX_POINTS}.")
    elif optimal < min_pts_needed:
        warning = (
            f"Rate-limited: only {optimal} points achievable "
            f"(need {min_pts_needed} for smooth output).")

    info = {
        "fundamental":         fundamental,
        "f_max":               f_max,
        "harmonics":           harmonics,
        "min_points_needed":   min_pts_needed,
        "max_points_available": max_pts_available,
        "num_points":          optimal,
        "sample_rate":         optimal * fundamental,
        "warning":             warning,
    }
    return optimal, info


def check_frequency_feasibility(frequency: float) -> tuple[bool, str]:
    """Quick feasibility check for a single tone in arbitrary-waveform mode."""
    if frequency <= 0:
        return False, "Frequency must be positive."
    if frequency * ARB_MIN_POINTS > ARB_MAX_RATE:
        max_f = ARB_MAX_RATE / ARB_MIN_POINTS
        return False, (f"Frequency too high for arbitrary mode "
                       f"(max ≈ {max_f/1e6:.1f} MHz with {ARB_MIN_POINTS} pts).")
    n = compute_optimal_points(frequency)
    sr = frequency * n
    return True, (f"OK — {n} points, sample rate {sr/1e6:.3g} MSa/s "
                  f"({n/MIN_POINTS_PER_CYCLE:.1f}× smoothness margin)")


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class ArbitraryWaveform:
    """
    Normalised arbitrary waveform plus metadata.

    ``data`` is held as a float64 numpy array in [-1, +1]. ``name`` must
    satisfy the 33500B naming rules (1..12 chars, starts with a letter).
    """
    name:        str             = "ARB1"
    data:        Optional[np.ndarray] = None
    frequency:   float           = 1000.0
    sample_rate: float           = 0.0
    num_points:  int             = 0
    description: str             = ""
    comb_info:   Optional[dict]  = field(default=None, repr=False)

    def __post_init__(self):
        # Enforce 12-char instrument naming limit.
        self.name = self.name[:12]

    # ---- I/O ----

    def save_csv(self, path: str) -> None:
        if self.data is None:
            raise ValueError("No waveform data to save")
        np.savetxt(path, self.data, delimiter=",",
                   header=f"name={self.name},freq={self.frequency:.6g}",
                   comments="# ")

    def load_csv(self, path: str) -> None:
        raw = np.loadtxt(path, delimiter=",", comments="#")
        if raw.ndim > 1:
            raw = raw[:, 0]
        max_abs = float(np.max(np.abs(raw))) if raw.size else 0.0
        self.data = raw / max_abs if max_abs > 0 else raw
        self.num_points = int(self.data.size)
        try:
            with open(path) as f:
                line = f.readline().strip()
                if line.startswith("# name="):
                    parts = dict(p.split("=") for p in
                                 line.lstrip("# ").split(","))
                    self.name = parts.get("name", self.name)[:12]
                    self.frequency = float(parts.get("freq", self.frequency))
        except Exception:
            pass

    def __repr__(self) -> str:
        pts = self.num_points if self.data is not None else 0
        return (f"ArbitraryWaveform(name={self.name!r}, "
                f"points={pts}, freq={self.frequency:.4g} Hz)")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class WaveformGenerator:
    """
    Factory producing :class:`ArbitraryWaveform` objects for common shapes.

    All generators normalise their output to [-1, +1] and pick a point count
    via :func:`compute_optimal_points` so the resulting sample rate stays
    within the 33500B's 250 MSa/s ceiling.
    """

    @staticmethod
    def sine(frequency: float, name: str = "ARB1") -> ArbitraryWaveform:
        n = compute_optimal_points(frequency)
        t = np.linspace(0, 1, n, endpoint=False)
        return _build(name, np.sin(2 * np.pi * t), frequency, n,
                      f"Sine at {frequency:.4g} Hz")

    @staticmethod
    def square(frequency: float, duty_cycle: float = 50.0,
               name: str = "ARB1") -> ArbitraryWaveform:
        n = compute_optimal_points(frequency)
        t = np.linspace(0, 1, n, endpoint=False)
        data = np.where(t < duty_cycle / 100.0, 1.0, -1.0)
        return _build(name, data, frequency, n,
                      f"Square {duty_cycle:.0f}% DC at {frequency:.4g} Hz")

    @staticmethod
    def ramp(frequency: float, symmetry: float = 100.0,
             name: str = "ARB1") -> ArbitraryWaveform:
        """symmetry=100% rising, 50% triangle, 0% falling."""
        n = compute_optimal_points(frequency)
        t = np.linspace(0, 1, n, endpoint=False)
        sym = max(0.0, min(1.0, symmetry / 100.0))
        if sym == 0.0:
            data = 1.0 - 2.0 * t
        elif sym == 1.0:
            data = 2.0 * t - 1.0
        else:
            up = 2.0 * t / sym - 1.0
            dn = 1.0 - 2.0 * (t - sym) / (1.0 - sym)
            data = np.where(t < sym, up, dn)
        return _build(name, data, frequency, n,
                      f"Ramp {symmetry:.0f}% sym at {frequency:.4g} Hz")

    @staticmethod
    def pulse(frequency: float, duty_cycle: float = 50.0,
              rise_frac: float = 0.01, fall_frac: float = 0.01,
              name: str = "ARB1") -> ArbitraryWaveform:
        n = compute_optimal_points(frequency)
        t = np.linspace(0, 1, n, endpoint=False)
        duty = duty_cycle / 100.0
        data = np.full(n, -1.0)
        rise_end = rise_frac
        flat_end = max(rise_frac, duty - fall_frac)
        # Rising edge
        m = t < rise_end
        if rise_frac > 0:
            data[m] = -1.0 + 2.0 * t[m] / rise_frac
        # Flat top
        m = (t >= rise_end) & (t < flat_end)
        data[m] = 1.0
        # Falling edge
        m = (t >= flat_end) & (t < duty)
        if fall_frac > 0:
            data[m] = 1.0 - 2.0 * (t[m] - flat_end) / fall_frac
        return _build(name, data, frequency, n,
                      f"Pulse {duty_cycle:.0f}% DC at {frequency:.4g} Hz")

    @staticmethod
    def gaussian(frequency: float, sigma: float = 0.15,
                 name: str = "ARB1") -> ArbitraryWaveform:
        n = compute_optimal_points(frequency)
        t = np.linspace(0, 1, n, endpoint=False)
        data = np.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
        data = 2.0 * (data / np.max(data)) - 1.0
        return _build(name, data, frequency, n,
                      f"Gaussian σ={sigma:.2f} at {frequency:.4g} Hz")

    @staticmethod
    def sinc(frequency: float, lobes: int = 4,
             name: str = "ARB1") -> ArbitraryWaveform:
        n = compute_optimal_points(frequency)
        t = np.linspace(-lobes, lobes, n)
        return _build(name, np.sinc(t), frequency, n,
                      f"Sinc {lobes} lobes at {frequency:.4g} Hz")

    @staticmethod
    def exponential(frequency: float, tau: float = 0.2,
                    decay: bool = True,
                    name: str = "ARB1") -> ArbitraryWaveform:
        n = compute_optimal_points(frequency)
        t = np.linspace(0, 1, n, endpoint=False)
        data = np.exp(-t / tau) if decay else 1.0 - np.exp(-t / tau)
        data = 2.0 * data - 1.0
        return _build(name, data, frequency, n,
                      f"Exp {'decay' if decay else 'rise'} τ={tau:.2f}")

    @staticmethod
    def frequency_comb(frequencies: list[float],
                       monte_carlo_iter: int = 1000,
                       name: str = "ARB1",
                       seed: int | None = None) -> ArbitraryWaveform:
        """
        Sum of sinusoids at the given tones, with Monte Carlo phase
        optimisation for maximum RMS (flatter envelope ⇒ better SNR
        after normalisation to ±1).
        """
        freqs = np.array(frequencies, dtype=float)
        if freqs.size == 0:
            raise ValueError("At least one frequency required")

        n, info = compute_optimal_points_for_comb(list(freqs))
        fundamental = info["fundamental"]
        t = np.linspace(0, 1.0 / fundamental, n, endpoint=False)

        rng = np.random.default_rng(seed)
        best_data = None
        best_rms = -1.0
        for _ in range(max(1, monte_carlo_iter)):
            phases = rng.uniform(0, 2 * np.pi, freqs.size)
            wave = np.zeros(n)
            for f, phi in zip(freqs, phases):
                wave += np.sin(2 * np.pi * f * t + phi)
            rms = float(np.sqrt(np.mean(wave ** 2)))
            if rms > best_rms:
                best_rms = rms
                best_data = wave.copy()

        max_abs = float(np.max(np.abs(best_data)))
        if max_abs > 0:
            best_data /= max_abs

        wf = ArbitraryWaveform(name=name, data=best_data,
                               frequency=fundamental,
                               sample_rate=info["sample_rate"],
                               num_points=n,
                               description=(
                                   f"Freq comb: {freqs.size} tones, "
                                   f"f0={fundamental:.4g} Hz, "
                                   f"fmax={info['f_max']:.4g} Hz, "
                                   f"{n} pts, {monte_carlo_iter} MC trials"),
                               comb_info=info)
        return wf


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build(name: str, data: np.ndarray, frequency: float,
           n: int, description: str) -> ArbitraryWaveform:
    max_abs = float(np.max(np.abs(data))) if data.size else 0.0
    if max_abs > 0:
        data = data / max_abs
    return ArbitraryWaveform(
        name=name, data=data,
        frequency=frequency,
        sample_rate=frequency * n,
        num_points=n,
        description=description,
    )
