"""FMCW radar simulator and basic range/Doppler processing.

The model is deliberately simple and readable: ideal point targets, a
perfectly linear chirp, and a dechirped (mixed and low-pass filtered) IF
signal. That is enough to show every idea in season 1, and each
simplification is called out where it matters.

Conventions
-----------
- Radial velocity is positive when the target moves *away* from the radar.
- The data cube has shape ``(n_chirps, n_samples)``: one row per chirp
  ("slow time"), one column per ADC sample ("fast time").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = 299_792_458.0  # speed of light, m/s


@dataclass(frozen=True)
class RadarConfig:
    """Chirp and ADC parameters.

    The defaults match the numbers used in episode 1: a 77-81 GHz sweep at
    30 MHz/us, sampled so the ADC window covers the whole ramp. That makes
    the range resolution exactly c / 2B (3.75 cm).
    """

    f_start: float = 77e9  # Hz
    bandwidth: float = 4e9  # Hz, swept during the ADC window
    slope: float = 30e12  # Hz/s (30 MHz/us)
    sample_rate: float = 3.84e6  # Hz, complex (I/Q) samples per second
    n_samples: int = 512  # ADC samples per chirp
    chirp_period: float = 160e-6  # s, ramp plus idle time
    n_chirps: int = 128

    @property
    def ramp_time(self) -> float:
        return self.bandwidth / self.slope

    @property
    def wavelength(self) -> float:
        return C / self.f_start

    @property
    def range_resolution(self) -> float:
        return C / (2 * self.bandwidth)

    @property
    def max_range(self) -> float:
        # Complex sampling: beat frequencies up to fs are unambiguous.
        return self.sample_rate * C / (2 * self.slope)

    @property
    def max_velocity(self) -> float:
        # Phase step between chirps must stay within +/- pi.
        return self.wavelength / (4 * self.chirp_period)

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2 * self.n_chirps * self.chirp_period)


@dataclass(frozen=True)
class Target:
    """An ideal point reflector."""

    range: float  # m, at the start of the frame
    velocity: float = 0.0  # m/s, positive = moving away
    amplitude: float = 1.0  # linear, at the IF output


def beat_frequency(config: RadarConfig, range_m: float) -> float:
    """f_b = S * 2R / c: the tone a target at ``range_m`` produces."""
    return config.slope * 2 * range_m / C


def phase_shift(config: RadarConfig, displacement_m: float) -> float:
    """Echo phase change in radians when a target moves by ``displacement_m``."""
    return 4 * np.pi * displacement_m / config.wavelength


def fast_time(config: RadarConfig) -> np.ndarray:
    return np.arange(config.n_samples) / config.sample_rate


def slow_time(config: RadarConfig) -> np.ndarray:
    return np.arange(config.n_chirps) * config.chirp_period


def simulate_if(
    config: RadarConfig,
    targets: list[Target],
    noise_std: float = 0.0,
    complex_if: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate the dechirped IF signal for one frame.

    Each target contributes ``a * exp(j * 2*pi * (f0*tau + S*tau*t))`` where
    ``tau`` is its round-trip delay at that instant. The ``f0*tau`` term is
    what rotates the phase from chirp to chirp (Doppler); the ``S*tau*t``
    term is the beat tone (range). The tiny residual video phase term is
    ignored.

    Returns a ``(n_chirps, n_samples)`` array: complex I/Q by default, or
    only the real (I) channel when ``complex_if`` is False.
    """
    t = fast_time(config)[np.newaxis, :]
    t_slow = slow_time(config)[:, np.newaxis]
    cube = np.zeros((config.n_chirps, config.n_samples), dtype=complex)

    for target in targets:
        r = target.range + target.velocity * (t_slow + t)
        tau = 2 * r / C
        phase = 2 * np.pi * (config.f_start * tau + config.slope * tau * t)
        cube += target.amplitude * np.exp(1j * phase)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=cube.shape) + 1j * rng.normal(size=cube.shape)
        cube += noise_std / np.sqrt(2) * noise

    return cube if complex_if else cube.real


def range_axis(config: RadarConfig, n_fft: int | None = None) -> np.ndarray:
    """Range in metres for each bin of :func:`range_fft`."""
    n_fft = n_fft or config.n_samples
    freqs = np.arange(n_fft) * config.sample_rate / n_fft
    return freqs * C / (2 * config.slope)


def velocity_axis(config: RadarConfig, n_fft: int | None = None) -> np.ndarray:
    """Velocity in m/s for each bin of :func:`doppler_fft` (zero-centred)."""
    n_fft = n_fft or config.n_chirps
    doppler = np.fft.fftshift(np.fft.fftfreq(n_fft, d=config.chirp_period))
    # A target moving away lengthens tau, so its IF phase rotates the
    # positive way: positive Doppler bin = positive (receding) velocity.
    return doppler * config.wavelength / 2


def range_fft(
    cube: np.ndarray, window: np.ndarray | None = None, n_fft: int | None = None
) -> np.ndarray:
    """FFT along fast time: each chirp becomes a range profile."""
    if window is not None:
        cube = cube * window[np.newaxis, :]
    return np.fft.fft(cube, n=n_fft, axis=-1)


def doppler_fft(
    range_profiles: np.ndarray,
    window: np.ndarray | None = None,
    n_fft: int | None = None,
) -> np.ndarray:
    """FFT along slow time (across chirps), zero velocity in the middle."""
    if window is not None:
        range_profiles = range_profiles * window[:, np.newaxis]
    return np.fft.fftshift(np.fft.fft(range_profiles, n=n_fft, axis=0), axes=0)


def range_doppler_map(cube: np.ndarray, db: bool = True) -> np.ndarray:
    """Both FFTs in one call. Returns ``(n_velocity, n_range)`` magnitudes."""
    rd = np.abs(doppler_fft(range_fft(cube)))
    if db:
        return 20 * np.log10(rd + 1e-12)
    return rd
