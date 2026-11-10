"""Checks that the simulator reproduces the numbers quoted in episode 1."""

import numpy as np
import pytest

from radar_signal_processing import (
    RadarConfig,
    Target,
    beat_frequency,
    phase_shift,
    range_axis,
    range_doppler_map,
    range_fft,
    simulate_if,
    velocity_axis,
)

CFG = RadarConfig()


def test_config_matches_episode_1():
    assert CFG.ramp_time == pytest.approx(133.3e-6, rel=1e-3)
    assert CFG.range_resolution == pytest.approx(0.0375, rel=1e-2)
    assert CFG.wavelength == pytest.approx(3.9e-3, rel=1e-2)
    # ADC window spans the whole sweep, so FFT bins are c/2B apart.
    assert CFG.n_samples / CFG.sample_rate == pytest.approx(CFG.ramp_time)
    assert range_axis(CFG)[1] == pytest.approx(CFG.range_resolution)


def test_beat_frequency_worked_example():
    assert beat_frequency(CFG, 10.0) == pytest.approx(2.0e6, rel=1e-3)
    assert beat_frequency(CFG, 14.0) == pytest.approx(2.8e6, rel=1e-3)


def test_phase_per_millimetre():
    assert np.degrees(phase_shift(CFG, 1e-3)) == pytest.approx(185, abs=1)


@pytest.mark.parametrize("ranges", [[10.0], [10.0, 14.0], [3.2]])
def test_range_profile_peaks(ranges):
    cube = simulate_if(CFG, [Target(r) for r in ranges])
    spectrum = np.abs(range_fft(cube)[0])
    peaks = np.sort(np.argsort(spectrum)[-len(ranges):])
    found = range_axis(CFG)[peaks]
    assert found == pytest.approx(ranges, abs=CFG.range_resolution)


@pytest.mark.parametrize("velocity", [-1.1, 0.0, 2.5])
def test_range_doppler_peak(velocity):
    cube = simulate_if(CFG, [Target(3.2, velocity)])
    rd = range_doppler_map(cube)
    v_idx, r_idx = np.unravel_index(np.argmax(rd), rd.shape)
    assert range_axis(CFG)[r_idx] == pytest.approx(3.2, abs=2 * CFG.range_resolution)
    assert velocity_axis(CFG)[v_idx] == pytest.approx(
        velocity, abs=CFG.velocity_resolution
    )


def test_real_if_and_noise():
    cube = simulate_if(CFG, [Target(10.0)], noise_std=0.1, complex_if=False, seed=0)
    assert np.isrealobj(cube)
    assert cube.shape == (CFG.n_chirps, CFG.n_samples)
