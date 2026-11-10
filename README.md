# radar_signal_processing

Code for **Signals & Systems Through Radar**, a series on the [Electrons to Vision](https://www.youtube.com/@ElectronstoVision) YouTube channel.

The series teaches signals and systems (sampling, I/Q, Fourier transforms, the FFT, windowing, convolution, filters and detection) through one real sensor: FMCW radar. Every number and plot in the videos comes from the simulator in this repo, so you can reproduce and change all of them.

## What's here

- `radar_signal_processing.fmcw`: an FMCW radar simulator (point targets, linear chirps, dechirped I/Q signal) plus range and Doppler FFT processing.
- `notebooks/`: one notebook per episode (coming as episodes are released).
- `tests/`: checks that the simulator reproduces the numbers quoted in the videos.

## Quick start

```bash
git clone https://github.com/Electrons-to-Vision/radar_signal_processing.git
cd radar_signal_processing
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

```python
from radar_signal_processing import RadarConfig, Target, simulate_if, range_doppler_map

cfg = RadarConfig()                      # 77-81 GHz, 30 MHz/us chirp
person = Target(range=3.2, velocity=-1.1)  # 3.2 m away, walking towards the radar
rd = range_doppler_map(simulate_if(cfg, [person], noise_std=0.05))
```

## Default radar parameters

| Parameter | Value |
|---|---|
| Sweep | 77 → 81 GHz (B = 4 GHz) |
| Chirp slope | 30 MHz/µs |
| Range resolution | c / 2B ≈ 3.75 cm |
| Max range (complex sampling, 3.84 MHz) | ≈ 19.2 m |
| Chirps per frame | 128, every 160 µs |
| Max / resolution velocity | ≈ ±6.1 m/s / ≈ 0.095 m/s |

## Episodes

| # | Topic | Notebook |
|---|---|---|
| 1 | How a tiny radar chip sees distance and speed | coming soon |

*From electrons to intelligent machines.*

## License

MIT, see [LICENSE](LICENSE).
