"""Display-only robust contrast; raw measurements are never transformed."""
import numpy as np


def detail_settings(pixels, invert=False):
    values = pixels[np.isfinite(pixels)]
    if not values.size:
        raise ValueError('Нет конечных значений')
    fitting = values
    if pixels.dtype.kind == 'u':
        maximum = np.iinfo(pixels.dtype).max
        interior = values[(values > 0) & (values < maximum)]
        if interior.size:
            fitting = interior
    low, middle, high = np.percentile(fitting, [0.5, 50, 99.5])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        low, high = float(low) - 0.5, float(low) + 0.5
    fraction = np.clip((middle - low) / (high - low), 0.001, 0.999)
    # Inversion follows gamma in render_pixels; target displayed median 45%.
    target = 0.55 if invert else 0.45
    gamma = float(np.clip(np.log(fraction) / np.log(target), 0.4, 3.0))
    return float(low), float(high), gamma
