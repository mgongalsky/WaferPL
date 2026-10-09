"""Display-only aligned layers; source measurements remain unchanged."""
import numpy as np
from registration import sample, inside, validate


def aligned_layers(images, transform, limit=900):
    left, right = images
    validate(left)
    validate(right)
    h, w = left.shape
    step = max(1.0, max(h, w)/limit)
    x, y = np.meshgrid(np.arange(0, w-1+1e-6, step), np.arange(0, h-1+1e-6, step))
    a, va = sample(left, x, y)
    va &= inside(x, y, transform.left)
    mx, my = transform.coordinates(x, y)
    b, vb = sample(right, mx, my)
    vb &= inside(mx, my, transform.right)

    def display(values, valid):
        output = np.zeros(values.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.percentile(values[valid], (1, 99))
            output[valid] = np.rint(np.clip((values[valid]-low)/max(high-low, 1e-12), 0, 1)*255).astype(np.uint8)
        return output

    return display(a, va), display(b, vb), va, vb


def compose_layers(layers, mode='blend', alpha=0.5, colors=((255, 0, 0), (0, 255, 255))):
    if mode not in ('blend', 'color') or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('Недопустимый режим или прозрачность')
    palette = np.array(colors, dtype=float, copy=True)
    if palette.shape != (2, 3) or not np.all(np.isfinite(palette)) or np.any((palette < 0) | (palette > 255)):
        raise ValueError('Нужны два цвета RGB в диапазоне 0–255')
    a, b, va, vb = layers
    if mode == 'color':
        # Keep the existing full-color overlay at the midpoint, then fade
        # either source to zero towards the opposite endpoint.
        gains = np.array([min(1.0, 2 * (1 - alpha)), min(1.0, 2 * alpha)])
        palette = palette / 255 * gains[:, None]
        weights = np.maximum(1.0, palette.sum(axis=0))
        rgb = (np.where(va, a, 0)[..., None] * palette[0]
               + np.where(vb, b, 0)[..., None] * palette[1]) / weights
        return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
    wa, wb = (1 - alpha) * va, alpha * vb
    weight = wa + wb
    gray = np.divide(a * wa + b * wb, weight, out=np.zeros(a.shape, dtype=float), where=weight > 0)
    return np.repeat(np.rint(gray).astype(np.uint8)[..., None], 3, axis=-1)
