"""Circle-based registration without modifying source measurements.

Transforms map left pixel centres to right pixel centres. Positive angles
are clockwise in image coordinates. Radius ratio supplies the fixed scale.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.radius)) or self.radius <= 0:
            raise ValueError('Нужны конечные координаты и положительный радиус')


@dataclass(frozen=True)
class Registration:
    left: Circle
    right: Circle
    angle: float
    dx: float = 0.0
    dy: float = 0.0
    score: float = 0.0
    overlap: float = 0.0

    def coordinates(self, x, y):
        theta = math.radians(self.angle)
        c, s = math.cos(theta), math.sin(theta)
        scale = self.right.radius / self.left.radius
        u, v = (x - self.left.x) * scale, (y - self.left.y) * scale
        return self.right.x + self.dx + c*u - s*v, self.right.y + self.dy + s*u + c*v


def sample(image, x, y):
    """Bilinear interpolation with explicit validity, never edge wrapping."""
    h, w = image.shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= w-1) & (y <= h-1)
    x = np.clip(np.where(valid, x, 0), 0, w-1)
    y = np.clip(np.where(valid, y, 0), 0, h-1)
    ix, iy = x.astype(np.int64), y.astype(np.int64)
    jx, jy = np.minimum(ix+1, w-1), np.minimum(iy+1, h-1)
    fx, fy = x-ix, y-iy
    a, b = image[iy, ix].astype(float), image[iy, jx].astype(float)
    c, d = image[jy, ix].astype(float), image[jy, jx].astype(float)
    values = (1-fy)*((1-fx)*a+fx*b) + fy*((1-fx)*c+fx*d)
    valid &= np.isfinite(values)
    return np.where(valid, values, 0), valid


def inside(x, y, circle):
    return (x-circle.x)**2 + (y-circle.y)**2 <= circle.radius**2


def validate(image):
    if image.ndim != 2 or min(image.shape) < 2 or image.dtype.kind not in 'uif':
        raise ValueError('Нужен двумерный числовой снимок не меньше 2×2')


def grid(image, circle, limit):
    h, w = image.shape
    x0, x1 = max(0, circle.x-circle.radius), min(w-1, circle.x+circle.radius)
    y0, y1 = max(0, circle.y-circle.radius), min(h-1, circle.y+circle.radius)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Круг не пересекается со снимком')
    step = max(1.0, max(x1-x0, y1-y0)/(limit-1))
    x, y = np.meshgrid(np.arange(x0, x1+1e-6, step), np.arange(y0, y1+1e-6, step))
    values, valid = sample(image, x, y)
    valid &= inside(x, y, circle)
    if np.count_nonzero(valid) < 64:
        raise ValueError('Слишком мало пикселей внутри круга')
    return x[valid], y[valid], values[valid]


def refine(left_image, right_image, left, right, angle=0.0, cancelled=None):
    """Refine rotation from local structure; preserve the supplied circles."""
    from rotation_registration import refine_rotation
    return refine_rotation(left_image, right_image, left, right, angle, cancelled)


def preview(left_image, right_image, transform, limit=900):
    """Red = left, cyan = right; independently scaled for detail comparison."""
    validate(left_image)
    validate(right_image)
    h, w = left_image.shape
    step = max(1.0, max(h, w)/limit)
    x, y = np.meshgrid(np.arange(0, w-1+1e-6, step), np.arange(0, h-1+1e-6, step))
    a, va = sample(left_image, x, y)
    va &= inside(x, y, transform.left)
    mx, my = transform.coordinates(x, y)
    b, vb = sample(right_image, mx, my)
    vb &= inside(mx, my, transform.right)

    def display(values, valid):
        result = np.zeros(values.shape, dtype=np.uint8)
        if np.any(valid):
            low, high = np.percentile(values[valid], (1, 99))
            result[valid] = np.rint(np.clip((values[valid]-low)/max(high-low, 1e-12), 0, 1)*255).astype(np.uint8)
        return result

    red, cyan = display(a, va), display(b, vb)
    return np.stack((red, cyan, cyan), axis=-1)
