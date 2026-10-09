"""TIFF samples stay unchanged; display conversion is separate from measurement."""
from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


@dataclass(frozen=True)
class Capture:
    path: Path
    sample: str
    sample_source: str
    mode: str
    raw: bool


def discover(folder: Path) -> list[Capture]:
    captures = []
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.tif', '.tiff'):
            continue
        stem = path.stem
        raw = stem.lower().endswith('.raw16')
        base = stem[:-6] if raw else stem
        match = re.search(r'\(([^()]*)\)$', base)
        mode = match.group(1) if match else 'Режим не указан'
        # Only an exact capture-name suffix links a labelled JPEG to its TIFF.
        labels = set()
        for sibling in path.parent.iterdir():
            if sibling.suffix.lower() in ('.jpg', '.jpeg') and sibling.stem.endswith('_' + base):
                labels.add(sibling.stem[:-(len(base) + 1)])
        if len(labels) == 1:
            sample = next(iter(labels))
            source = 'Имя соответствующего JPG'
        else:
            sample = path.parent.name
            source = 'Имя папки; отдельный образец не определён'
        captures.append(Capture(path, sample, source, mode, raw))
    return sorted(captures, key=lambda c: (str(c.path.parent), c.sample, not c.raw, c.path.name))


def scn_metadata(path: Path) -> dict[str, str]:
    """Read XML metadata only; do not decode the proprietary SCN image payload."""
    if not path.exists():
        return {'SCN': 'Нет соседнего файла; доступны имя TIFF и TIFF-теги'}
    values = {'SCN': path.name}
    try:
        data = path.read_bytes()
        # Image Lab's binary MIME parts are not reliably parsed by email.parser.
        # Extract complete XML roots, without interpreting binary image data.
        for block in re.finditer(rb'<root\b[^>]*>.*?</root>', data, re.DOTALL):
            if len(block.group()) > 1_000_000:
                continue
            try:
                root = ET.fromstring(block.group())
            except ET.ParseError:
                continue
            settings = root.find('protocol_settings')
            if settings is not None:
                for node in settings.iter():
                    if len(node) == 0 and node.text and node.text.strip():
                        values[node.tag] = node.text.strip()
    except OSError as error:
        values['SCN error'] = str(error)
    return values


def capture_metadata(capture: Capture) -> dict[str, str]:
    base = capture.path.stem
    if capture.raw:
        base = base[:-6]
    base = re.sub(r'\([^()]*\)$', '', base)
    return scn_metadata(capture.path.with_name(base + '.scn'))


@dataclass
class Frame:
    pixels: np.ndarray
    tags: dict[str, str]
    white_is_zero: bool
    page_count: int


def read_frame(path: Path, page_index: int = 0) -> Frame:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[page_index]
        pixels = page.asarray()
        photo = int(page.photometric)
        if photo == 2 and pixels.ndim == 3 and int(page.planarconfig) == 2:
            pixels = np.moveaxis(pixels, 0, -1)
        supported = (photo in (0, 1) and pixels.ndim == 2) or (
            photo == 2 and pixels.ndim == 3 and pixels.shape[-1] in (3, 4)
        )
        if not supported or pixels.dtype.kind not in 'uif':
            raise ValueError(f'Неподдерживаемый TIFF: shape={pixels.shape}, dtype={pixels.dtype}, photometric={photo}')
        names = ('Make', 'Model', 'Software', 'DateTime', 'ImageDescription',
                 'BitsPerSample', 'PhotometricInterpretation', 'Orientation',
                 'XResolution', 'YResolution', 'ResolutionUnit')
        tags = {name: str(page.tags[name].value) for name in names if name in page.tags}
        tags['Координаты'] = 'Массив TIFF: x — столбец, y — строка, начало (0, 0); без поворота'
        pixels.setflags(write=False)
        return Frame(pixels, tags, photo == 0, len(tif.pages))


def pixel_value(pixels: np.ndarray, x: int, y: int):
    if not (0 <= x < pixels.shape[1] and 0 <= y < pixels.shape[0]):
        raise IndexError('Пиксель вне изображения')
    return pixels[y, x].tolist()


def display_limits(pixels: np.ndarray, auto: bool = True) -> tuple[float, float]:
    values = pixels[..., :3] if pixels.ndim == 3 else pixels
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError('Изображение не содержит конечных значений')
    low, high = np.percentile(finite, (1, 99.5)) if auto else (finite.min(), finite.max())
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    return float(low), float(high) if high > low else float(low) + 1.0


def render_pixels(pixels: np.ndarray, low: float, high: float,
                  gamma: float = 1.0, invert: bool = False) -> np.ndarray:
    if not all(np.isfinite(v) for v in (low, high, gamma)) or high <= low or gamma <= 0:
        raise ValueError('Нужны конечные числа: максимум > минимума, гамма > 0')
    values = pixels[..., :3] if pixels.ndim == 3 else pixels
    scaled = np.array(values, dtype=np.float64, copy=True)
    scaled = np.clip((scaled - low) / (high - low), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0, posinf=1, neginf=0)
    scaled = scaled ** (1.0 / gamma)
    if invert:
        scaled = 1.0 - scaled
    return np.rint(scaled * 255).astype(np.uint8)


def image_coordinates(canvas_x: float, canvas_y: float, display_size: tuple[int, int],
                      source_size: tuple[int, int]) -> tuple[int, int] | None:
    width, height = display_size
    if width <= 0 or height <= 0 or not (0 <= canvas_x < width and 0 <= canvas_y < height):
        return None
    return int(canvas_x * source_size[0] / width), int(canvas_y * source_size[1] / height)
