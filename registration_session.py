"""Versioned alignment parameters in original TIFF pixel coordinates."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile

from registration import Circle, Registration


CONVENTION = 'left_pixel_centres_to_right_pixel_centres; x=column,y=row; clockwise_degrees'


def number(value, low=-1e9, high=1e9):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Недопустимое числовое значение параметра')
    return float(value)


def validate_sources(sources):
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError('Нужны сведения о двух исходных снимках')
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError('Некорректные сведения об исходном снимке')
        if not isinstance(source.get('path'), str) or not source['path']:
            raise ValueError('Не указан исходный файл TIFF')
        if type(source.get('page')) is not int or source['page'] < 0:
            raise ValueError('Некорректный номер страницы TIFF')
        shape = source.get('shape')
        if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v < 2 for v in shape):
            raise ValueError('Некорректный размер исходного массива')
        if source.get('dtype') != 'uint16' or type(source.get('white_is_zero')) is not bool:
            raise ValueError('Ожидается исходный RAW16 и его полярность')
        for key in ('size_bytes', 'mtime_ns'):
            if type(source.get(key)) is not int or source[key] < 0:
                raise ValueError('Нет сведений о версии исходного файла')


def matrix_for(transform):
    angle = math.radians(transform.angle)
    scale = transform.right.radius/transform.left.radius
    c, s = scale*math.cos(angle), scale*math.sin(angle)
    return [[c, -s, transform.right.x+transform.dx-c*transform.left.x+s*transform.left.y],
            [s, c, transform.right.y+transform.dy-s*transform.left.x-c*transform.left.y],
            [0.0, 0.0, 1.0]]


def make_session(sources, transform, orientations, mode, opacity, colors, refined=False):
    document = {
        'format': 'WaferPL alignment', 'version': 1, 'coordinates': CONVENTION,
        'sources': sources, 'transform': asdict(transform),
        'matrix_left_to_right': matrix_for(transform),
        'orientations': list(orientations), 'refined': refined,
        'display': {'mode': mode, 'opacity': opacity, 'colors': [list(c) for c in colors],
                    'normalization': 'independent_percentiles_1_99_display_only'},
    }
    validate_session(document, sources)
    return document


def validate_session(document, expected_sources):
    """Validate everything before applying any state to the UI."""
    try:
        if (document['format'] != 'WaferPL alignment' or type(document['version']) is not int
                or document['version'] != 1 or document['coordinates'] != CONVENTION):
            raise ValueError('Неизвестный формат параметров совмещения')
        validate_sources(document['sources'])
        validate_sources(expected_sources)
        if document['sources'] != expected_sources:
            raise ValueError('Параметры относятся к другим файлам, страницам или версиям снимков. Откройте исходную пару в том же порядке.')
        raw = document['transform']
        circles = []
        for key in ('left', 'right'):
            c = raw[key]
            circles.append(Circle(number(c['x']), number(c['y']), number(c['radius'], 1, 1e9)))
        if circles[0].radius != circles[1].radius:
            raise ValueError('В этой версии радиусы кругов должны совпадать')
        transform = Registration(*circles, number(raw['angle'], -360, 360),
                                 number(raw['dx'], 0, 0), number(raw['dy'], 0, 0),
                                 number(raw['score'], -1.000001, 1.000001), number(raw['overlap'], 0, 1))
        orientations = document['orientations']
        if not isinstance(orientations, list) or len(orientations) != 2:
            raise ValueError('Нужны два угла ориентации')
        orientations = [number(v, -180, 180) for v in orientations]
        difference = (orientations[1]-orientations[0]-transform.angle+180) % 360-180
        if abs(difference) > 1e-6:
            raise ValueError('Углы и преобразование не согласованы')
        matrix = document['matrix_left_to_right']
        expected = matrix_for(transform)
        if not isinstance(matrix, list) or len(matrix) != 3:
            raise ValueError('Некорректная матрица преобразования')
        for row, target in zip(matrix, expected):
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError('Некорректная матрица преобразования')
            for actual, value in zip(row, target):
                if not math.isclose(number(actual, -1e12, 1e12), value, rel_tol=1e-10, abs_tol=1e-7):
                    raise ValueError('Матрица и круги не согласованы')
        display = document['display']
        if display['mode'] not in ('blend', 'color') or display['normalization'] != 'independent_percentiles_1_99_display_only':
            raise ValueError('Неизвестный режим отображения')
        if type(display['opacity']) is not int or not 0 <= display['opacity'] <= 100:
            raise ValueError('Некорректная прозрачность')
        colors = display['colors']
        if not isinstance(colors, list) or len(colors) != 2:
            raise ValueError('Нужны два цвета')
        for color in colors:
            if not isinstance(color, list) or len(color) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in color):
                raise ValueError('Некорректный цвет RGB')
        if type(document['refined']) is not bool:
            raise ValueError('Некорректный признак уточнения')
        return transform, orientations, display, document['refined']
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Неполный или повреждённый файл параметров') from error


def save_session(path, document):
    validate_session(document, document.get('sources'))
    text = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.waferpl-', suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def load_session(path, expected_sources):
    with open(path, encoding='utf-8') as stream:
        document = json.load(stream)
    validate_session(document, expected_sources)
    return document
