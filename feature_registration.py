"""SIFT correspondences and bounded rigid consensus in original pixel coordinates."""
from dataclasses import dataclass
import math
import numpy as np
from registration import Registration, validate


@dataclass(frozen=True)
class FeatureResult:
    transform: Registration
    left_points: np.ndarray
    right_points: np.ndarray
    errors: np.ndarray
    matches: int

    @property
    def rms(self):
        return float(np.sqrt(np.mean(self.errors ** 2)))


def check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError('Поиск соответствий отменён')


def fit_correspondences(a, b, left, right, angle=0.0, cancelled=None):
    """Fit rotation/translation at fixed circle scale; reject weak or ambiguous support."""
    check_cancelled(cancelled)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (a.ndim != 2 or a.shape[1:] != (2,) or b.shape != a.shape
            or len(a) < 8 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or not math.isfinite(angle)):
        raise ValueError('Недостаточно корректных пар характерных точек: требуется минимум 8')
    scale = right.radius / left.radius
    source = (a - [left.x, left.y]) * scale
    target = b - [right.x, right.y]
    tolerance = max(1.5, right.radius * 0.007)
    bound = right.radius * 0.12

    def model(indices):
        u, v = source[indices], target[indices]
        uc, vc = u.mean(axis=0), v.mean(axis=0)
        u, v = u - uc, v - vc
        if np.sum(u * u) < (right.radius * 0.08) ** 2:
            return None
        theta = math.degrees(math.atan2(np.sum(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]), np.sum(u*v)))
        delta = (theta-angle+180) % 360-180
        theta = angle + delta
        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        rotation = np.array([[c, -s], [s, c]])
        shift = vc - rotation @ uc
        if abs(delta) > 15 or np.max(np.abs(shift)) > bound:
            return None
        errors = np.linalg.norm(source @ rotation.T + shift - target, axis=1)
        return theta, shift, errors

    rng = np.random.default_rng(481)
    hypotheses = []
    best = None
    best_count = 0
    best_error = float('inf')
    for iteration in range(1600):
        if iteration % 32 == 0:
            check_cancelled(cancelled)
        proposal = model(rng.choice(len(a), 2, replace=False))
        if proposal is None:
            continue
        mask = proposal[2] <= tolerance
        count = int(mask.sum())
        error = float(np.median(proposal[2][mask])) if count else float('inf')
        hypotheses.append((proposal[0], proposal[1], count))
        if count > best_count or (count == best_count and error < best_error):
            best, best_count, best_error = proposal, count, error
    if best is None or best_count < max(8, math.ceil(len(a) * 0.30)):
        raise ValueError('Пары точек не подтверждают общее положение. Ручное совмещение сохранено.')
    for _ in range(4):
        mask = best[2] <= tolerance
        fitted = model(np.flatnonzero(mask))
        if fitted is None:
            raise ValueError('Поправка выходит за пределы поиска. Уточните круги и ручной угол.')
        best = fitted
    mask = best[2] <= tolerance
    count = int(mask.sum())
    if count < max(8, math.ceil(len(a) * 0.30)):
        raise ValueError('Недостаточно согласованных соответствий после уточнения')
    # A cluster or line cannot reliably establish the pose of the whole wafer.
    covariance = np.cov(source[mask].T)
    if np.linalg.eigvalsh(covariance)[0] < (right.radius * 0.08) ** 2:
        raise ValueError('Совпадения сосредоточены на слишком маленьком участке пластины')
    for theta, shift, support in hypotheses:
        separated = abs(theta-best[0]) > 3 or np.linalg.norm(shift-best[1]) > 3*tolerance
        if separated and support >= count * 0.85:
            raise ValueError('Найдено несколько сопоставимых вариантов совмещения')
    if abs(best[0]-angle) > 14.8 or np.max(np.abs(best[1])) > bound * 0.99:
        raise ValueError('Достигнута граница поиска. Уточните начальные круги и угол.')
    check_cancelled(cancelled)
    transform = Registration(left, right, float(best[0]), float(best[1][0]), float(best[1][1]),
                             score=count/len(a))
    return FeatureResult(transform, a[mask].copy(), b[mask].copy(), best[2][mask].copy(), len(a))


def extract_features(image, circle, cancelled=None):
    check_cancelled(cancelled)
    validate(image)
    try:
        import cv2
    except ImportError as error:
        raise ValueError('Для поиска точек установите зависимости проекта: нужен opencv-python-headless') from error
    h, w = image.shape
    factor = min(1.0, 1400 / max(h, w))
    width, height = max(2, round(w*factor)), max(2, round(h*factor))
    values = cv2.resize(np.asarray(image, dtype=np.float32), (width, height), interpolation=cv2.INTER_AREA)
    y, x = np.mgrid[:height, :width]
    ox, oy = (x+0.5)*w/width-0.5, (y+0.5)*h/height-0.5
    mask = ((ox-circle.x)**2+(oy-circle.y)**2 < (circle.radius*0.92)**2) & np.isfinite(values)
    if mask.sum() < 256:
        raise ValueError('Недостаточно видимой области внутри пластины')
    low, high = np.percentile(values[mask], (1, 99))
    if high-low < 1e-6:
        raise ValueError('Внутри пластины нет различимых деталей')
    gray = np.rint(np.clip(np.nan_to_num((values-low)/(high-low)), 0, 1)*255).astype(np.uint8)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    detector = cv2.SIFT_create(nfeatures=1600, contrastThreshold=0.012, edgeThreshold=12)
    keypoints, descriptors = detector.detectAndCompute(gray, mask.astype(np.uint8)*255)
    check_cancelled(cancelled)
    if descriptors is None or len(keypoints) < 8:
        raise ValueError('Недостаточно характерных точек внутри пластины')
    # SIFT can assign several orientations to one point. Count each location once.
    unique = {}
    for i, keypoint in enumerate(keypoints):
        key = (round(keypoint.pt[0]), round(keypoint.pt[1]))
        if key not in unique or keypoint.response > keypoints[unique[key]].response:
            unique[key] = i
    indices = list(unique.values())
    points = np.array([keypoints[i].pt for i in indices], dtype=float)
    points[:, 0] = (points[:, 0]+0.5)*w/width-0.5
    points[:, 1] = (points[:, 1]+0.5)*h/height-0.5
    return points, descriptors[indices]


def refine_features(left_image, right_image, left, right, angle=0.0, cancelled=None):
    a, da = extract_features(left_image, left, cancelled)
    b, db = extract_features(right_image, right, cancelled)
    import cv2
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def distinctive(first, second):
        pairs = matcher.knnMatch(first, second, k=2)
        return {pair[0].queryIdx: pair[0].trainIdx for pair in pairs
                if len(pair) == 2 and pair[0].distance < 0.78*pair[1].distance}

    forward, reverse = distinctive(da, db), distinctive(db, da)
    check_cancelled(cancelled)
    pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    if len(pairs) < 8:
        raise ValueError('Мало взаимных совпадений деталей. Укажите общие ориентиры вручную.')
    ia, ib = np.asarray(pairs).T
    return fit_correspondences(a[ia], b[ib], left, right, angle, cancelled)
