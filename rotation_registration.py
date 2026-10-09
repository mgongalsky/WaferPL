"""Conservative rotation voting from local angular structure, with fixed circles.

This is a local ±15 degree search, not proof of physical feature identity.
Changed patches may disagree; a spatially distributed consensus is required.
"""
import math
import numpy as np
from registration import Registration, sample, validate


def refine_rotation(left_image, right_image, left, right, angle=0.0, cancelled=None):
    validate(left_image)
    validate(right_image)
    if not math.isfinite(angle):
        raise ValueError('Угол должен быть конечным')

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError('Поиск поворота отменён')

    check()
    # Stay away from the circular rim: a circular boundary cannot define angle.
    radii = np.linspace(0.20, 0.90, 48)[:, None]
    angles = np.deg2rad(np.arange(360))[None, :]

    def polar(image, circle, rotation):
        theta = angles + math.radians(rotation)
        x = circle.x + circle.radius*radii*np.cos(theta)
        y = circle.y + circle.radius*radii*np.sin(theta)
        values, valid = sample(image, x, y)
        gradient = (np.roll(values, -1, axis=1)-np.roll(values, 1, axis=1))/2
        valid &= np.roll(valid, -1, axis=1) & np.roll(valid, 1, axis=1)
        return values, gradient, valid

    values, reference, reference_valid = polar(left_image, left, 0)
    finite = values[reference_valid]
    if finite.size < 256:
        raise ValueError('Недостаточно видимой части пластины. Ручной угол сохранён.')
    contrast = float(np.percentile(finite, 95)-np.percentile(finite, 5))
    if contrast <= 1e-12:
        raise ValueError('Нет различимых деталей для поворота. Ручной угол сохранён.')
    # Reject radial shading and interpolation noise around a featureless circle.
    if np.std(reference[reference_valid]) < contrast*0.005:
        raise ValueError('Круг без направленных деталей не задаёт поворот. Ручной угол сохранён.')
    patches = [(slice(r*12, (r+1)*12), slice(s*30, (s+1)*30))
               for r in range(4) for s in range(12)]
    candidates = angle + np.linspace(-15, 15, 61)
    scores = np.full((len(candidates), len(patches)), np.nan)
    coverage = np.zeros(len(candidates))
    for index, theta in enumerate(candidates):
        check()
        _, other, other_valid = polar(right_image, right, float(theta))
        common = reference_valid & other_valid
        coverage[index] = np.count_nonzero(common)/np.count_nonzero(reference_valid)
        for patch_index, patch in enumerate(patches):
            valid = common[patch]
            if np.count_nonzero(valid) < 180:
                continue
            a, b = reference[patch][valid], other[patch][valid]
            a, b = a-a.mean(), b-b.mean()
            sa, sb = float(np.sqrt(np.mean(a*a))), float(np.sqrt(np.mean(b*b)))
            if sa < contrast*0.005 or sb <= 1e-12:
                continue
            scores[index, patch_index] = float(np.mean(a*b)/(sa*sb))
    # Each patch votes only if its best angle is distinct from distant alternatives.
    votes = []
    informative = 0
    for patch_index in range(len(patches)):
        curve = scores[:, patch_index]
        if np.count_nonzero(np.isfinite(curve)) < 50:
            continue
        informative += 1
        best = int(np.nanargmax(curve))
        distant = np.abs(candidates-candidates[best]) >= 5
        alternatives = curve[distant & np.isfinite(curve)]
        if curve[best] < 0.70 or not alternatives.size or curve[best]-np.max(alternatives) < 0.04:
            continue
        if best in (0, len(candidates)-1):
            continue
        votes.append((patch_index, best, float(curve[best])))
    check()
    if len(votes) < 8:
        raise ValueError('Общих устойчивых деталей недостаточно. Ручной угол сохранён; укажите направление по знакомой метке.')
    vote_angles = np.array([candidates[v[1]] for v in votes])
    counts = np.array([np.count_nonzero(np.abs(vote_angles-theta) <= 2) for theta in candidates])
    winner = int(np.argmax(counts))
    supporters = [v for v, theta in zip(votes, vote_angles) if abs(theta-candidates[winner]) <= 2]
    sectors = {v[0] % 12 for v in supporters}
    bands = {v[0] // 12 for v in supporters}
    distant = np.abs(candidates-candidates[winner]) >= 6
    rival = int(np.max(counts[distant])) if np.any(distant) else 0
    required = max(8, math.ceil(informative*0.30))
    if (len(supporters) < required or len(sectors) < 3 or len(bands) < 2
            or rival >= len(supporters)*0.75):
        raise ValueError('Детали дают противоречивые варианты поворота. Ручной угол сохранён.')
    # Optimize only the consensus patches; changed areas cannot drag the centres.
    near = np.abs(candidates-candidates[winner]) <= 2
    consensus = np.nanmean(scores[:, [v[0] for v in supporters]], axis=1)
    chosen = int(np.argmax(np.where(near & np.isfinite(consensus), consensus, -np.inf)))
    if coverage[chosen] < 0.35 or abs(candidates[chosen]-angle) >= 14.5:
        raise ValueError('Мало перекрытия или поворот у границы поиска. Уточните ручной угол.')
    check()
    return Registration(left, right, float(candidates[chosen]), score=float(consensus[chosen]),
                        overlap=float(coverage[chosen]))
