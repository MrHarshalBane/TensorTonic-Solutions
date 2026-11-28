import math
from typing import List


def generate_anchors(
    feature_size: int,
    image_size: int,
    scales: List[float],
    aspect_ratios: List[float],
) -> List[List[float]]:
    stride = image_size / feature_size

    # Precompute half-widths and half-heights for all (scale, aspect_ratio) pairs
    box_halves = []
    for s in scales:
        for r in aspect_ratios:
            sqrt_r = math.sqrt(r)
            half_w = (s * sqrt_r) / 2.0
            half_h = (s / sqrt_r) / 2.0
            box_halves.append((half_w, half_h))

    anchors: List[List[float]] = []
    for i in range(feature_size):
        cy = (i + 0.5) * stride
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            for half_w, half_h in box_halves:
                anchors.append([cx - half_w, cy - half_h, cx + half_w, cy + half_h])

    return anchors
