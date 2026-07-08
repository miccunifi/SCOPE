from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from utils import box_area_xyxy


Tile = Tuple[int, int, int, int]


def generate_tiles(width: int, height: int, tile_size: int) -> List[Tile]:
    tiles: List[Tile] = []
    for y1 in range(0, height, tile_size):
        for x1 in range(0, width, tile_size):
            x2 = min(width, x1 + tile_size)
            y2 = min(height, y1 + tile_size)
            tiles.append((x1, y1, x2, y2))
    return tiles


def tile_gradient_scores(img_bgr: np.ndarray, tile_size: int) -> List[Tuple[Tile, float]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    h, w = gray.shape[:2]
    scores = []
    for tile in generate_tiles(w, h, tile_size):
        x1, y1, x2, y2 = tile
        scores.append((tile, float(mag[y1:y2, x1:x2].mean())))
    return scores


def tile_entropy_scores(img_bgr: np.ndarray, tile_size: int) -> List[Tuple[Tile, float]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scores = []

    for tile in generate_tiles(w, h, tile_size):
        x1, y1, x2, y2 = tile
        patch = gray[y1:y2, x1:x2]
        hist = cv2.calcHist([patch], [0], None, [256], [0, 256]).flatten()
        prob = hist / max(1.0, hist.sum())
        prob = prob[prob > 0]
        entropy = float(-(prob * np.log2(prob)).sum())
        scores.append((tile, entropy))

    return scores


def tile_proposal_scores(
    img_bgr: np.ndarray,
    proposal_boxes: List[List[float]],
    tile_size: int,
) -> List[Tuple[Tile, float]]:
    h, w = img_bgr.shape[:2]
    tiles = generate_tiles(w, h, tile_size)
    scores = []

    for tile in tiles:
        x1, y1, x2, y2 = tile
        tile_box = [x1, y1, x2, y2]
        tile_area = box_area_xyxy(tile_box)
        score = 0.0

        for box in proposal_boxes:
            bx1, by1, bx2, by2 = box
            ix1 = max(x1, bx1)
            iy1 = max(y1, by1)
            ix2 = min(x2, bx2)
            iy2 = min(y2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            score += inter / max(1.0, tile_area)

        scores.append((tile, score))

    return scores


def select_topk_tiles(
    scored_tiles: List[Tuple[Tile, float]],
    topk_ratio: float,
) -> List[Tile]:
    if not scored_tiles:
        return []

    k = max(1, int(round(len(scored_tiles) * topk_ratio)))
    scored_tiles = sorted(scored_tiles, key=lambda x: x[1], reverse=True)
    return [tile for tile, _ in scored_tiles[:k]]