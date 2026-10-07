"""Synthetic multi-look SAR intensity scenes: speckled sea, optional coastline, ships.

Intensity of a pixel with mean level mu is Gamma(L, mu/L): the standard multi-look speckle
model (L = number of looks). The sea mean level varies slowly (wind streaks, range
dependence); land is brighter and more textured; ships are small oriented rectangles with a
few extra-bright point scatterers. SCR is the ship-to-local-clutter mean ratio (linear).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

LOOKS = 4
SIZE = 192


@dataclass(frozen=True)
class Ship:
    cx: float
    cy: float
    length: float
    width: float
    angle: float
    scr: float

    @property
    def match_radius(self) -> float:
        return self.length / 2 + 3


@dataclass
class Scene:
    img: np.ndarray
    ships: list[Ship]
    land: np.ndarray  # ground-truth land mask


def _smooth(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    n = gaussian_filter(rng.normal(size=shape), sigma)
    return n / n.std()


def ship_mask(shape: tuple[int, int], s: Ship) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    dx, dy = xx - s.cx, yy - s.cy
    u = dx * np.cos(s.angle) + dy * np.sin(s.angle)
    v = -dx * np.sin(s.angle) + dy * np.cos(s.angle)
    return (np.abs(u) <= s.length / 2) & (np.abs(v) <= s.width / 2)


def make_scene(kind: str = "homogeneous", seed: int = 0, n_ships: int = 8,
               scr_range: tuple[float, float] = (1.1, 8.0), pair_scr: float = 1.7) -> Scene:
    """kind: 'homogeneous' open sea | 'coastal' coastline + strongly varying sea | 'pairs' close ship pairs | 'convoy' rows of four."""
    rng = np.random.default_rng(seed)
    n = SIZE
    tex = 0.5 if kind == "coastal" else 0.15
    mu = np.exp(tex * _smooth(rng, (n, n), 14) - tex**2 / 2)
    land = np.zeros((n, n), bool)
    coast = np.full(n, n + 100.0)
    if kind == "coastal":
        mu *= np.linspace(0.6, 1.6, n)[None, :]  # range-dependent clutter level
        wiggle = gaussian_filter1d(rng.normal(size=n), 8)
        coast = 0.62 * n + 14 * wiggle / wiggle.std()
        land = np.mgrid[:n, :n][0] > coast[None, :]
        mu[land] = 6.0 * np.exp(0.6 * _smooth(rng, (n, n), 5) - 0.18)[land]

    ships: list[Ship] = []

    def free(x, y, sep):
        return all(np.hypot(x - s.cx, y - s.cy) > sep for s in ships)

    def sample_ship(scr, cx=None, cy=None, angle=None, length=None):
        for _ in range(200):
            x = cx if cx is not None else rng.uniform(26, n - 26)
            y = cy if cy is not None else rng.uniform(26, n - 26)
            if kind == "coastal" and y > coast[int(np.clip(x, 0, n - 1))] - 26:
                if cx is not None:
                    return None
                continue
            return Ship(x, y, length or rng.uniform(10, 18), rng.uniform(3, 5), angle if angle is not None else rng.uniform(0, np.pi), scr)
        return None

    if kind == "convoy":  # two convoys of four ships side by side, 10 px apart: ~15-25% of a training ring
        anchors, attempts = 0, 0
        while anchors < 2:
            attempts += 1
            if attempts > 2000:
                raise RuntimeError("could not place convoys: scene too crowded")
            a = sample_ship(pair_scr, length=13.0)
            if a is None or not free(a.cx, a.cy, 70):
                continue
            row = [Ship(a.cx - off * np.sin(a.angle), a.cy + off * np.cos(a.angle), 13.0, a.width, a.angle, pair_scr) for off in (-15, -5, 5, 15)]
            if all(26 < b.cx < n - 26 and 26 < b.cy < n - 26 for b in row):
                ships += row
                anchors += 1
    elif kind == "pairs":
        anchors, attempts = 0, 0
        while anchors < 3:
            attempts += 1
            if attempts > 2000:
                raise RuntimeError("could not place ship pairs: scene too crowded")
            a = sample_ship(pair_scr)
            if a is None or not free(a.cx, a.cy, 55):
                continue
            off = rng.uniform(9, 12)  # lateral offset: neighbour sits inside a CFAR training ring
            b = Ship(a.cx - off * np.sin(a.angle), a.cy + off * np.cos(a.angle), a.length, a.width, a.angle, pair_scr)
            if 26 < b.cx < n - 26 and 26 < b.cy < n - 26:
                ships += [a, b]
                anchors += 1
    else:
        attempts = 0
        while len(ships) < n_ships:
            attempts += 1
            if attempts > 2000:
                raise RuntimeError("could not place ships: scene too crowded")
            scr = float(np.exp(rng.uniform(np.log(scr_range[0]), np.log(scr_range[1]))))
            s = sample_ship(scr)
            if s is not None and free(s.cx, s.cy, 32):
                ships.append(s)

    for s in ships:
        m = ship_mask((n, n), s)
        local = mu[int(s.cy), int(s.cx)]
        mu[m] = local * s.scr
        idx = np.argwhere(m)
        for y, x in idx[rng.choice(len(idx), size=3, replace=False)]:  # bright point scatterers
            mu[y, x] *= 2.5
    img = rng.gamma(LOOKS, mu / LOOKS)
    return Scene(img.astype(np.float32), ships, land)
