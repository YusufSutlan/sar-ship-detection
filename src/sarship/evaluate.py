"""Detector comparison.

Protocol (mirrors operational practice): choose each detector's threshold on *open-sea*
calibration scenes so that it raises about 0.5 false alarms per scene, then freeze it and see
what happens in scenarios the design did not assume: a coastline with strongly varying
clutter, and ships sailing close together.

Usage:
    python -m sarship.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .detect import detect, land_mask, ratio_map
from .sim import Scene, make_scene, ship_mask

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
METHODS = ("CA", "OS")
GRID = np.geomspace(1.5, 25, 40)
FA_BUDGET = 0.5  # false alarms per scene
SCR_BINS = [1.1, 1.4, 1.7, 2.0, 3.0, 8.0]


class Item:
    """A scene with its precomputed CFAR ratio maps."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.ratio = {m: ratio_map(scene.img, m) for m in METHODS}
        self.land = land_mask(scene.img)


def build(kind: str, seeds, **kw) -> list[Item]:
    return [Item(make_scene(kind, s, **kw)) for s in seeds]


def score(item: Item, method: str, threshold: float, use_land_mask: bool = True):
    """Return (per-ship detected flags, number of false alarms) for one scene."""
    dets = detect(item.ratio[method], threshold, exclude=item.land if use_land_mask else None)
    ships = item.scene.ships
    hit = [bool(len(dets)) and bool((np.hypot(dets[:, 0] - s.cx, dets[:, 1] - s.cy) <= s.match_radius).any()) for s in ships]
    on_ship = [any(np.hypot(x - s.cx, y - s.cy) <= s.match_radius for s in ships) for x, y in dets]
    return hit, int(len(dets) - sum(on_ship))


def summarize(items, method, threshold, use_land_mask=True) -> dict:
    hits, scrs, fa = [], [], 0
    for it in items:
        h, f = score(it, method, threshold, use_land_mask)
        hits += h
        scrs += [s.scr for s in it.scene.ships]
        fa += f
    return {"pd": float(np.mean(hits)), "fa_per_scene": fa / len(items), "hits": np.array(hits), "scr": np.array(scrs)}


def mean_ship_ratio(items, method) -> float:
    """Average CFAR ratio over ship pixels: how strongly the detector 'sees' the ships,
    independent of any threshold. Contaminated training cells push it down."""
    vals = []
    for it in items:
        for s in it.scene.ships:
            vals.append(it.ratio[method][ship_mask(it.scene.img.shape, s)].mean())
    return float(np.mean(vals))


def roc(items, method) -> list[dict]:
    return [{"threshold": float(t), **{k: v for k, v in summarize(items, method, t).items() if k in ("pd", "fa_per_scene")}} for t in GRID]


def calibrate(items, method) -> float:
    """Smallest threshold from which the false-alarm budget holds for every higher threshold too.
    (Counting objects is not monotone: at very low thresholds clutter merges into a few huge blobs.)"""
    rows = roc(items, method)
    best = rows[-1]["threshold"]
    for r in reversed(rows):
        if r["fa_per_scene"] > FA_BUDGET:
            break
        best = r["threshold"]
    return best


def plot_roc(roc_rows: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for m, rows in roc_rows.items():
        peak = int(np.argmax([r["fa_per_scene"] for r in rows]))
        rows = rows[peak:]  # left of the peak, clutter merges into huge blobs and counts stop meaning anything
        fa = np.array([r["fa_per_scene"] for r in rows]) + 1e-3
        ax.plot(fa, [r["pd"] for r in rows], "o-", ms=3, label=f"{m}-CFAR")
    ax.set_xscale("log")
    ax.axvline(FA_BUDGET, color="k", ls=":", lw=1)
    ax.set_xlabel("false alarms per scene (log)")
    ax.set_ylabel("probability of detection")
    ax.set_title("Open sea, ships with SCR 1.1-8 (0.4-9 dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_open_sea.png", dpi=130)
    plt.close(fig)


def plot_pd_by_scr(res: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    centers = [np.sqrt(a * b) for a, b in zip(SCR_BINS[:-1], SCR_BINS[1:])]
    for m, r in res.items():
        pds = [r["hits"][(r["scr"] >= a) & (r["scr"] < b)].mean() for a, b in zip(SCR_BINS[:-1], SCR_BINS[1:])]
        ax.plot(10 * np.log10(centers), pds, "o-", label=f"{m}-CFAR")
    ax.set_xlabel("ship-to-clutter ratio (dB)")
    ax.set_ylabel("probability of detection")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Detection vs. ship brightness (threshold fixed at 0.5 FA/scene)")
    fig.tight_layout()
    fig.savefig(FIGURES / "pd_vs_scr.png", dpi=130)
    plt.close(fig)


def plot_example(item: Item, thresholds: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    img = np.log1p(item.scene.img)
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("SAR intensity (log), coastal scene")
    for ax, m in zip(axes[1:], METHODS):
        ax.imshow(img, cmap="gray")
        dets = detect(item.ratio[m], thresholds[m], exclude=None)
        for s in item.scene.ships:
            ax.add_patch(plt.Circle((s.cx, s.cy), s.match_radius, fill=False, ec="lime", lw=1.2))
        ax.scatter(dets[:, 0], dets[:, 1], marker="x", c="red", s=40)
        ax.set_title(f"{m}-CFAR, no land mask: {len(dets)} detections (green = true ships)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "coastal_example.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    calib = build("homogeneous", range(1000, 1020))
    thr = {m: calibrate(calib, m) for m in METHODS}
    print("Thresholds from open-sea calibration (0.5 FA/scene):", {m: round(t, 2) for m, t in thr.items()})

    open_sea = build("homogeneous", range(30))
    roc_rows = {m: roc(open_sea, m) for m in METHODS}
    open_res = {m: summarize(open_sea, m, thr[m]) for m in METHODS}
    coastal = build("coastal", range(30), n_ships=6)
    pairs = build("pairs", range(30))
    convoys = build("convoy", range(30))
    isolated = build("homogeneous", range(2000, 2030), n_ships=6, scr_range=(1.7, 1.7))

    report = {"thresholds": thr, "open_sea": {}, "coastal": {}, "pairs": {}}
    print(f"\n{'':<9}{'open sea':>22}{'coastal, no mask':>22}{'coastal, land mask':>22}")
    print(f"{'':<9}{'Pd':>11}{'FA/scene':>11}{'Pd':>11}{'FA/scene':>11}{'Pd':>11}{'FA/scene':>11}")
    for m in METHODS:
        o = open_res[m]
        c0 = summarize(coastal, m, thr[m], use_land_mask=False)
        c1 = summarize(coastal, m, thr[m], use_land_mask=True)
        print(f"{m + '-CFAR':<9}{o['pd']:>11.2f}{o['fa_per_scene']:>11.2f}{c0['pd']:>11.2f}{c0['fa_per_scene']:>11.2f}{c1['pd']:>11.2f}{c1['fa_per_scene']:>11.2f}")
        report["open_sea"][m] = {k: o[k] for k in ("pd", "fa_per_scene")}
        report["coastal"][m] = {"no_mask": {k: c0[k] for k in ("pd", "fa_per_scene")}, "land_mask": {k: c1[k] for k in ("pd", "fa_per_scene")}}

    print("\nShips of SCR 1.7 (2.3 dB): isolated, in a close pair, and in a 4-ship convoy (neighbours inside the training ring)")
    for m in METHODS:
        rows = {}
        for name, items in (("isolated", isolated), ("pair", pairs), ("convoy", convoys)):
            r = summarize(items, m, thr[m])
            rows[name] = {"pd": r["pd"], "mean_ratio": mean_ship_ratio(items, m)}
        line = "   ".join(f"{k}: Pd {v['pd']:.2f}, ratio {v['mean_ratio']:.2f}" for k, v in rows.items())
        print(f"  {m}-CFAR  {line}")
        report["pairs"][m] = rows

    plot_roc(roc_rows)
    plot_pd_by_scr(open_res)
    plot_example(coastal[3], thr)
    (ROOT / "reports" / "results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
