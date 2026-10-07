import numpy as np

from sarship.detect import ca_background, detect, land_mask, os_background, ratio_map, ring_footprint
from sarship.evaluate import GRID, Item, build, mean_ship_ratio, roc, score
from sarship.sim import LOOKS, SIZE, Scene, Ship, make_scene, ship_mask


def test_scenes_are_deterministic():
    a, b = make_scene("coastal", 5, n_ships=4), make_scene("coastal", 5, n_ships=4)
    assert np.array_equal(a.img, b.img) and a.ships == b.ships


def test_speckle_is_gamma_with_the_right_number_of_looks():
    rng = np.random.default_rng(0)
    x = rng.gamma(LOOKS, 1 / LOOKS, 200_000)
    assert abs(x.mean() - 1) < 0.01 and abs(x.var() - 1 / LOOKS) < 0.01
    sc = make_scene("homogeneous", 0, n_ships=0)
    assert sc.img.shape == (SIZE, SIZE) and sc.ships == [] and not sc.land.any()


def test_ring_footprint_excludes_the_guard_window():
    fp = ring_footprint(7, 15)
    assert fp.shape == (31, 31) and not fp[8:23, 8:23].any() and fp.sum() == 31**2 - 15**2


def test_cfar_ratio_is_about_one_in_ship_free_open_sea():
    img = make_scene("homogeneous", 1, n_ships=0).img
    assert abs(ratio_map(img, "CA").mean() - 1.0) < 0.05
    assert 0.7 < ratio_map(img, "OS").mean() < 0.95  # a high percentile sits above the mean


def test_bright_isolated_ship_is_detected_and_open_sea_is_quiet():
    sc = make_scene("homogeneous", 2, n_ships=3, scr_range=(15, 15))
    item = Item(sc)
    for m in ("CA", "OS"):
        hit, fa = score(item, m, 3.5)
        assert all(hit) and fa <= 3


def test_land_mask_covers_land_and_rarely_hides_a_ship():
    sc = make_scene("coastal", 3, n_ships=5, scr_range=(20, 25))
    assert (land_mask(sc.img) & sc.land).sum() / sc.land.sum() > 0.9
    masked = total = 0
    for seed in range(100, 112):  # known limitation: a ship hugging the coast can fall inside the mask
        scene = make_scene("coastal", seed, n_ships=6)
        mask = land_mask(scene.img)
        masked += sum(bool(mask[int(s.cy), int(s.cx)]) for s in scene.ships)
        total += len(scene.ships)
    assert masked / total < 0.03


def test_land_mask_removes_coastal_false_alarms():
    items = build("coastal", range(4), n_ships=4)
    raw = sum(score(it, "CA", 3.09, use_land_mask=False)[1] for it in items)
    masked = sum(score(it, "CA", 3.09, use_land_mask=True)[1] for it in items)
    assert masked < 0.5 * raw


def test_neighbours_inside_the_training_ring_mask_a_ship():
    iso = build("homogeneous", range(10), n_ships=6, scr_range=(1.7, 1.7))
    conv = build("convoy", range(10))
    for m in ("CA", "OS"):
        assert mean_ship_ratio(conv, m) < mean_ship_ratio(iso, m)


def test_false_alarms_fall_as_the_threshold_rises():
    rows = roc(build("homogeneous", range(3)), "CA")
    fa = [r["fa_per_scene"] for r in rows]
    peak = int(np.argmax(fa))  # below the peak, clutter merges into a few huge blobs
    assert len(rows) == len(GRID) and fa[-1] == 0
    assert all(a >= b for a, b in zip(fa[peak:], fa[peak + 1 :]))


def test_matching_counts_a_detection_off_any_ship_as_a_false_alarm():
    ship = Ship(60, 60, 12, 4, 0.0, 5.0)
    img = np.ones((SIZE, SIZE), np.float32)
    item = Item(Scene(img, [ship], np.zeros_like(img, bool)))
    item.ratio = {m: np.zeros((SIZE, SIZE)) for m in ("CA", "OS")}
    item.ratio["CA"][58:62, 58:62] = 10  # on the ship
    item.ratio["CA"][150:154, 150:154] = 10  # far away
    assert score(item, "CA", 5.0) == ([True], 1)
