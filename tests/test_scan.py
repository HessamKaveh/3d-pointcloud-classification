import numpy as np
import pytest

import scan


@pytest.fixture(scope="module")
def sphere():
    """2048 points on the unit sphere (a stand-in for a ModelNet40 object); normals point outwards."""
    p = np.random.default_rng(0).standard_normal((2048, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p.astype(np.float32), p.astype(np.float32).copy()


def _sphere_residual(p):
    """Relative RMS deviation from the least-squares sphere through p."""
    A = np.c_[2 * p, np.ones(len(p))]
    sol = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    c, r = sol[:3], np.sqrt(sol[3] + sol[:3] @ sol[:3])
    return np.sqrt(np.mean((np.linalg.norm(p - c, axis=1) - r) ** 2)) / r


@pytest.mark.parametrize("cond", scan.ALL_CONDITIONS)
def test_every_condition_gives_normalised_cloud(sphere, cond):
    x = scan.scan(*sphere, np.random.default_rng(1), az=40.0, **scan.condition(cond))
    assert x.shape == (1024, 3) and x.dtype == np.float32
    assert np.all(np.isfinite(x))
    assert np.linalg.norm(x, axis=1).max() == pytest.approx(1.0, abs=1e-4)
    np.testing.assert_allclose(x.mean(0), 0, atol=0.02)   # centred before resampling to 1024 points


def test_scan_is_deterministic_given_seed(sphere):
    kw = scan.condition("realistic")
    a = scan.scan(*sphere, np.random.default_rng(7), **kw)
    b = scan.scan(*sphere, np.random.default_rng(7), **kw)
    np.testing.assert_array_equal(a, b)


def test_hidden_point_removal_keeps_the_near_side(sphere):
    p, _ = sphere
    d = scan.camera_dir(30.0, 20.0)
    vis = scan.hidden_point_removal(p.astype(np.float64), d * 4.0)
    assert 0.2 * len(p) < len(vis) < 0.6 * len(p)
    assert np.mean(p[vis] @ d > 0) > 0.95                 # HPR is approximate near the silhouette
    assert np.mean(p[vis] @ d) > 0.4
    assert np.linalg.norm(d) == pytest.approx(1.0)


def test_depth_noise_grows_with_severity(sphere):
    res = [_sphere_residual(scan.scan(*sphere, np.random.default_rng(3), az=0.0, **scan.condition(f"noise_s{s}")))
           for s in (1, 2, 3)]
    clean = _sphere_residual(scan.scan(*sphere, np.random.default_rng(3), az=0.0))
    assert clean < 1e-4
    assert clean < res[0] < res[1] < res[2]


def test_random_scan_augmentation(sphere):
    rng = np.random.default_rng(5)
    for _ in range(5):
        x = scan.random_scan(*sphere, rng)
        assert x.shape == (1024, 3) and x.dtype == np.float32
