import numpy as np
import pytest
import torch

import evaluate as ev
from model import build, knn
from outlier_filter import sor


@pytest.mark.parametrize("name", ["pointnet", "dgcnn"])
def test_model_logits_shape(name):
    torch.manual_seed(0)
    m = build(name, num_classes=40).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 64, 3)).shape == (2, 40)


def test_pointnet_is_permutation_invariant_and_checks_shape():
    torch.manual_seed(0)
    m = build("pointnet").eval()
    x = torch.randn(2, 128, 3)
    with torch.no_grad():
        torch.testing.assert_close(m(x), m(x[:, torch.randperm(128)]), rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError):
        m(torch.randn(2, 3, 128))


def test_knn_includes_self():
    x = torch.randn(2, 3, 50)
    idx = knn(x, 4)
    assert idx.shape == (2, 50, 4)
    assert torch.all(idx[:, :, 0] == torch.arange(50))


def test_conformal_coverage_on_calibrated_probs():
    rng = np.random.default_rng(0)
    p = ev.softmax(2.0 * rng.standard_normal((6000, 40)))
    np.testing.assert_allclose(p.sum(1), 1.0)
    y = np.array([rng.choice(40, p=row) for row in p])          # labels drawn from the model: exchangeable
    covs, sizes = zip(*(ev.conformal(p[:3000], y[:3000], p[3000:], y[3000:], alpha=a) for a in (0.05, 0.1, 0.2)))
    for cov, a in zip(covs, (0.05, 0.1, 0.2)):
        assert cov >= 1 - a - 0.015
    assert sizes[0] > sizes[1] > sizes[2]                          # stricter target -> larger sets


def test_ece():
    y = np.arange(1000) % 40
    assert ev.ece(np.eye(40)[y], y) == pytest.approx(0.0)
    p = np.full((1000, 40), 0.25 / 39); p[np.arange(1000), (y + 1) % 40] = 0.75
    assert ev.ece(p, y) == pytest.approx(0.75)                   # 75 % confident, always wrong


def test_statistical_outlier_removal_drops_far_points():
    rng = np.random.default_rng(0)
    p = rng.standard_normal((1000, 3)); p /= np.linalg.norm(p, axis=1, keepdims=True)
    out = np.vstack([p, 5.0 * rng.standard_normal((24, 3)) + 10.0])
    q, frac = sor(out, rng=rng)
    assert q.shape == (1024, 3)
    assert frac >= 24 / 1024
    assert np.linalg.norm(q, axis=1).max() == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(q, axis=1).min() > 0.8                  # only the sphere survived
