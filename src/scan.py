"""A virtual RGB-D camera for ModelNet40 point clouds.

ModelNet40 point clouds are sampled uniformly from the whole CAD surface. A robot never sees that: a depth camera
sees one side of the object, with depth noise that grows with distance, holes, "flying pixels" at depth edges and
leftovers of the table the object stands on. This module turns a full 2048-point cloud into such a scan.

Scale: every object is normalised to the unit sphere and treated as 50 cm across, so 1 unit = 0.25 m.

    view      hidden point removal (Katz et al., 2007) from a camera at elevation `el`, azimuth `az`
    noise     stereo depth noise along the viewing ray, sigma_z = 0.01 * z^2 m (1 cm at 1 m, as for a ZED-class
              stereo camera); severity = camera distance z in metres
    flying    a fraction of the silhouette points (surface nearly parallel to the ray) pushed back along the ray
              by U(0, 50 cm), like mixed foreground/background pixels at depth discontinuities
    holes     a fraction of the visible points removed in 3 compact patches (specular / textureless regions)
    plane     a fraction of the final points replaced by samples of the supporting table plane under the object
              (imperfect plane removal / segmentation)

Every scan is re-centred and re-scaled to the unit sphere, as a deployed pipeline would do with a segmented cloud.
"""
import numpy as np
from scipy.spatial import ConvexHull

UNIT_M = 0.25           # metres per unit (object diameter 2 units = 50 cm)
UP = 1                  # ModelNet40 HDF5 clouds are y-up

SEVERITIES = {          # severity 1, 2, 3
    "noise": [1.0, 1.5, 2.0],          # camera distance (m) -> sigma 1.0 / 2.25 / 4.0 cm
    "flying": [0.05, 0.10, 0.20],
    "holes": [0.10, 0.25, 0.40],
    "plane": [0.10, 0.25, 0.40],
}
FACTORS = list(SEVERITIES)


def normalise(p):
    p = p - p.mean(0)
    return p / (np.linalg.norm(p, axis=1).max() + 1e-9)


def camera_dir(az, el):
    """Unit vector from the object centre to the camera (y-up)."""
    el, az = np.deg2rad(el), np.deg2rad(az)
    return np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], np.float64)


def hidden_point_removal(p, cam, gamma=2.6):
    """Indices of the points visible from `cam` (spherical flipping + convex hull, Katz et al. 2007)."""
    q = p - cam
    n = np.linalg.norm(q, axis=1, keepdims=True)
    R = n.max() * (10 ** gamma)
    flipped = q + 2 * (R - n) * q / n
    hull = ConvexHull(np.vstack([flipped, np.zeros((1, 3))]))
    v = hull.vertices
    return v[v < len(p)]


def scan(points, normals, rng, az=None, el=30.0, dist_m=1.0, noise=None, flying=0.0, holes=0.0, plane=0.0,
         view=True, n_out=1024):
    """Simulate one depth-camera scan of a unit-sphere object. Returns (n_out, 3) float32, unit-sphere normalised."""
    p = points.astype(np.float64)
    nrm = normals.astype(np.float64)
    az = rng.uniform(0, 360) if az is None else az
    d = camera_dir(az, el)
    z_m = noise if noise is not None else dist_m
    cam = d * (z_m / UNIT_M)
    if view:
        vis = hidden_point_removal(p, cam)
        p, nrm = p[vis], nrm[vis]
    if holes > 0 and len(p) > 50:
        keep = np.ones(len(p), bool)
        n_remove = int(holes * len(p))
        centres = p[rng.choice(len(p), 3, replace=False)]
        per = [n_remove // 3 + (i < n_remove % 3) for i in range(3)]
        for c, k in zip(centres, per):
            dd = np.linalg.norm(p - c, axis=1)
            dd[~keep] = np.inf
            keep[np.argsort(dd)[:k]] = False
        p, nrm = p[keep], nrm[keep]
    ray = p - cam
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    if flying > 0:
        grazing = np.abs((nrm * ray).sum(1))          # small = surface seen edge-on = depth discontinuity
        k = int(flying * len(p))
        idx = np.argsort(grazing + 1e-3 * rng.random(len(p)))[:k]
        p[idx] += ray[idx] * rng.uniform(0, 0.5 / UNIT_M, size=(k, 1))
    if noise is not None:
        z = np.linalg.norm(p - cam, axis=1) * UNIT_M                       # metres
        sigma = 0.01 * z ** 2 / UNIT_M                                     # units
        p += ray * (rng.standard_normal(len(p)) * sigma)[:, None]
    n_plane = int(round(plane * n_out))
    n_obj = n_out - n_plane
    p = p[rng.choice(len(p), n_obj, replace=len(p) < n_obj)]
    if n_plane:
        lo = points[:, UP].min()
        ext = np.abs(points).max(0) * 1.5
        pl = rng.uniform(-1, 1, size=(n_plane, 3)) * ext
        pl[:, UP] = lo
        if noise is not None:
            z = np.linalg.norm(pl - cam, axis=1) * UNIT_M
            r = (pl - cam) / np.linalg.norm(pl - cam, axis=1, keepdims=True)
            pl += r * (rng.standard_normal(n_plane) * 0.01 * z ** 2 / UNIT_M)[:, None]
        p = np.vstack([p, pl])
    return normalise(p).astype(np.float32)


def condition(name):
    """Named evaluation conditions -> kwargs for scan(). 'cad' = the standard full-surface benchmark input."""
    if name == "cad":
        return dict(view=False)
    if name == "view":
        return dict()
    if name.startswith("el"):                          # el0, el30, el60, el90
        return dict(el=float(name[2:]))
    if name == "realistic":
        return dict(noise=1.0, flying=0.05, holes=0.10, plane=0.10)
    f, s = name.rsplit("_s", 1)                        # e.g. noise_s2
    return {f: SEVERITIES[f][int(s) - 1]}


ALL_CONDITIONS = (["cad", "view"] + [f"{f}_s{s}" for f in FACTORS for s in (1, 2, 3)]
                  + ["realistic", "el0", "el30", "el60", "el90"])


def random_scan(points, normals, rng, factors=("noise", "flying", "holes"), p_view=0.8):
    """Training-time augmentation: a random simulated scan. 'plane' is deliberately NOT simulated, so that it can
    be used at test time as an artefact the simulator did not model."""
    if rng.random() > p_view:
        return normalise(points[rng.choice(len(points), 1024, replace=False)]).astype(np.float32)
    kw = dict(el=rng.uniform(-10, 85))
    if "noise" in factors and rng.random() < 0.5:
        kw["noise"] = rng.uniform(0.5, 2.0)
    if "flying" in factors and rng.random() < 0.5:
        kw["flying"] = rng.uniform(0, 0.2)
    if "holes" in factors and rng.random() < 0.5:
        kw["holes"] = rng.uniform(0, 0.4)
    return scan(points, normals, rng, **kw)
