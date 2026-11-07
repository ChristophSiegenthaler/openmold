"""Local wall thickness by the shrinking-ball (inscribed sphere) method."""
import numpy as np, trimesh
from scipy.spatial import cKDTree
from .common import boundary, tet_volumes, vertex_normals


def compute(P, T, samples=600000, sample_radius=0.08, initial_radius=None, minimum=0.3):
    F = boundary(T)
    N, _ = vertex_normals(P, F)
    sv = np.unique(F)
    if initial_radius is None:
        initial_radius = np.linalg.norm(P.max(0) - P.min(0)) / 3
    tm = trimesh.Trimesh(P, F, process=False)
    S, _ = trimesh.sample.sample_surface_even(tm, samples, radius=sample_radius, seed=1)
    S = np.vstack([S, P[sv]])
    tree = cKDTree(S)
    p, nin = P[sv], -N[sv]
    r = np.full(len(sv), float(initial_radius))
    for _ in range(60):
        d, j = tree.query(p + r[:, None] * nin, k=1)
        pq = S[j] - p
        dist2 = (pq ** 2).sum(1)
        den = 2 * (pq * nin).sum(1)
        update = (d < r - 1e-3) & (dist2 > 1e-4) & (den > 1e-9)
        new = np.minimum(np.where(update, dist2 / np.maximum(den, 1e-12), r), r)
        if np.allclose(new, r):
            break
        r = new
    ts = np.zeros(len(P))
    ts[sv] = 2 * r
    # thickness at volume nodes: the largest inscribed ball that contains the node
    centres = p + r[:, None] * nin
    th = np.zeros(len(P))
    vt = cKDTree(P)
    for k in np.argsort(r):
        idx = vt.query_ball_point(centres[k], r[k] * 1.02)
        th[idx] = np.maximum(th[idx], 2 * r[k])
    dw, _ = tree.query(P)
    th[th == 0] = 2 * dw[th == 0]
    th = np.maximum(th, minimum)
    info = dict(volume_mm3=float(tet_volumes(P, T).sum()), nominal_wall_mm=float(np.median(ts[sv])),
                thickness_p95_mm=float(np.percentile(ts[sv], 95)), thickness_max_surface_mm=float(ts[sv].max()),
                thickness_max_mm=float(th.max()))
    return dict(F=F, ts=ts, th=th, sv=sv, dw=dw), info
