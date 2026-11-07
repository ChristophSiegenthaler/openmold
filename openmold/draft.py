"""Draft-angle check of the CAD surface against the demoulding direction."""
import numpy as np
from .common import unit


def analyse(P, F, axis, limit_deg, bins=(0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 10.0)):
    """Angle of each triangle to the demould axis. 0 deg means no draft; 90 deg means facing the demould direction.

    The check cannot tell undercuts apart from drafted faces because it does not know the parting line.
    """
    a = unit(axis)
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    area = np.linalg.norm(n, axis=1) / 2
    nn = n / np.maximum(2 * area, 1e-12)[:, None]
    draft = np.degrees(np.arcsin(np.clip(np.abs(nn @ a), 0, 1)))
    side = draft < 10.0   # faces that actually slide along the tool during demoulding
    hist, lo = {}, 0.0
    for hi in bins:
        hist[f"{lo:g}-{hi:g}"] = float(area[(draft >= lo) & (draft < hi)].sum())
        lo = hi
    info = dict(limit_deg=float(limit_deg), side_area_mm2=float(area[side].sum()),
                below_limit_area_mm2=float(area[side & (draft < limit_deg)].sum()),
                below_limit_pct=float(100 * area[side & (draft < limit_deg)].sum() / max(area[side].sum(), 1e-9)),
                min_draft_deg=float(draft[side].min()) if side.any() else None, histogram_mm2=hist)
    return draft, info
