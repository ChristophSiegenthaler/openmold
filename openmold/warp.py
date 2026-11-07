"""Thermo-elastic warpage after ejection with CalculiX (C3D10), and its split into shrinkage and distortion.

Only thermal contraction after ejection is modelled: a free part cools from its simulated temperature field to
room temperature. Shrinkage while still constrained in the mould and packing effects are not included.
"""
import numpy as np, meshio
from scipy.spatial import cKDTree
from .ccx import read_frd
from .common import basis, boundary, in_box, similarity_fit, unit


def fmt_t(t):
    return f"{t:g}"


def default_supports(P):
    """Statically determinate 3-2-1 supports: centre node, farthest +x node, farthest +z node on the mesh."""
    c = (P.min(0) + P.max(0)) / 2
    a = int(np.argmin(np.linalg.norm(P - c, axis=1)))
    b = int(np.argmax(P[:, 0] - 0.1 * np.abs(P[:, 1] - c[1]) - 0.1 * np.abs(P[:, 2] - c[2])))
    d = int(np.argmax(P[:, 2] - 0.1 * np.abs(P[:, 1] - c[1]) - 0.1 * np.abs(P[:, 0] - c[0])))
    return a, b, d


def write_deck(mesh_file, thermal_dir, out_dir, cfg, t_eject):
    m = meshio.read(mesh_file)
    P, T = m.points, m.cells_dict["tetra10"]
    th = np.load(f"{thermal_dir}/th.npz")
    NT, times = np.load(f"{thermal_dir}/NT.npy"), np.load(f"{thermal_dir}/times.npy")
    k = int(np.clip(np.searchsorted(times, t_eject), 1, len(times) - 1))
    w = (t_eject - times[k - 1]) / (times[k] - times[k - 1])
    Tt = NT[k - 1] * (1 - w) + NT[k] * w
    dd, jj = cKDTree(th["P"]).query(P, k=4)
    wg = 1 / np.maximum(dd, 1e-3)
    Tn = (Tt[jj] * wg).sum(1) / wg.sum(1)
    sup = cfg["warp"]["supports"]
    if sup:
        a, b, c = [int(np.argmin(np.linalg.norm(P - np.array(x), axis=1))) for x in sup]
    else:
        a, b, c = default_supports(P)
    mech = cfg["material"]["mechanical"]
    job = f"me_{fmt_t(t_eject)}"
    with open(f"{out_dir}/{job}.inp", "w") as f:
        f.write("*NODE\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(P) + 1), P]), fmt="%d,%.6f,%.6f,%.6f")
        f.write("*ELEMENT,TYPE=C3D10,ELSET=EALL\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(T) + 1), T + 1]), fmt="%d", delimiter=",")
        f.write(f"*NSET,NSET=NALL,GENERATE\n1,{len(P)},1\n")
        f.write(f"*MATERIAL,NAME=MAT\n*ELASTIC\n{mech['youngs_modulus']:g},{mech['poisson']:g}\n*EXPANSION\n{mech['expansion']:g}\n")
        f.write("*SOLID SECTION,ELSET=EALL,MATERIAL=MAT\n*INITIAL CONDITIONS,TYPE=TEMPERATURE\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(P) + 1), Tn]), fmt="%d,%.3f")
        f.write(f"*BOUNDARY\n{a + 1},1,3\n{b + 1},2,3\n{c + 1},2,2\n")
        f.write(f"*STEP\n*STATIC\n*TEMPERATURE\nNALL,{cfg['process']['room_temp']}\n*NODE FILE\nU\n*END STEP\n")
    return job


def metrics(P, W, wcfg):
    out = {}

    def nearest(x):
        return int(np.argmin(np.linalg.norm(P - np.array(x, float), axis=1)))

    for item in wcfg["flatness"]:
        idx = np.where(in_box(P, item["box"]))[0]
        if len(idx) < 3:
            raise ValueError(f"warp.flatness '{item['name']}': the box contains fewer than 3 nodes")
        n = unit(item["normal"])
        e1, e2 = basis(n)
        X = np.column_stack([P[idx] @ e1, P[idx] @ e2, np.ones(len(idx))])
        y = W[idx] @ n
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        dev = y - X @ coef
        out[f"flatness_{item['name']}_mm"] = float(dev.max() - dev.min())
    for item in wcfg["bows"]:
        out[f"bow_{item['name']}_mm"] = float((W[nearest(item["mid"])] - W[nearest(item["ref"])]) @ unit(item["direction"]))
    for item in wcfg["lateral"]:
        n = unit(item["plane_normal"])
        vals = [np.linalg.norm(W[nearest(x)] - (W[nearest(x)] @ n) * n) for x in item["points"]]
        out[f"lateral_{item['name']}_mm"] = float(max(vals))
    return out


def postprocess(mesh_file, out_dir, cfg, t_eject):
    job = f"me_{fmt_t(t_eject)}"
    m = meshio.read(mesh_file)
    P = m.points
    T10 = m.cells_dict["tetra10"]
    R = read_frd(f"{out_dir}/{job}.frd", "DISP")
    if not R:
        raise RuntimeError(f"no displacement results in {out_dir}/{job}.frd")
    _, ids, v = R[-1]
    U = np.zeros_like(P)
    U[ids - 1] = v[:, :3]
    Q = P + U
    Rm, s, cp, cq = similarity_fit(P, Q)
    W = Q - (s * ((P - cp) @ Rm.T) + cq)
    res = dict(t_eject=float(t_eject), displacement_max_mm=float(np.linalg.norm(U, axis=1).max()),
               thermal_shrink_after_eject_pct=float((1 - s) * 100), warp_max_mm=float(np.linalg.norm(W, axis=1).max()))
    res.update(metrics(P, W, cfg["warp"]))
    np.savez(f"{out_dir}/warp_{fmt_t(t_eject)}.npz", P=P, U=U, W=W, F=boundary(T10[:, :4]))
    return res
