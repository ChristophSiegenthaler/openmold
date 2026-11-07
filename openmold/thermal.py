"""Transient cooling with CalculiX (DC3D4) and evaluation of cooling times and sink estimate.

CalculiX units: mm, t, s, mJ. So k stays numerically the same as W/m.K, density is t/mm3 (kg/m3 * 1e-12),
specific heat is mJ/t.K (J/kg.K * 1e6) and film coefficients are mW/mm2.K (W/m2.K * 1e-3).
"""
import numpy as np, meshio, scipy.sparse as sp
from scipy.spatial import cKDTree
from .common import nodal_volumes, orient_tets, unit, vertex_normals
from .ccx import read_frd


def _table(value, scale=1.0):
    if isinstance(value, (int, float)):
        return f"{value * scale:g}\n"
    return "".join(f"{v * scale:g},{t:g}\n" for t, v in value)


def zone_mask(cen, zones):
    """Film coefficient override per boundary face from cylinder zones. Returns an array of W/m2.K, NaN where none applies."""
    h = np.full(len(cen), np.nan)
    for zone in zones:
        for cyl in zone["cylinders"]:
            base, axis = np.asarray(cyl["base"], float), unit(cyl["axis"])
            rel = cen - base
            along = rel @ axis
            radial = np.linalg.norm(rel - along[:, None] * axis, axis=1)
            h[(radial < cyl["radius"]) & (along > cyl.get("start", 0.0)) & (along < cyl["length"])] = zone["htc"]
    return h


def write_deck(mesh_file, out_dir, cfg, job="th"):
    m = meshio.read(mesh_file)
    P = m.points
    T = orient_tets(P, m.cells_dict["tetra"])
    faces = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
    keys = np.vstack([np.sort(T[:, fd], axis=1) for fd in faces])
    eid = np.tile(np.arange(len(T)), 4)
    fid = np.repeat(np.arange(1, 5), len(T))
    _, idx, cnt = np.unique(keys, axis=0, return_index=True, return_counts=True)
    b = idx[cnt == 1]
    be, bf = eid[b], fid[b]
    cen = P[keys[b]].mean(1)
    pr, mat = cfg["process"], cfg["material"]
    # temperature-dependent properties make the problem nonlinear: fixed increments can diverge, so let CalculiX cut them
    adaptive = any(not isinstance(mat["thermal"][key], (int, float)) for key in ("conductivity", "specific_heat"))
    h = zone_mask(cen, cfg["cooling"]["htc_zones"])
    zoned = ~np.isnan(h)
    h[~zoned] = pr["htc"]
    np.savez(f"{out_dir}/th.npz", P=P, T=T, F=keys[b], zoned=zoned)
    with open(f"{out_dir}/{job}.inp", "w") as f:
        f.write("*NODE\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(P) + 1), P]), fmt="%d,%.6f,%.6f,%.6f")
        f.write("*ELEMENT,TYPE=DC3D4,ELSET=EALL\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(T) + 1), T + 1]), fmt="%d", delimiter=",")
        f.write(f"*NSET,NSET=NALL,GENERATE\n1,{len(P)},1\n")
        f.write(f"*MATERIAL,NAME=MAT\n*DENSITY\n{mat['density'] * 1e-12:g}\n")
        f.write("*CONDUCTIVITY\n" + _table(mat["thermal"]["conductivity"]))
        f.write("*SPECIFIC HEAT\n" + _table(mat["thermal"]["specific_heat"], 1e6))
        f.write("*SOLID SECTION,ELSET=EALL,MATERIAL=MAT\n")
        f.write(f"*INITIAL CONDITIONS,TYPE=TEMPERATURE\nNALL,{pr['melt_temp']}\n")
        film = "".join(f"{e + 1},F{fc},{pr['mold_temp']},{hh * 1e-3:g}\n" for e, fc, hh in zip(be, bf, h))
        for k, (dt, dur) in enumerate(cfg["cooling"]["steps"]):
            if adaptive:
                f.write(f"*STEP,INC=10000\n*HEAT TRANSFER\n{dt},{dur},{dt * 1e-4:g},{dt}\n")
            else:
                f.write(f"*STEP,INC=10000\n*HEAT TRANSFER,DIRECT\n{dt},{dur}\n")
            if k == 0:
                f.write("*FILM\n" + film)
            f.write("*NODE FILE,FREQUENCY=1\nNT\n*END STEP\n")
    return dict(thermal_nodes=len(P), thermal_elements=len(T), zoned_faces=int(zoned.sum()))


def crossing_times(NT, times, threshold):
    out = np.full(NT.shape[1], np.inf)
    for i in range(1, len(times)):
        m = (NT[i] < threshold) & np.isinf(out)
        a, b = NT[i - 1, m], NT[i, m]
        out[m] = times[i - 1] + (a - threshold) / np.maximum(a - b, 1e-9) * (times[i] - times[i - 1])
    return out


def field_at(NT, times, t):
    k = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    w = (t - times[k - 1]) / (times[k] - times[k - 1])
    return NT[k - 1] * (1 - w) + NT[k] * w


def _volume_quantile(tx, vn, q):
    o = np.argsort(tx)
    c = np.cumsum(vn[o]) / vn.sum()
    return float(tx[o][min(np.searchsorted(c, q), len(o) - 1)])


def _smooth_on_surface(values, F, n_nodes, iterations=8):
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.vstack([E, E[:, ::-1]])
    A = sp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n_nodes, n_nodes))
    A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel()
    s = values.copy()
    on = deg > 0
    for _ in range(iterations):
        s[on] = 0.5 * s[on] + 0.5 * (A @ s)[on] / deg[on]
    return s


def wall_reference(P, F, tnf, geo, probe):
    """Node representing the centre of the nominal wall. Its freeze time is the assumed end of packing."""
    from .common import in_box
    if probe:
        cand = np.where(in_box(P, probe))[0]
        if len(cand) == 0:
            raise ValueError("cooling.wall_probe contains no mesh nodes")
        return cand[np.argmax(tnf[cand])], "wall_probe box"
    nominal = np.median(geo["ts"][geo["sv"]])
    _, j = cKDTree(geo["P"]).query(P)
    depth, _ = cKDTree(P[np.unique(F)]).query(P)
    cand = np.where((np.abs(geo["th"][j] - nominal) < 0.1 * nominal) & (depth > 0.3 * nominal))[0]
    if len(cand) == 0:
        raise ValueError(f"the thermal mesh has no nodes inside the {nominal:.1f} mm nominal wall. "
                         "Make mesh.thermal smaller (about a third of the wall) or set cooling.wall_probe.")
    target = np.percentile(tnf[cand], 90)
    return cand[np.argmin(np.abs(tnf[cand] - target))], "automatic (90th percentile of nodes inside the nominal wall)"


def postprocess(out_dir, cfg, geo, job="th"):
    d = np.load(f"{out_dir}/th.npz")
    P, T, F = d["P"], d["T"], d["F"]
    mat, pr = cfg["material"], cfg["process"]
    TE, TNF = mat["thermal"]["eject_temp"], mat["thermal"]["no_flow_temp"]
    R = read_frd(f"{out_dir}/{job}.frd")
    if not R:
        raise RuntimeError(f"no temperature results in {out_dir}/{job}.frd")
    times = np.array([0.0] + [r[0] for r in R])
    NT = np.zeros((len(times), len(P)))
    NT[0] = pr["melt_temp"]
    for k, (_, ids, v) in enumerate(R):
        NT[k + 1, ids - 1] = v[:, 0]
    np.save(f"{out_dir}/NT.npy", NT)
    np.save(f"{out_dir}/times.npy", times)
    vn = nodal_volumes(P, T)
    te, tnf = crossing_times(NT, times, TE), crossing_times(NT, times, TNF)
    res = dict(t_eject_95=_volume_quantile(te, vn, 0.95), t_eject_100=float(te.max()),
               t_frozen_100=float(tnf.max()), t_frozen_95=_volume_quantile(tnf, vn, 0.95))
    ref, how = wall_reference(P, F, tnf, geo, cfg["cooling"]["wall_probe"])
    res.update(t_wall_frozen=float(tnf[ref]), t_wall_eject=float(te[ref]), wall_reference=how,
               wall_reference_xyz=P[ref].round(2).tolist())
    for tt in cfg["cooling"]["report_times"]:
        k = int(np.argmin(np.abs(times - tt)))
        res[f"Tmax_{tt:g}s"] = float(NT[k].max())
        res[f"molten_vol_{tt:g}s_mm3"] = float(vn[NT[k] > TNF].sum())
    res["hotspot_xyz"] = P[int(np.argmax(tnf))].round(1).tolist()

    # sink estimate: molten core under each surface point at the end of packing
    tp = res["t_wall_frozen"]
    Tp = field_at(NT, times, tp)
    sv = np.unique(F)
    N, _ = vertex_normals(P, F)
    tree = cKDTree(P)
    _, j = cKDTree(geo["P"][geo["sv"]]).query(P[sv])
    thick = geo["ts"][geo["sv"]][j]
    s = np.linspace(0.02, 1, 40)[None, :, None] * thick[:, None, None]
    pts = P[sv][:, None, :] - N[sv][:, None, :] * s
    dd, jj = tree.query(pts.reshape(-1, 3), k=4)
    wgt = 1 / np.maximum(dd, 1e-3)
    Tq = (Tp[jj] * wgt).sum(1) / wgt.sum(1)
    molten = (Tq.reshape(len(sv), 40) > TNF).sum(1) / 40 * thick
    sink = np.zeros(len(P))
    sink[sv] = mat["shrinkage"]["sink_volumetric"] * molten / 2
    sink_s = _smooth_on_surface(sink, F, len(P))
    res.update(t_pack_end_assumed=tp, Tmax_at_pack_end=float(Tp.max()), molten_vol_at_pack_end_mm3=float(vn[Tp > TNF].sum()),
               sink_max_mm=float(sink.max()), sink_max_xyz=P[int(np.argmax(sink))].round(1).tolist(),
               sink_smoothed_max_mm=float(sink_s.max()), sink_smoothed_xyz=P[int(np.argmax(sink_s))].round(1).tolist())
    np.savez(f"{out_dir}/thres.npz", te=te, tnf=tnf, sink=sink, sink_s=sink_s, Tp=Tp)
    return res
