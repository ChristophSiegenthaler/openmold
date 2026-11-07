"""Isothermal 3D Hele-Shaw fill solver on a tet mesh (control volume method).

Units: mm, s, MPa. Local conductance is h^2/(12*eta), with h the local inscribed-sphere thickness.
The flow rate is constant (volume / fill time). There is no frozen layer, shear heating or compressibility.
"""
import math
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla


def zero_shear_viscosity(flow, temp_c):
    """Zero-shear viscosity in Pa.s at temp_c, for the cross or cross_wlf model at zero pressure."""
    if flow["model"] == "cross":
        return float(flow["eta0"])
    dT = temp_c + 273.15 - flow["D2"]
    if flow["A2"] + dT <= 0:
        raise ValueError("temperature is below the Cross-WLF validity range")
    return float(flow["D1"] * math.exp(-flow["A1"] * dT / (flow["A2"] + dT)))


def gate_nodes(P, F, gates):
    surface = np.unique(F)
    ids = []
    for g in gates:
        d = np.linalg.norm(P[surface] - np.asarray(g["point"], float), axis=1)
        near = surface[d < g.get("radius", 1.0)]
        ids.append(near if len(near) >= 3 else surface[np.argsort(d)[:4]])
    return np.unique(np.concatenate(ids))


def solve(P, T, th, F, flow, melt_temp, fill_time, gates, step_fraction=0.004, initial_shear_rate=100.0, log=print):
    NN, NE = len(P), len(T)
    eta0 = zero_shear_viscosity(flow, melt_temp) * 1e-6
    tau, n = float(flow["tau"]) * 1e-6, float(flow["n"])

    def visc(rate):
        return eta0 / (1 + (eta0 * rate / tau) ** (1 - n))

    X = P[T]
    J = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=1)
    Ji = np.linalg.inv(J)
    vol = np.abs(np.linalg.det(J)) / 6
    G = np.zeros((NE, 4, 3))
    G[:, 1:, :] = np.transpose(Ji, (0, 2, 1))
    G[:, 0, :] = -G[:, 1:, :].sum(1)
    Kloc = np.einsum("eia,eja->eij", G, G) * vol[:, None, None]
    he = th[T].mean(1)
    Vn = np.zeros(NN)
    np.add.at(Vn, T.ravel(), np.repeat(vol / 4, 4))
    rows = np.repeat(T, 4, axis=1).ravel()
    cols = np.tile(T, (1, 4)).ravel()

    gate = gate_nodes(P, F, gates)
    Q = vol.sum() / fill_time
    f = np.zeros(NN)
    f[gate] = 1
    tf = np.full(NN, np.nan)
    tf[gate] = 0
    p98, p = None, np.zeros(NN)
    t, it = 0.0, 0
    gam = np.full(NE, float(initial_shear_rate))
    hist = []
    while True:
        full = f >= 1 - 1e-9
        if full.all():
            break
        ke = he ** 2 / (12 * visc(gam))
        wet = full[T].any(1)
        K = sp.csr_matrix(((Kloc * ke[:, None, None])[wet].ravel(),
                           (rows.reshape(NE, 16)[wet].ravel(), cols.reshape(NE, 16)[wet].ravel())), shape=(NN, NN))
        unk = np.where(full)[0]
        unk = unk[~np.isin(unk, gate)]
        p = np.zeros(NN)
        p[gate] = 1.0
        if len(unk):
            p[unk] = sla.spsolve(K[unk][:, unk].tocsc(), -K[unk][:, gate] @ p[gate])
        q = -(K @ p)
        s = Q / -q[gate].sum()
        p *= s
        q *= s
        front = np.where((~full) & (q > 1e-12))[0]
        if len(front) == 0:
            raise RuntimeError("the melt front stopped: some of the part is not connected to the gate")
        tt = Vn[front] * (1 - f[front]) / q[front]
        k = max(1, int(step_fraction * NN))
        dt = np.partition(tt, min(k, len(tt) - 1))[min(k, len(tt) - 1)]
        add = np.minimum(q[front] * dt, Vn[front] * (1 - f[front]))
        f[front] += add / Vn[front]
        tnew = t + add.sum() / Q
        done = f[front] >= 1 - 1e-9
        newly = front[done]
        f[newly] = 1
        tf[newly] = t + np.minimum(tt[done], dt) * (tnew - t) / dt
        t = tnew
        u = np.linalg.norm(np.einsum("eia,ei->ea", G, p[T]), axis=1) * ke
        gam = np.where(wet, np.maximum(6 * u / he, 1.0), gam)
        hist.append((t, p[gate].mean(), full.sum() / NN))
        if p98 is None and full.mean() >= 0.98:
            p98 = p.copy()
        it += 1
        if it % 100 == 0:
            log(f"    fill step {it}: t={t:.3f} s, gate {p[gate].mean():.1f} MPa, {full.mean():.0%} filled")
    tf[np.isnan(tf)] = t
    if p98 is None:
        p98 = p.copy()

    # weld lines: fronts arriving at a node from opposing directions
    gt = np.einsum("eia,ei->ea", G, tf[T])
    gt /= np.maximum(np.linalg.norm(gt, axis=1), 1e-12)[:, None]
    order = np.argsort(T.ravel(), kind="stable")
    en = np.repeat(np.arange(NE), 4)[order]
    nd = T.ravel()[order]
    starts, ends = np.searchsorted(nd, np.arange(NN)), np.searchsorted(nd, np.arange(NN), side="right")
    weld = np.zeros(NN)
    for i in range(NN):
        V = gt[en[starts[i]:ends[i]]]
        weld[i] = -(V @ V.T).min() if len(V) else 0.0
    gn = np.zeros(NN)
    np.maximum.at(gn, T.ravel(), np.repeat(gam, 4))
    return dict(tf=tf, p=p98, pend=p, gam=gn, weld=weld, hist=np.array(hist), gate=gate)


def defects(P, T, sv, tf, weld, weld_cos_threshold=0.65, tf_window=(0.05, 0.985), exclude_boxes=()):
    """Weld-line and air-trap candidates on the surface. Exclude boxes mark vented zones such as the parting line."""
    from .common import in_box
    w = sv[(weld[sv] > weld_cos_threshold) & (tf[sv] < tf_window[1] * tf.max()) & (tf[sv] > tf_window[0])]
    E = np.vstack([T[:, [a, b]] for a in range(4) for b in range(4) if a != b])
    mx = np.full(len(P), -1.0)
    np.maximum.at(mx, E[:, 0], tf[E[:, 1]])
    keep = np.ones(len(P), bool)
    for box in exclude_boxes:
        keep &= ~in_box(P, box)
    at = sv[(tf[sv] > mx[sv] + 1e-6) & keep[sv]]
    return P[w], P[at]
