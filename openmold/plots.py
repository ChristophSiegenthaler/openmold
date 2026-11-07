"""Report figures. Cameras come from the part's bounding box and demould axis, so any part orientation works."""
import numpy as np, pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import basis, nodal_volumes, unit
from .warp import fmt_t

pv.OFF_SCREEN = True
PALETTE = ["#c0392b", "#2874a6", "#1e8449", "#7d3c98", "#b07d0a", "#117a8b", "#a04000", "#34495e"]


class Frame:
    def __init__(self, P, axis):
        self.c = (P.min(0) + P.max(0)) / 2
        self.d = np.linalg.norm(P.max(0) - P.min(0))
        self.up = unit(axis)
        self.e1, self.e2 = basis(axis)

    def camera(self, view, detail=None):
        c, d, up, e1, e2 = self.c, self.d, self.up, self.e1, self.e2
        if view == "iso":
            return [c + d * (1.13 * e1 + 1.35 * up + 1.37 * e2), c, up]
        if view == "isob":
            return [c + d * (1.13 * e1 - 1.15 * up + 1.37 * e2), c, up]
        if view == "top":
            return [c + 1.6 * d * up, c, e2]
        if view == "bottom":
            return [c - 1.6 * d * up, c, e2]
        if view == "detail":
            focus = np.asarray(detail["center"], float)
            return [focus + 2.2 * detail["radius"] * (0.6 * e1 + 0.7 * up + 0.6 * e2), focus, up]
        raise ValueError(view)


def surf(P, F, field=None, name="v"):
    g = pv.PolyData(P, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    if field is not None:
        g.point_data[name] = field
    return g


def grid_figure(meshes, name, path, clim, frame, cmap="jet", views=("iso", "isob"), title="", extra=None, detail=None):
    rows, cols = len(views), len(meshes)
    p = pv.Plotter(shape=(rows, cols), window_size=(500 * cols, 480 * rows), border=False)
    for i, (pid, g) in enumerate(meshes):
        for r, view in enumerate(views):
            p.subplot(r, i)
            p.add_mesh(g, scalars=name, cmap=cmap, clim=clim, show_scalar_bar=(i == cols - 1 and r == 0),
                       scalar_bar_args=dict(title=title or name, vertical=True, position_x=0.85, position_y=0.1, height=0.8))
            if extra:
                extra(p, pid)
            p.add_text(f"{pid}  ({view})", font_size=11)
            p.camera_position = frame.camera(view, detail)
    p.screenshot(str(path))
    p.close()


def section_figure(rows, path, section, frame, titles):
    """rows: list of (per-part field getter, name, cmap, clim, scalar bar title, row label)."""
    ids = list(titles)
    n = np.asarray(section["normal"], float)
    o = np.asarray(section["origin"], float)
    p = pv.Plotter(shape=(len(rows), len(ids)), window_size=(500 * len(ids), 480 * len(rows)))
    cam = [o + 1.3 * frame.d * unit(n), o + (frame.c - o) - (frame.c - o).dot(unit(n)) * unit(n), frame.up]
    for r, (get, name, cmap, clim, bar, label) in enumerate(rows):
        for i, pid in enumerate(ids):
            grid, field = get(pid)
            grid.point_data[name] = field
            p.subplot(r, i)
            p.add_mesh(grid.slice(normal=n, origin=o), scalars=name, cmap=cmap, clim=clim, show_scalar_bar=(i == len(ids) - 1),
                       scalar_bar_args=dict(title=bar))
            p.add_text(f"{pid}  {label}", font_size=10)
            p.camera_position = cam
    p.screenshot(str(path))
    p.close()


def make_all(cfg, out):
    """Write every figure and return {figure key: path}. Parts missing a result are left out of that figure."""
    ids = [p["id"] for p in cfg["parts"]]
    mat, pr = cfg["material"], cfg["process"]
    TNF, TE = mat["thermal"]["no_flow_temp"], mat["thermal"]["eject_temp"]
    d = out / "plots"
    d.mkdir(parents=True, exist_ok=True)
    figs = {}
    has = lambda pid, name: (out / pid / name).exists()
    with_geo = [i for i in ids if has(i, "geo.npz")]
    if not with_geo:
        return figs
    geo = {i: np.load(out / i / "geo.npz") for i in with_geo}
    frame = Frame(geo[with_geo[0]]["P"], cfg["demould_axis"])
    section = cfg["plots"]["section"] or {"origin": frame.c.tolist(), "normal": frame.e1.tolist()}
    detail = cfg["plots"]["detail"]

    ms = [(i, surf(geo[i]["P"], geo[i]["F"], np.zeros(len(geo[i]["P"])), "z")) for i in with_geo]
    p = pv.Plotter(shape=(2, len(ms)), window_size=(500 * len(ms), 960), border=False)
    for k, (pid, g) in enumerate(ms):
        for r, view in enumerate(("iso", "isob")):
            p.subplot(r, k)
            p.add_mesh(g, color="lightsteelblue")
            p.add_text(f"{pid}  ({view})", font_size=11)
            p.camera_position = frame.camera(view)
    figs["view"] = d / "view.png"
    p.screenshot(str(figs["view"]))
    p.close()

    tmax = float(np.ceil(max(np.percentile(geo[i]["ts"][geo[i]["sv"]], 99.9) for i in with_geo)))
    ms = [(i, surf(geo[i]["P"], geo[i]["F"], geo[i]["ts"], "t")) for i in with_geo]
    figs["thickness"] = d / "thickness.png"
    grid_figure(ms, "t", figs["thickness"], (0, tmax), frame, views=("iso", "top"), title="Wall thickness [mm]")
    grids = {i: pv.UnstructuredGrid({pv.CellType.TETRA: geo[i]["T"]}, geo[i]["P"]) for i in with_geo}
    figs["thickness_section"] = d / "thickness_section.png"
    section_figure([(lambda i: (grids[i], geo[i]["th"]), "t", "jet", (0, tmax), "Local thickness [mm]", "section")],
                   figs["thickness_section"], section, frame, with_geo)

    with_fill = [i for i in with_geo if has(i, "fill.npz")]
    if with_fill:
        fills = {i: np.load(out / i / "fill.npz") for i in with_fill}
        defects = {i: np.load(out / i / "defects.npz") for i in with_fill}

        def marks(plotter, pid):
            if len(defects[pid]["weld"]):
                plotter.add_points(defects[pid]["weld"], color="magenta", point_size=6, render_points_as_spheres=True)
            if len(defects[pid]["air"]):
                plotter.add_points(defects[pid]["air"], color="black", point_size=12, render_points_as_spheres=True)
        ms = [(i, surf(geo[i]["P"], geo[i]["F"], fills[i]["tf"], "tf")) for i in with_fill]
        figs["fill_time"] = d / "fill_time.png"
        grid_figure(ms, "tf", figs["fill_time"], (0, pr["fill_time"]), frame, views=("isob", "iso") + (("detail",) if detail else ()),
                    title="Fill time [s]", extra=marks, detail=detail)

    with_th = [i for i in with_geo if has(i, "thres.npz")]
    if with_th:
        th = {i: np.load(out / i / "th.npz") for i in with_th}
        NT = {i: np.load(out / i / "NT.npy") for i in with_th}
        times = {i: np.load(out / i / "times.npy") for i in with_th}
        res = {i: np.load(out / i / "thres.npz") for i in with_th}
        grids_th = {i: pv.UnstructuredGrid({pv.CellType.TETRA: th[i]["T"]}, th[i]["P"]) for i in with_th}
        section_times = cfg["plots"]["section_times"] or [round(pr["eject_times"][0] * f) for f in (0.5, 1.0, 1.5)]
        rows = [((lambda tt: (lambda i: (grids_th[i], NT[i][int(np.argmin(np.abs(times[i] - tt)))])))(tt), "T", "inferno",
                 (pr["mold_temp"], pr["melt_temp"]), "T [C]", f"T at {tt:g} s") for tt in section_times]
        finite = max(float(np.percentile(res[i]["tnf"][np.isfinite(res[i]["tnf"])], 100)) for i in with_th)
        rows.append((lambda i: (grids_th[i], np.minimum(res[i]["tnf"], finite)), "ts", "viridis", (0, finite),
                     "time to solid [s]", f"time to <{TNF:g} C"))
        figs["thermal_section"] = d / "thermal_section.png"
        section_figure(rows, figs["thermal_section"], section, frame, with_th)

        smax = max(float(res[i]["sink_s"].max()) for i in with_th)
        ms = [(i, surf(th[i]["P"], th[i]["F"], res[i]["sink_s"], "sink")) for i in with_th]
        figs["sink"] = d / "sink.png"
        grid_figure(ms, "sink", figs["sink"], (0, max(smax, 1e-3)), frame, cmap="magma_r", views=("iso", "isob"), title="Sink estimate [mm]")

        start = cfg["plots"]["curve_start"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        tend = 0.0
        for k, i in enumerate(with_th):
            colour = PALETTE[k % len(PALETTE)]
            t, m = times[i], times[i] >= start
            vn = nodal_volumes(th[i]["P"], th[i]["T"])
            ax[0].plot(t[m], NT[i][m].max(1), color=colour, label=i)
            ax[1].plot(t[m], [(vn * (x > TNF)).sum() / 1000 for x in NT[i][m]], color=colour, label=i)
            te = res[i]["te"]
            tend = max(tend, float(te[np.isfinite(te)].max()) if np.isfinite(te).any() else t[-1])
        ax[0].axhline(TNF, ls="--", c="gray", lw=0.8)
        ax[0].axhline(TE, ls=":", c="gray", lw=0.8)
        limit = min(tend * 1.4, float(times[with_th[0]][-1]))
        ax[0].set(xlabel="time after fill [s]", ylabel="max part temperature [C]", xlim=(0, limit), ylim=(pr["mold_temp"], pr["melt_temp"] + 15))
        ax[1].set(xlabel="time after fill [s]", ylabel=f"molten volume >{TNF:g} C [cm3]", xlim=(0, limit))
        for a in ax:
            a.grid(alpha=0.3)
            a.legend()
        plt.tight_layout()
        figs["cooling_curves"] = d / "cooling_curves.png"
        plt.savefig(figs["cooling_curves"], dpi=110)
        plt.close(fig)

    te0 = fmt_t(pr["eject_times"][0])
    with_w = [i for i in with_geo if has(i, f"warp_{te0}.npz")]
    if with_w:
        k = cfg["plots"]["warp_scale"]
        w = {i: np.load(out / i / f"warp_{te0}.npz") for i in with_w}
        mags = {i: np.linalg.norm(w[i]["W"], axis=1) for i in with_w}
        ms = [(i, surf(w[i]["P"] + k * w[i]["W"], w[i]["F"], mags[i], "w")) for i in with_w]
        figs["warpage"] = d / "warpage.png"
        grid_figure(ms, "w", figs["warpage"], (0, max(float(m.max()) for m in mags.values())), frame, cmap="turbo",
                    views=("iso", "isob"), title=f"Warpage [mm], shape x{k:g}")

    with_d = [i for i in with_geo if has(i, "draft.npz")]
    if with_d:
        limit = (mat.get("recommended_draft_deg") or 2.0)
        ds = {i: np.load(out / i / "draft.npz") for i in with_d}
        ms = []
        for i in with_d:
            P, F, a = ds[i]["P"], ds[i]["F"], ds[i]["draft"]
            g = surf(P, F)
            g.cell_data["draft"] = np.minimum(a, 2 * limit)
            ms.append((i, g))
        figs["draft"] = d / "draft.png"
        grid_figure(ms, "draft", figs["draft"], (0, 2 * limit), frame, cmap="RdYlGn", views=("iso", "isob"), title="Draft [deg]")
    return figs
