"""Runs the analysis steps per part. Finished steps are skipped unless their inputs changed."""
import json, os, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from .config import digest, file_digest

STEPS = ["mesh", "thickness", "fill", "thermal", "warp", "draft", "export"]


class Stamps:
    def __init__(self, directory):
        self.path = Path(directory) / ".stamps.json"
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def fresh(self, name, key, outputs):
        return self.data.get(name) == key and all(Path(o).exists() for o in outputs)

    def mark(self, name, key):
        self.data[name] = key
        self.path.write_text(json.dumps(self.data, indent=1))


def _save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))


def run_part(cfg, part, only=None, force=False, log=print):
    """Run all selected steps for one part. Returns the part id."""
    # imported here so that worker processes start with the thread settings of the parent
    from . import ccx, draft, exports, fill, geometry, thermal, thickness, warp
    from .common import load_tets
    pid = part["id"]
    d = Path(cfg["output"]) / pid
    d.mkdir(parents=True, exist_ok=True)
    stamps = Stamps(d)
    mat, pr = cfg["material"], cfg["process"]
    tools = cfg["tools"]

    def step(group, name, key, outputs, fn):
        if only and group not in only:
            return
        if not force and stamps.fresh(name, key, outputs):
            log(f"  {pid}: {name} is up to date")
            return
        t0 = time.time()
        log(f"  {pid}: {name} ...")
        fn()
        stamps.mark(name, key)
        log(f"  {pid}: {name} done in {time.time() - t0:.0f} s")

    src = file_digest(part["step"])
    k_mesh = {kind: digest(src, cfg["mesh"][kind]) for kind in ("fill", "thermal", "warp", "surface")}
    for kind, order in (("fill", 1), ("thermal", 1), ("warp", 2)):
        step("mesh", f"mesh_{kind}", k_mesh[kind], [d / f"{kind}.msh"],
             lambda kind=kind, order=order: log(f"  {pid}: {kind} mesh has {geometry.volume_mesh(part['step'], cfg['mesh'][kind], order, d / f'{kind}.msh')} nodes"))

    def do_surface():
        P, F = geometry.surface_mesh(part["step"], cfg["mesh"]["surface"])
        np.savez(d / "surface.npz", P=P, F=F)
    step("mesh", "mesh_surface", k_mesh["surface"], [d / "surface.npz"], do_surface)

    k_geo = digest(k_mesh["fill"], cfg["thickness"])

    def do_thickness():
        P, T = load_tets(d / "fill.msh")
        tc = cfg["thickness"]
        res, info = thickness.compute(P, T, tc["samples"], tc["sample_radius"], tc["initial_radius"], tc["min"])
        np.savez(d / "geo.npz", P=P, T=T, **res)
        _save(d / "geo.json", info)
    step("thickness", "thickness", k_geo, [d / "geo.npz", d / "geo.json"], do_thickness)

    fc = cfg["fill"]
    k_fill = digest(k_geo, mat["flow"], pr["melt_temp"], pr["fill_time"], cfg["gates"], fc)

    def do_fill():
        g = np.load(d / "geo.npz")
        r = fill.solve(g["P"], g["T"], g["th"], g["F"], mat["flow"], pr["melt_temp"], pr["fill_time"], cfg["gates"],
                       fc["step_fraction"], fc["initial_shear_rate"], log=log)
        w, a = fill.defects(g["P"], g["T"], g["sv"], r["tf"], r["weld"], fc["weld_cos_threshold"], fc["weld_tf_window"], fc["air_trap_exclude"])
        np.savez(d / "fill.npz", **r)
        np.savez(d / "defects.npz", weld=w, air=a)
        _save(d / "fill.json", dict(pressure_98_mpa=float(r["p"][r["gate"]].mean()), weld_nodes=len(w), air_trap_nodes=len(a),
                                    zero_shear_viscosity_pas=fill.zero_shear_viscosity(mat["flow"], pr["melt_temp"]), gate_nodes=len(r["gate"])))
    step("fill", "fill", k_fill, [d / "fill.npz", d / "defects.npz", d / "fill.json"], do_fill)

    k_th = digest(k_mesh["thermal"], k_geo, mat["density"], mat["thermal"], mat.get("shrinkage"), pr["melt_temp"], pr["mold_temp"], pr["htc"], cfg["cooling"])

    def do_thermal():
        info = thermal.write_deck(d / "thermal.msh", d, cfg)
        ccx.run("th", d, tools["ccx"], tools["ccx_threads"])
        res = thermal.postprocess(d, cfg, np.load(d / "geo.npz"))
        res.update(info)
        _save(d / "th.json", res)
        (d / "th.frd").unlink()   # large; everything needed is in NT.npy and thres.npz
    step("thermal", "thermal", k_th, [d / "NT.npy", d / "thres.npz", d / "th.json"], do_thermal)

    for t in pr["eject_times"]:
        key = warp.fmt_t(t)
        k_w = digest(k_th, k_mesh["warp"], mat["mechanical"], pr["room_temp"], t, cfg["warp"])

        def do_warp(t=t, key=key):
            job = warp.write_deck(d / "warp.msh", d, d, cfg, t)
            ccx.run(job, d, tools["ccx"], tools["ccx_threads"])
            _save(d / f"warp_{key}.json", warp.postprocess(d / "warp.msh", d, cfg, t))
            (d / f"{job}.frd").unlink()
        step("warp", f"warp_{key}", k_w, [d / f"warp_{key}.npz", d / f"warp_{key}.json"], do_warp)

    limit = mat.get("recommended_draft_deg") or 2.0
    k_dr = digest(k_mesh["surface"], cfg["demould_axis"], limit)

    def do_draft():
        s = np.load(d / "surface.npz")
        a, info = draft.analyse(s["P"], s["F"], cfg["demould_axis"], limit)
        np.savez(d / "draft.npz", P=s["P"], F=s["F"], draft=a)
        _save(d / "draft.json", info)
    step("draft", "draft", k_dr, [d / "draft.npz", d / "draft.json"], do_draft)

    if pid in cfg["exports"]["parts"]:
        t = cfg["exports"]["eject_time"] or pr["eject_times"][0]
        key = warp.fmt_t(t)
        k_e = digest(k_mesh["surface"], k_th, k_mesh["warp"], mat["mechanical"], pr["room_temp"], t, cfg["warp"], cfg["exports"])
        step("export", "export", k_e, [d / "export.json"],
             lambda: _save(d / "export.json", exports.export_part(pid, d, Path(cfg["output"]) / "exports", cfg["exports"], t, tools, log)))
    return pid


def _worker(cfg, part, only, force):
    def log(msg):
        print(msg, flush=True)
    return run_part(cfg, part, only, force, log)


def run_all(cfg, parts=None, only=None, force=False, jobs=1, report=True, log=print):
    from . import plots, report as rep
    selected = [p for p in cfg["parts"] if not parts or p["id"] in parts]
    missing = set(parts or []) - {p["id"] for p in cfg["parts"]}
    if missing:
        raise ValueError(f"unknown part(s): {', '.join(sorted(missing))}")
    only = set(only) if only else None
    failed = {}
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # the fill solver and numpy; CalculiX sets its own count
    if jobs > 1 and len(selected) > 1:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_worker, cfg, p, only, force): p["id"] for p in selected}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as err:
                    failed[futures[fut]] = str(err)
    else:
        for p in selected:
            try:
                run_part(cfg, p, only, force, log)
            except Exception as err:
                log(traceback.format_exc())
                failed[p["id"]] = str(err)
    out = Path(cfg["output"])
    target = None
    if report:
        log("Writing figures and report ...")
        target = rep.build(cfg, out, plots.make_all(cfg, out))
        if cfg["report"]["pdf"]:
            log(f"PDF: {rep.to_pdf(target, cfg['tools']['chromium'])}")
    return target, failed
