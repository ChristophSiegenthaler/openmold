"""Self-contained HTML report: images, mesh data and three.js are all inlined, so the single file works offline."""
import base64, html, json, shutil, struct, subprocess
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .plots import PALETTE, Frame
from .warp import fmt_t

ASSETS = Path(__file__).parent / "assets"


def _load_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def collect(cfg, out):
    """Merge all per-part JSON results into one summary."""
    summary = {}
    for p in cfg["parts"]:
        d = out / p["id"]
        row = dict(label=p["label"], **_load_json(d / "geo.json"), **_load_json(d / "fill.json"), **_load_json(d / "th.json"))
        row["draft"] = _load_json(d / "draft.json")
        row["warp"] = {fmt_t(t): _load_json(d / f"warp_{fmt_t(t)}.json") for t in cfg["process"]["eject_times"]}
        summary[p["id"]] = row
    return summary


def write_tables(cfg, out, summary):
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    rows = metric_rows(cfg, summary)
    lines = ["metric," + ",".join(summary)]
    for r in rows:
        if r.get("group"):
            continue
        lines.append('"' + r["label"] + '",' + ",".join("" if v is None else f"{v:.4g}" for v in r["values"]))
    (out / "summary.csv").write_text("\n".join(lines) + "\n")


def metric_rows(cfg, summary):
    ids = list(summary)
    rho = cfg["material"]["density"]
    te = cfg["process"]["eject_times"]
    TNF = cfg["material"]["thermal"]["no_flow_temp"]
    TE = cfg["material"]["thermal"]["eject_temp"]
    get = lambda key: [summary[i].get(key) for i in ids]
    rows = [dict(group="Geometry"),
            dict(label="Part volume [cm3]", values=[None if v is None else v / 1000 for v in get("volume_mm3")], rank=False),
            dict(label=f"Shot weight at {rho:g} kg/m3 [g]", values=[None if v is None else v * rho * 1e-6 for v in get("volume_mm3")], rank=True),
            dict(label="Nominal wall, median [mm]", values=get("nominal_wall_mm"), rank=False),
            dict(label="Thickest section [mm]", values=get("thickness_max_mm"), rank=True),
            dict(label=f"Side faces under {summary[ids[0]]['draft'].get('limit_deg', 0):g} deg draft [%]",
                 values=[summary[i]["draft"].get("below_limit_pct") for i in ids], rank=True),
            dict(group="Filling"),
            dict(label="Gate pressure at 98 % fill, isothermal [MPa]", values=get("pressure_98_mpa"), rank=True),
            dict(group="Cooling"),
            dict(label=f"Whole part below {TNF:g} C [s]", values=get("t_frozen_100"), rank=True),
            dict(label=f"95 % of volume below {TE:g} C [s]", values=get("t_eject_95"), rank=True),
            dict(label="Sink estimate, smoothed [mm]", values=get("sink_smoothed_max_mm"), rank=True),
            dict(label="Sink estimate, peak [mm]", values=get("sink_max_mm"), rank=True)]
    for t in te:
        key = fmt_t(t)
        rows.append(dict(group=f"Warpage after ejection at {key} s"))
        rows.append(dict(label="Max. distortion [mm]", values=[summary[i]["warp"][key].get("warp_max_mm") for i in ids], rank=True))
        names = sorted({k for i in ids for k in summary[i]["warp"][key] if k.startswith(("flatness_", "bow_", "lateral_"))})
        for name in names:
            label = name.removesuffix("_mm").replace("_", " ").capitalize() + " [mm]"
            rows.append(dict(label=label, values=[summary[i]["warp"][key].get(name) for i in ids], rank=name.startswith(("flatness_", "lateral_"))))
    return [r for r in rows if r.get("group") or any(v is not None for v in r["values"])]


def _number(v):
    if v is None:
        return "–"
    if np.isinf(v):
        return "not reached"
    return f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}"


def _table(cfg, summary):
    ids = list(summary)
    head = "".join(f'<th><span class="tag" style="--k:{PALETTE[i % len(PALETTE)]}">{html.escape(p)}</span></th>' for i, p in enumerate(ids))
    body = []
    for r in metric_rows(cfg, summary):
        if r.get("group"):
            body.append(f'<tr class="grp"><td colspan="{len(ids) + 1}">{html.escape(r["group"])}</td></tr>')
            continue
        vals = [v for v in r["values"] if v is not None]
        spread = r.get("rank") and len(vals) > 1 and min(vals) != max(vals) and (np.isinf(max(vals)) or max(vals) - min(vals) > 1e-3 * abs(max(vals)))
        cells = []
        for v in r["values"]:
            cls = "num"
            if v is not None and spread:
                cls += " best" if v == min(vals) else " worst" if v == max(vals) else ""
            cells.append(f'<td class="{cls}">{_number(v)}</td>')
        body.append(f'<tr><td>{html.escape(r["label"])}</td>{"".join(cells)}</tr>')
    return f'<div class="tbl"><table><thead><tr><th>Result</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _mesh_blob(cfg, out, pid):
    d = out / pid
    g, fl, df = np.load(d / "geo.npz"), np.load(d / "fill.npz"), np.load(d / "defects.npz")
    P, F, tf = g["P"], g["F"], fl["tf"]
    tmax = cfg["process"]["fill_time"]
    sv, inv = np.unique(F, return_inverse=True)
    wide = len(sv) > 65535
    idx = inv.reshape(-1, 3).astype(np.uint32 if wide else np.uint16)
    scale = float(np.abs(P[sv]).max() / 32000)
    pos = np.round(P[sv] / scale).astype(np.int16)
    tq = np.round(np.clip(tf[sv] / tmax, 0, 1) * 65535).astype(np.uint16)
    tree = cKDTree(P)

    def points(X):
        if len(X) == 0:
            return np.zeros((0, 4), np.float32)
        _, j = tree.query(X)
        a = np.column_stack([X, tf[j]]).astype(np.float32)
        return a[np.argsort(a[:, 3])]

    W, A = points(df["weld"]), points(df["air"])
    b = struct.pack("<5If", len(sv), len(idx), len(W), len(A), int(wide), scale) + pos.tobytes() + tq.tobytes() + idx.tobytes()
    b += b"\0" * (-len(b) % 4) + W.tobytes() + A.tobytes()
    return base64.b64encode(b).decode()


def _animation(cfg, out, summary):
    ids = [p for p in summary if (out / p / "fill.npz").exists()]
    if not ids:
        return "", ""
    geo = np.load(out / ids[0] / "geo.npz")
    fr = Frame(geo["P"], cfg["demould_axis"])
    views = {"gate": fr.camera("isob"), "open": fr.camera("iso")}
    detail = cfg["plots"]["detail"]
    if detail:
        views["detail"] = fr.camera("detail", detail)
    fa = dict(tmax=cfg["process"]["fill_time"], up=fr.up.tolist(),
              parts=[dict(id=p, label=summary[p]["label"], colour=PALETTE[list(summary).index(p) % len(PALETTE)]) for p in ids],
              views={k: dict(p=[float(x) for x in v[0]], t=[float(x) for x in v[1]]) for k, v in views.items()})
    tmax = cfg["process"]["fill_time"]
    buttons = '<button type="button" data-v="gate" aria-pressed="true">Below</button><button type="button" data-v="open" aria-pressed="false">Above</button>'
    if detail:
        buttons += '<button type="button" data-v="detail" aria-pressed="false">Detail</button>'
    block = f"""<h3>3D fill animation</h3>
  <p>The same fill times, played back on all parts at once. Press Play or drag the slider. Drag any view to rotate all of them. Markers are drawn through the walls so they stay visible.</p>
  <p class="print-hint" style="display:none">The interactive animation is only available in the HTML version of this report.</p>
  <div id="fa" style="--cols:{1 if len(ids) == 1 else 2}">
    <div class="bar" role="group" aria-label="Fill animation controls">
      <button id="fa-play" type="button">Play</button>
      <div class="time" id="fa-tread">0.00 <small>s</small></div>
      <div class="scrub"><input id="fa-t" type="range" min="0" max="{tmax:g}" step="0.005" value="{0.6 * tmax:g}" aria-label="Fill time in seconds"></div>
      <div class="group" id="fa-speed"><span class="lbl">Speed</span><button type="button" data-s="0.1" aria-pressed="false">0.1×</button><button type="button" data-s="0.25" aria-pressed="true">0.25×</button><button type="button" data-s="1" aria-pressed="false">1×</button></div>
      <div class="group" id="fa-views"><span class="lbl">View</span>{buttons}</div>
      <div class="group" id="fa-zoom"><span class="lbl">Zoom</span><button type="button" data-z="0.8" aria-label="Zoom in">+</button><button type="button" data-z="1.25" aria-label="Zoom out">−</button></div>
      <label class="chk"><input id="fa-marks" type="checkbox" checked> Weld lines &amp; air traps</label>
    </div>
    <div class="grid" id="fa-grid"></div>
    <div class="legend">
      <span class="cbar"><span>0 s</span><span class="g"></span><span>{tmax:g} s</span></span>
      <span>filled, coloured by arrival time</span>
      <span><span class="dot" style="--k:var(--ghost)"></span>still empty</span>
      <span><span class="dot" style="--k:var(--weld)"></span>weld line</span>
      <span><span class="dot" style="--k:var(--air)"></span>air trap</span>
    </div>
  </div>"""
    meshes = "".join(f'<script type="text/plain" id="mesh-{p}">{_mesh_blob(cfg, out, p)}</script>\n' for p in ids)
    libs = "".join(f"<script>\n{(ASSETS / f).read_text().replace('</script', '<' + chr(92) + '/script')}\n</script>\n" for f in ("three.min.js", "OrbitControls.js"))
    scripts = f"{meshes}{libs}<script>window.FA={json.dumps(fa)};</script>\n<script>\n{(ASSETS / 'animation.js').read_text()}\n</script>\n"
    return block, scripts


def _figure(figs, key, caption):
    if key not in figs:
        return ""
    data = base64.b64encode(Path(figs[key]).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def build(cfg, out, figs):
    summary = collect(cfg, out)
    write_tables(cfg, out, summary)
    mat, pr, mesh = cfg["material"], cfg["process"], cfg["mesh"]
    title = cfg["report"]["title"] or f"{cfg['project']['name']}: moulding comparison"
    ids = list(summary)
    t0 = fmt_t(pr["eject_times"][0])
    block, scripts = _animation(cfg, out, summary)
    notes = Path(cfg["report"]["notes_html"]).read_text() if cfg["report"]["notes_html"] else ""
    warn = "" if mat.get("measured") else '<p><b>This material card is not measured.</b> Its values are generic estimates, so absolute numbers are only orders of magnitude.</p>'
    tags = " ".join(f'<span class="tag" style="--k:{PALETTE[i % len(PALETTE)]}">{html.escape(p)}</span> {html.escape(summary[p]["label"])}' for i, p in enumerate(ids))
    zones = "".join(f"<li>{html.escape(z.get('name', 'zone'))}: {z['htc']:g} W/m²K</li>" for z in cfg["cooling"]["htc_zones"])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{html.escape(cfg['project']['name'])}</title>
<style>
{(ASSETS / 'report.css').read_text()}
</style></head><body>
<div class="wrap">
<header>
  <div class="eyebrow">Injection-moulding simulation · {html.escape(mat['name'])}</div>
  <h1>{html.escape(title)}</h1>
  <p class="lede">{tags}</p>
</header>
{f'<section><h2>Notes</h2>{notes}</section>' if notes else ''}
<section><h2>Key numbers</h2>{_table(cfg, summary)}
<p style="color:var(--muted);font-size:14px">Green marks the lowest value and red the highest where lower is better. Use these as a ranking and as orders of magnitude, not as tolerances.</p></section>
<section><h2>Geometry and wall thickness</h2>
{_figure(figs, 'view', 'The parts as meshed for the fill analysis.')}
{_figure(figs, 'thickness', 'Surface wall thickness from the inscribed-sphere method.')}
{_figure(figs, 'thickness_section', 'Local thickness on the section plane.')}
{_figure(figs, 'draft', 'Draft angle against the demould direction. Faces more than 10 degrees off are not sliding faces. The check does not know the parting line, so it cannot detect undercuts.')}
</section>
<section><h2>Filling</h2>
<p>Melt {pr['melt_temp']:g} °C, fill time {pr['fill_time']:g} s, constant flow rate.</p>
{_figure(figs, 'fill_time', 'Fill time. Magenta: weld line candidates. Black: air trap candidates.')}
{block}
</section>
<section><h2>Cooling and sink marks</h2>
<p>Mould {pr['mold_temp']:g} °C, heat transfer {pr['htc']:g} W/m²K. Packing is assumed to stay effective until the centre of the nominal wall is solid. Melt that is still molten at that moment shrinks without compensation, which gives the sink estimate.</p>
{f'<ul>{zones}</ul>' if zones else ''}
{_figure(figs, 'cooling_curves', 'Maximum part temperature and remaining molten volume. The first seconds are left out because of a numerical overshoot (see limits).')}
{_figure(figs, 'thermal_section', 'Temperature on the section plane, and the time to reach the no-flow temperature.')}
{_figure(figs, 'sink', 'Estimated sink depth, smoothed.')}
</section>
<section><h2>Warpage</h2>
<p>The part leaves the mould at {t0} s with its simulated temperature field and cools freely to {pr['room_temp']:g} °C. Rigid-body motion and uniform shrinkage are removed.</p>
{_figure(figs, 'warpage', f'Distortion magnitude after ejection at {t0} s. The shape is exaggerated, the colours are true values.')}
</section>
<section><h2>Method and limits</h2>
<div class="note">
<p><b>Material:</b> {html.escape(mat['name'])}. Source: {html.escape(mat.get('source', 'not stated'))}.</p>
{warn}
<p><b>Tool chain:</b> gmsh (meshes: fill {mesh['fill']:g} mm, thermal {mesh['thermal']:g} mm, warpage {mesh['warp']:g} mm) → openmold fill solver (isothermal Hele-Shaw, control volume) → CalculiX (DC3D4 cooling, C3D10 warpage) → PyVista and three.js.</p>
<p><b>Not modelled:</b> frozen layer and shear heating in filling, so absolute pressures are low. Packing and pvT behaviour. Shrinkage while the part is still constrained in the mould, so the shrinkage here is not a tool shrink factor. Flow-induced anisotropy. Mould steel and cooling channels, which are replaced by heat transfer coefficients. The gate position is an input.</p>
<p><b>Known artefact:</b> the implicit thermal solution on linear tets overshoots near the hot core in the first seconds (consistent mass matrix). It dies out early and the cooling curves start after it.</p>
<p><b>Validation:</b> the test suite checks the fill pressure, slab cooling and free shrinkage against analytical solutions. The pipeline has not been checked against moulded parts or a commercial solver.</p>
</div></section>
</div>
{scripts}</body></html>
"""
    target = out / "report.html"
    target.write_text(page)
    return target


def to_pdf(html_path, chromium="chromium"):
    if not shutil.which(chromium):
        raise RuntimeError(f"PDF export needs '{chromium}', which was not found")
    pdf = Path(html_path).with_suffix(".pdf")
    subprocess.run([chromium, "--headless", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{Path(html_path).resolve()}"],
                   capture_output=True, check=True)
    return pdf
