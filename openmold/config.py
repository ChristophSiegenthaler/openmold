"""Project configuration: load YAML, apply defaults, validate, resolve paths."""
import copy, hashlib, json, math, re
from pathlib import Path
import yaml

PACKAGE_MATERIALS = Path(__file__).parent / "materials"

DEFAULTS = {
    "demould_axis": [0, 1, 0],
    "process": {"melt_temp": 225.0, "mold_temp": 40.0, "room_temp": 23.0, "fill_time": 2.0,
                "htc": 2500.0, "eject_times": [35.0]},
    "cooling": {"steps": [[0.5, 4], [2, 26], [5, 30], [10, 180]], "htc_zones": [],
                "report_times": [20, 30, 45, 60], "wall_probe": None},
    "mesh": {"fill": 1.3, "thermal": 1.0, "warp": 2.0, "surface": 1.0},
    "thickness": {"samples": 600000, "sample_radius": 0.08, "initial_radius": None, "min": 0.3},
    "fill": {"step_fraction": 0.004, "initial_shear_rate": 100.0, "weld_cos_threshold": 0.65,
             "weld_tf_window": [0.05, 0.985], "air_trap_exclude": []},
    "warp": {"supports": None, "flatness": [], "bows": [], "lateral": []},
    "exports": {"parts": [], "eject_time": None, "scales": [1], "coloured_ply": True, "step": False},
    "plots": {"warp_scale": 30, "section": None, "section_times": None, "detail": None, "curve_start": 6.0},
    "report": {"title": None, "notes_html": None, "pdf": False},
    "tools": {"ccx": "ccx", "freecad": "freecadcmd", "chromium": "chromium", "ccx_threads": 1},
}


class ConfigError(ValueError):
    pass


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_material(ref, base_dir):
    candidates = [Path(base_dir) / ref, PACKAGE_MATERIALS / f"{ref}.yaml"]
    for path in candidates:
        if path.is_file():
            return yaml.safe_load(path.read_text())
    raise ConfigError(f"material '{ref}' not found. Looked at: {', '.join(map(str, candidates))}")


def _check_table(value, label, errors):
    if isinstance(value, (int, float)):
        return
    temps = [row[0] for row in value] if isinstance(value, list) and value else None
    if not temps or temps != sorted(temps) or any(len(row) != 2 for row in value):
        errors.append(f"{label} must be a number or an ascending table of [T, value] rows")


def validate(cfg):
    errors = []
    if not cfg.get("project", {}).get("name"):
        errors.append("project.name is required")
    parts = cfg.get("parts") or []
    if not parts:
        errors.append("parts must list at least one part")
    ids = [p.get("id", "") for p in parts]
    if len(set(ids)) != len(ids):
        errors.append("part ids must be unique")
    for p in parts:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(p.get("id", ""))):
            errors.append(f"part id '{p.get('id')}' may only use letters, digits, - and _")
        if not p.get("step") or not Path(p["step"]).is_file():
            errors.append(f"part '{p.get('id')}': STEP file not found: {p.get('step')}")
    if not cfg.get("gates"):
        errors.append("gates must list at least one gate")
    for g in cfg.get("gates") or []:
        if len(g.get("point", [])) != 3:
            errors.append("each gate needs point: [x, y, z]")
    if not any(cfg["demould_axis"]) or len(cfg["demould_axis"]) != 3:
        errors.append("demould_axis must be a non-zero [x, y, z] vector")

    m = cfg["material"]
    flow = m.get("flow", {})
    need = {"cross": ["eta0", "tau", "n"], "cross_wlf": ["n", "tau", "D1", "D2", "A1", "A2"]}.get(flow.get("model"))
    if need is None:
        errors.append("material.flow.model must be cross or cross_wlf")
    else:
        errors += [f"material.flow.{k} is required" for k in need if k not in flow]
    for key in ("conductivity", "specific_heat", "no_flow_temp", "eject_temp"):
        if key not in m.get("thermal", {}):
            errors.append(f"material.thermal.{key} is required")
    for key in ("conductivity", "specific_heat"):
        if key in m.get("thermal", {}):
            _check_table(m["thermal"][key], f"material.thermal.{key}", errors)
    for key in ("youngs_modulus", "poisson", "expansion"):
        if key not in m.get("mechanical", {}):
            errors.append(f"material.mechanical.{key} is required")
    if "density" not in m:
        errors.append("material.density is required")

    pr = cfg["process"]
    if pr["mold_temp"] >= pr["melt_temp"]:
        errors.append("process.mold_temp must be below process.melt_temp")
    total = sum(d for _, d in cfg["cooling"]["steps"])
    first = cfg["cooling"]["steps"][0][0] if cfg["cooling"]["steps"] else 0
    for t in pr["eject_times"]:
        if not first < t <= total:
            errors.append(f"eject time {t} s is outside the simulated cooling window ({first} to {total} s)")
    if m.get("thermal", {}).get("eject_temp", 0) >= m.get("thermal", {}).get("no_flow_temp", math.inf):
        errors.append("material.thermal.eject_temp must be below no_flow_temp")
    if cfg["warp"]["supports"] is not None and len(cfg["warp"]["supports"]) != 3:
        errors.append("warp.supports must list exactly 3 points or be omitted")
    known = set(ids)
    for p in cfg["exports"]["parts"]:
        if p not in known:
            errors.append(f"exports.parts names unknown part '{p}'")
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
    return cfg


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - set(DEFAULTS) - {"project", "parts", "gates", "material"}
    if unknown:
        raise ConfigError(f"unknown config section(s): {', '.join(sorted(unknown))}")
    cfg = _merge(DEFAULTS, raw)
    base = path.parent
    for p in cfg.get("parts") or []:
        p["id"] = str(p.get("id", ""))
        if p.get("step"):
            p["step"] = str((base / p["step"]).resolve())
        p.setdefault("label", p["id"])
    if "material" not in cfg:
        raise ConfigError("material is required (a packaged name such as generic_tpu_65d, or a path)")
    if isinstance(cfg["material"], str):
        cfg["material"] = load_material(cfg["material"], base)
    cfg["_dir"] = str(base)
    cfg["output"] = str((base / cfg.get("project", {}).get("output", "runs/" + str(cfg.get("project", {}).get("name", "project")))).resolve())
    if cfg["report"]["notes_html"]:
        cfg["report"]["notes_html"] = str((base / cfg["report"]["notes_html"]).resolve())
    return validate(cfg)


def digest(*objs):
    return hashlib.sha256(json.dumps(objs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
