"""Checks of each solver against analytical solutions, on simple shapes."""
import math
import numpy as np, pytest, gmsh
from openmold import ccx, fill, geometry, thermal, warp
from openmold.common import load_tets

MAT = dict(density=1173, thermal=dict(conductivity=0.22, specific_heat=1900, no_flow_temp=160, eject_temp=80),
           mechanical=dict(youngs_modulus=900, poisson=0.45, expansion=1.4e-4), shrinkage=dict(sink_volumetric=0.06))
CFG = dict(material=MAT, process=dict(melt_temp=225.0, mold_temp=40.0, room_temp=23.0, htc=2500.0),
           cooling=dict(steps=[[0.5, 4], [1, 26]], htc_zones=[], report_times=[], wall_probe=None),
           warp=dict(supports=None, flatness=[], bows=[], lateral=[]))


def make_box(path, dx, dy, dz):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.addBox(0, 0, 0, dx, dy, dz)
    gmsh.model.occ.synchronize()
    gmsh.write(str(path))
    gmsh.finalize()


def test_fill_pressure_matches_hele_shaw_strip(tmp_path):
    """Newtonian strip gated across one end: dp = 12 eta Q L / (W h^3) and the front moves at constant speed."""
    L, W, h, tfill, eta = 100.0, 20.0, 4.0, 2.0, 1000.0
    make_box(tmp_path / "s.step", L, h, W)
    geometry.volume_mesh(str(tmp_path / "s.step"), 2.0, 1, tmp_path / "s.msh")
    P, T = load_tets(tmp_path / "s.msh")
    from openmold.common import boundary
    F = boundary(T)
    th = np.full(len(P), h)
    flow = dict(model="cross", eta0=eta, tau=1e12, n=0.35)   # huge tau makes it Newtonian
    gates = [{"point": [0, y, z], "radius": 1.5} for y in (1, 3) for z in (2, 6, 10, 14, 18)]
    r = fill.solve(P, T, th, F, flow, 225.0, tfill, gates, log=lambda m: None)
    Q = L * W * h / tfill
    expected = 12 * (eta * 1e-6) * Q * (0.98 * L) / (W * h ** 3)
    assert r["p"][r["gate"]].mean() == pytest.approx(expected, rel=0.2)
    mid = np.abs(P[:, 0] - L / 2) < 2
    assert r["tf"][mid].mean() == pytest.approx(tfill / 2, rel=0.1)


def analytic_slab_time(T_target, half, cfg):
    """First term of the slab series with finite Biot number, for the centre temperature."""
    m, pr = cfg["material"], cfg["process"]
    k, alpha = m["thermal"]["conductivity"], m["thermal"]["conductivity"] / (m["density"] * m["thermal"]["specific_heat"])
    half = half * 1e-3   # mm to m
    bi = pr["htc"] * half / k
    z = 1.0
    for _ in range(200):
        z = math.atan(bi / z)
    c1 = 4 * math.sin(z) / (2 * z + math.sin(2 * z))
    theta = (T_target - pr["mold_temp"]) / (pr["melt_temp"] - pr["mold_temp"])
    return math.log(c1 / theta) / z ** 2 * half ** 2 / alpha


@pytest.mark.solver
def test_slab_cooling_matches_series_solution(tmp_path):
    t = 4.0
    make_box(tmp_path / "p.step", 60, t, 60)
    geometry.volume_mesh(str(tmp_path / "p.step"), 1.2, 1, tmp_path / "thermal.msh")
    thermal.write_deck(tmp_path / "thermal.msh", tmp_path, CFG)
    ccx.run("th", tmp_path)
    d = np.load(tmp_path / "th.npz")
    NT = np.array([v[np.argsort(i), 0] for _, i, v in ccx.read_frd(tmp_path / "th.frd")])
    times = np.array([r[0] for r in ccx.read_frd(tmp_path / "th.frd")])
    P = d["P"]
    centre = int(np.argmin(np.linalg.norm(P - [30, t / 2, 30], axis=1)))
    sim = thermal.crossing_times(NT, times, 120.0)[centre]
    assert sim == pytest.approx(analytic_slab_time(120.0, t / 2, CFG), rel=0.15)


@pytest.mark.solver
def test_uniform_temperature_gives_pure_shrink_and_no_warp(tmp_path):
    make_box(tmp_path / "b.step", 40, 10, 20)
    geometry.volume_mesh(str(tmp_path / "b.step"), 3.0, 2, tmp_path / "warp.msh")
    geometry.volume_mesh(str(tmp_path / "b.step"), 3.0, 1, tmp_path / "thermal.msh")
    th = thermal.write_deck(tmp_path / "thermal.msh", tmp_path, CFG)
    d = np.load(tmp_path / "th.npz")
    T0 = 80.0
    np.save(tmp_path / "NT.npy", np.full((2, len(d["P"])), T0))
    np.save(tmp_path / "times.npy", np.array([0.0, 100.0]))
    warp.write_deck(tmp_path / "warp.msh", tmp_path, tmp_path, CFG, 50)
    ccx.run("me_50", tmp_path)
    res = warp.postprocess(tmp_path / "warp.msh", tmp_path, CFG, 50)
    assert res["thermal_shrink_after_eject_pct"] == pytest.approx(1.4e-4 * (T0 - 23.0) * 100, rel=0.01)
    assert res["warp_max_mm"] < 1e-3


@pytest.mark.solver
def test_tabular_properties_match_constant_ones(tmp_path):
    """A flat table takes the adaptive path and must give the same cooling as the constant value."""
    t = 4.0
    cfg = {**CFG, "material": {**MAT, "thermal": dict(MAT["thermal"], conductivity=[[0, 0.22], [300, 0.22]], specific_heat=[[0, 1900], [300, 1900]])}}
    make_box(tmp_path / "p.step", 60, t, 60)
    geometry.volume_mesh(str(tmp_path / "p.step"), 1.2, 1, tmp_path / "thermal.msh")
    thermal.write_deck(tmp_path / "thermal.msh", tmp_path, cfg)
    ccx.run("th", tmp_path)
    d = np.load(tmp_path / "th.npz")
    R = ccx.read_frd(tmp_path / "th.frd")
    NT = np.array([v[np.argsort(i), 0] for _, i, v in R])
    times = np.array([r[0] for r in R])
    centre = int(np.argmin(np.linalg.norm(d["P"] - [30, t / 2, 30], axis=1)))
    assert thermal.crossing_times(NT, times, 120.0)[centre] == pytest.approx(analytic_slab_time(120.0, t / 2, CFG), rel=0.15)
