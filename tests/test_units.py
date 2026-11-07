import math
import numpy as np, pytest, yaml
from openmold.common import basis, boundary, in_box, orient_tets, similarity_fit
from openmold.ccx import read_frd
from openmold.config import ConfigError, load_config
from openmold.draft import analyse
from openmold.fill import gate_nodes, zero_shear_viscosity
from openmold.thermal import crossing_times, zone_mask


def write(tmp_path, **over):
    (tmp_path / "a.step").write_text("x")
    data = dict(project={"name": "t"}, parts=[{"id": "A", "step": "a.step"}], material="generic_tpu_65d", gates=[{"point": [0, 0, 0]}])
    data.update(over)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_config_ok(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg["material"]["density"] == 1173
    assert cfg["process"]["fill_time"] == 2.0
    assert cfg["parts"][0]["label"] == "A"


@pytest.mark.parametrize("over,msg", [
    (dict(gates=[]), "gates"),
    (dict(parts=[{"id": "A", "step": "missing.step"}]), "STEP file not found"),
    (dict(parts=[{"id": "A/B", "step": "a.step"}]), "may only use"),
    (dict(process={"eject_times": [9999]}), "outside the simulated cooling window"),
    (dict(process={"mold_temp": 300}), "below process.melt_temp"),
    (dict(exports={"parts": ["nope"]}), "unknown part"),
    (dict(warpage={}), "unknown config section"),
    (dict(material="no_such_card"), "not found"),
])
def test_config_rejects(tmp_path, over, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(write(tmp_path, **over))


def test_cross_wlf_at_reference_temperature():
    flow = dict(model="cross_wlf", D1=1e5, D2=400.0, A1=30.0, A2=50.0, tau=1e5, n=0.3)
    assert zero_shear_viscosity(flow, 400 - 273.15) == pytest.approx(1e5)
    hotter = zero_shear_viscosity(flow, 500 - 273.15)
    assert hotter == pytest.approx(1e5 * math.exp(-30 * 100 / 150))
    assert zero_shear_viscosity(dict(model="cross", eta0=1500), 225) == 1500


def test_boundary_of_a_single_tet_faces_outward():
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    T = orient_tets(P, np.array([[0, 2, 1, 3]]))
    F = boundary(T)
    assert len(F) == 4
    centre = P.mean(0)
    for tri in F:
        n = np.cross(P[tri[1]] - P[tri[0]], P[tri[2]] - P[tri[0]])
        assert n @ (P[tri].mean(0) - centre) > 0


def test_in_box_treats_null_as_unbounded():
    P = np.array([[0, 0, 0], [0, 10, 0], [5, 10, 0]], float)
    assert in_box(P, {"min": [None, 5, None]}).tolist() == [False, True, True]
    assert in_box(P, {"min": [None, 5, None], "max": [1, None, None]}).tolist() == [False, True, False]


def test_basis_is_orthonormal():
    for axis in ([0, 1, 0], [1, 0, 0], [1, 2, 3]):
        a = np.array(axis, float) / np.linalg.norm(axis)
        e1, e2 = basis(axis)
        assert abs(e1 @ a) < 1e-12 and abs(e2 @ a) < 1e-12 and abs(e1 @ e2) < 1e-12


def test_similarity_fit_recovers_scale_and_rotation():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 3))
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    B = 0.97 * A @ R.T + [1, 2, 3]
    Rf, sf, ca, cb = similarity_fit(A, B)
    assert sf == pytest.approx(0.97)
    assert np.allclose(sf * ((A - ca) @ Rf.T) + cb, B)


def test_draft_of_a_tilted_face():
    # a triangle tilted 3 degrees away from the demould axis
    t = math.tan(math.radians(3))
    P = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 10 * t]], float)
    angle, info = analyse(P, np.array([[0, 1, 2]]), [0, 1, 0], limit_deg=2)
    assert angle[0] == pytest.approx(3, abs=1e-6)
    assert info["below_limit_pct"] == 0
    assert analyse(P, np.array([[0, 1, 2]]), [0, 1, 0], limit_deg=5)[1]["below_limit_pct"] == 100


def test_gate_nodes_falls_back_to_nearest():
    P = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5], [9, 9, 9]], float)
    F = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 4]])
    assert set(gate_nodes(P, F, [{"point": [0, 0, 0], "radius": 0.5}])) == {0, 1, 2, 3}


def test_crossing_times_interpolate():
    times = np.array([0.0, 10.0, 20.0])
    NT = np.array([[100.0], [60.0], [20.0]])
    assert crossing_times(NT, times, 80.0)[0] == pytest.approx(5.0)
    assert np.isinf(crossing_times(NT, times, 10.0)[0])


def test_zone_mask_limits_radius_and_length():
    cen = np.array([[0.5, 5, 0], [0.5, 15, 0], [5, 5, 0], [0.5, 0.5, 0]], float)
    zones = [{"htc": 500, "cylinders": [{"base": [0, 0, 0], "axis": [0, 1, 0], "radius": 1, "start": 1, "length": 10}]}]
    h = zone_mask(cen, zones)
    assert h[0] == 500 and np.isnan(h[1:]).all()


def test_read_frd_parses_blocks(tmp_path):
    def row(i, *vals):
        return " -1" + f"{i:10d}" + "".join(f"{v:12.5E}" for v in vals) + "\n"
    text = ("  100CL  101 1.00000E+00          2                     0    1           1\n"
            " -4  NDTEMP      1    1\n -5  T           1    1    0    0\n" + row(1, 50.0) + row(2, 60.0) + " -3\n"
            "  100CL  102 2.00000E+00          2                     0    2           1\n"
            " -4  DISP        4    1\n -5  D1          1    2    1    0\n" + row(1, 0.1, 0.2, 0.3) + row(2, 0.4, 0.5, 0.6) + " -3\n")
    path = tmp_path / "x.frd"
    path.write_text(text)
    temps = read_frd(path)
    assert len(temps) == 1 and temps[0][0] == 1.0 and temps[0][2][1, 0] == 60.0
    disp = read_frd(path, "DISP")
    assert disp[0][1].tolist() == [1, 2] and disp[0][2][1].tolist() == [0.4, 0.5, 0.6]


def test_report_number_format():
    from openmold.report import _number
    assert _number(None) == "–" and _number(float("inf")) == "not reached"
    assert _number(0.0123) == "0.0123" and _number(1234.5) == "1234"


def _wall_cloud():
    # a 4 mm wall with surface nodes at y=0 and y=4 and a centre row at y=2 that freezes last
    x = np.arange(0.0, 20.0)
    P = np.vstack([np.column_stack([x, np.full(20, y), np.zeros(20)]) for y in (0.0, 2.0, 4.0)])
    surface = np.r_[0:20, 40:60]
    F = np.column_stack([surface, np.roll(surface, 1), np.roll(surface, 2)])
    geo = dict(P=P, th=np.full(60, 4.0), ts=np.full(60, 4.0), sv=surface)
    tnf = np.r_[np.full(20, 1.0), np.full(20, 13.0), np.full(20, 1.0)]
    return P, F, geo, tnf


def test_wall_reference_uses_a_node_inside_the_wall():
    from openmold.thermal import wall_reference
    P, F, geo, tnf = _wall_cloud()
    node, _ = wall_reference(P, F, tnf, geo, None)
    assert P[node, 1] == 2.0 and tnf[node] == 13.0


def test_wall_reference_refuses_a_mesh_without_interior_nodes():
    from openmold.thermal import wall_reference
    P, F, geo, tnf = _wall_cloud()
    keep = P[:, 1] != 2.0
    index = np.cumsum(keep) - 1
    geo = dict(P=P[keep], th=geo["th"][keep], ts=geo["ts"][keep], sv=index[geo["sv"]])
    with pytest.raises(ValueError, match="no nodes inside"):
        wall_reference(P[keep], index[F], tnf[keep], geo, None)


def test_interpolation_reproduces_a_quadratic_field_in_batches():
    from openmold.exports import interpolate_displacement
    corners = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], float)
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    P = np.vstack([corners] + [(corners[a] + corners[b])[None] / 2 for a, b in edges])
    U = np.column_stack([P[:, 0] ** 2, P[:, 1] * P[:, 2], P[:, 2]])
    rng = np.random.default_rng(1)
    L = rng.dirichlet(np.ones(4), 50)
    X = L @ corners
    got = interpolate_displacement(X, P, U, np.arange(10)[None], candidates=1, batch=7)
    assert np.allclose(got, np.column_stack([X[:, 0] ** 2, X[:, 1] * X[:, 2], X[:, 2]]))


def test_solver_threads_default_to_one(tmp_path):
    from openmold.cli import _warnings
    cfg = load_config(write(tmp_path))
    assert cfg["tools"]["ccx_threads"] == 1 and _warnings(cfg) == []
    assert _warnings(load_config(write(tmp_path, tools={"ccx_threads": 4})))
