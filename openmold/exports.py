"""Deformed geometry export: the fine CAD surface moved by the C3D10 displacement field."""
import subprocess, tempfile
from pathlib import Path
import numpy as np, meshio, trimesh, matplotlib
from scipy.spatial import cKDTree
from .common import similarity_fit
from .warp import fmt_t

_STEP_SCRIPT = """
import Mesh, Part, sys
src, dst = {src!r}, {dst!r}
m = Mesh.Mesh(src)
shape = Part.Shape()
shape.makeShapeFromMesh(m.Topology, 0.001)
solid = Part.makeSolid(Part.makeShell(shape.Faces))
solid.exportStep(dst)
"""


def interpolate_displacement(X, P, U, T10, candidates=24, batch=100000):
    """Quadratic interpolation of U at points X. Works in batches so that large surfaces stay within memory."""
    C = P[T10[:, :4]]
    tree = cKDTree(C.mean(1))
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]   # Abaqus/CalculiX tet10 mid-node order
    out = np.zeros_like(X)
    for start in range(0, len(X), batch):
        x = X[start:start + batch]
        _, cand = tree.query(x, k=candidates)
        cand = cand.reshape(len(x), -1)   # k=1 returns a flat array
        best = np.full(len(x), -1)
        bmin = np.full(len(x), -9.0)
        BL = np.zeros((len(x), 4))
        for j in range(cand.shape[1]):
            e = cand[:, j]
            c = C[e]
            A = np.stack([c[:, 1] - c[:, 0], c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]], axis=2)
            l = np.linalg.solve(A, (x - c[:, 0])[..., None])[..., 0]
            L = np.column_stack([1 - l.sum(1), l])
            mn = L.min(1)
            better = mn > bmin
            best[better], bmin[better], BL[better] = e[better], mn[better], L[better]
        L = np.clip(BL, 0, None)
        L /= L.sum(1, keepdims=True)
        N = np.column_stack([L[:, i] * (2 * L[:, i] - 1) for i in range(4)] + [4 * L[:, a] * L[:, b] for a, b in edges])
        out[start:start + batch] = (N[:, :, None] * U[T10[best]]).sum(1)
    return out


def export_part(pid, part_dir, out_dir, ecfg, t_eject, tools, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surf = np.load(f"{part_dir}/surface.npz")
    X, F = surf["P"], surf["F"]
    w = np.load(f"{part_dir}/warp_{fmt_t(t_eject)}.npz")
    T10 = meshio.read(f"{part_dir}/warp.msh").cells_dict["tetra10"]
    Q = X + interpolate_displacement(X, w["P"], w["U"], T10)
    # as-molded: rotation and translation removed, shrinkage kept. warp-only: shrinkage removed as well.
    R, s, cx, cq = similarity_fit(X, Q)
    as_molded = (Q - cq) @ R + cx
    warp_only = (Q - cq) @ R / s + cx
    base = f"{pid}_warped_{fmt_t(t_eject)}s"
    written = []
    for tag, V in (("as-molded", as_molded), ("warp-only", warp_only)):
        for k in ecfg["scales"]:
            mesh = trimesh.Trimesh(X + k * (V - X), F, process=False)
            trimesh.repair.fix_normals(mesh)
            if not mesh.is_watertight:
                raise RuntimeError(f"{pid}: exported surface is not watertight")
            name = f"{base}_{tag}" + ("" if k == 1 else f"_x{k:g}")
            mesh.export(out_dir / f"{name}.stl")
            written.append(f"{name}.stl")
            if ecfg["step"] and k == 1:
                written.append(_to_step(out_dir / f"{name}.stl", tools, log))
    dev = np.linalg.norm(warp_only - X, axis=1)
    if ecfg["coloured_ply"]:
        colours = (matplotlib.colormaps["turbo"](dev / max(dev.max(), 1e-9)) * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(warp_only, F, vertex_colors=colours, process=False)
        trimesh.repair.fix_normals(mesh)
        mesh.export(out_dir / f"{base}_warp-only_coloured.ply")
        written.append(f"{base}_warp-only_coloured.ply")
    log(f"  {pid}: exported {len(written)} files, max distortion on the surface {dev.max():.3f} mm")
    return dict(files=written, distortion_max_mm=float(dev.max()), scale_factor=float(s))


def _to_step(stl, tools, log):
    dst = stl.with_suffix(".step")
    log(f"  writing faceted STEP {dst.name}; this takes minutes and the file gets large")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_STEP_SCRIPT.format(src=str(stl), dst=str(dst)))
    subprocess.run([tools["freecad"], "-c", f"exec(open({f.name!r}).read())"], capture_output=True)
    Path(f.name).unlink()
    if not dst.is_file():
        raise RuntimeError(f"FreeCAD did not write {dst}. Is '{tools['freecad']}' installed?")
    return dst.name
