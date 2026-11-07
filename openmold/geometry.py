"""STEP import and meshing with gmsh."""
import numpy as np, gmsh


def _open(step):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(step)
    gmsh.model.occ.synchronize()
    vols = gmsh.model.getEntities(3)
    if len(vols) != 1:
        gmsh.finalize()
        raise ValueError(f"{step}: expected exactly one solid, found {len(vols)}. Export a single closed solid.")
    return vols


def volume_mesh(step, size, order, out):
    """Tet mesh. Returns the node count."""
    vols = _open(step)
    try:
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1 if order == 2 else 0)
        gmsh.model.addPhysicalGroup(3, [e[1] for e in vols], 1)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(out))
        return len(gmsh.model.mesh.getNodes()[0])
    finally:
        gmsh.finalize()


def surface_mesh(step, size):
    """Conforming, curvature-refined surface triangulation of the CAD solid. Returns points and triangles."""
    _open(step)
    try:
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size * 0.1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        # keep fine edge sizes on small fillets from spreading into large flat faces
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.model.mesh.generate(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, _, conn = gmsh.model.mesh.getElements(2)
        lookup = np.zeros(int(tags.max()) + 1, np.int64)
        lookup[tags.astype(np.int64)] = np.arange(len(tags))
        return xyz.reshape(-1, 3), lookup[conn[0].astype(np.int64)].reshape(-1, 3)
    finally:
        gmsh.finalize()
