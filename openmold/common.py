"""Mesh helpers shared by all analysis steps."""
import numpy as np, meshio


def load_tets(path):
    m = meshio.read(path)
    P = m.points.astype(float)
    T = m.cells_dict["tetra"].astype(np.int64)
    return P, orient_tets(P, T)


def orient_tets(P, T):
    v = np.einsum("ij,ij->i", np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), P[T[:, 3]] - P[T[:, 0]])
    T = T.copy()
    T[v < 0] = T[v < 0][:, [0, 2, 1, 3]]
    return T


def boundary(T):
    """Outward-facing boundary triangles of positively oriented tets."""
    F = np.vstack([T[:, [1, 2, 3]], T[:, [0, 3, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 1]]])
    key = np.sort(F, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return F[idx[cnt == 1]]


def tet_volumes(P, T):
    return np.abs(np.einsum("ij,ij->i", np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]), P[T[:, 3]] - P[T[:, 0]])) / 6


def nodal_volumes(P, T):
    vn = np.zeros(len(P))
    np.add.at(vn, T.ravel(), np.repeat(tet_volumes(P, T) / 4, 4))
    return vn


def vertex_normals(P, F):
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    length = np.linalg.norm(N, axis=1)
    length[length == 0] = 1
    return N / length[:, None], fn


def in_box(P, box):
    """Boolean mask of points inside box {min: [x,y,z], max: [x,y,z]}. null means unbounded on that axis."""
    mask = np.ones(len(P), bool)
    for k in range(3):
        lo = (box.get("min") or [None] * 3)[k]
        hi = (box.get("max") or [None] * 3)[k]
        if lo is not None:
            mask &= P[:, k] >= lo
        if hi is not None:
            mask &= P[:, k] <= hi
    return mask


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def basis(axis):
    """Two unit vectors perpendicular to axis, right-handed so that e1 x axis = e2."""
    a = unit(axis)
    ref = np.eye(3)[0] if abs(a[0]) < 0.9 else np.eye(3)[1]
    e1 = unit(ref - ref.dot(a) * a)
    return e1, np.cross(e1, a)


def similarity_fit(A, B):
    """Best-fit rotation, translation and uniform scale of A onto B. Returns R, s, ca, cb with B ~ s*(A-ca)@R.T + cb."""
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = (U @ D @ Vt).T
    s = np.trace(np.diag(S) @ D) / (X ** 2).sum()
    return R, s, ca, cb
