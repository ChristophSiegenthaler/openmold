"""Writes two simple open boxes as STEP files: demo_solid.step and demo_hollow.step. Usage: make_demo_parts.py [dir]"""
import sys
from pathlib import Path
import gmsh

X, Y, Z, WALL = 60.0, 30.0, 40.0, 2.5


def box(hollow):
    occ = gmsh.model.occ
    outer = occ.addBox(-X / 2, 0, -Z / 2, X, Y, Z)
    inner = occ.addBox(-X / 2 + WALL, WALL, -Z / 2 + WALL, X - 2 * WALL, Y, Z - 2 * WALL)
    shell = occ.cut([(3, outer)], [(3, inner)])[0]
    columns = [(3, occ.addCylinder(sx * 25, 0, sz * 15, 0, Y, 0, 4.0)) for sx in (-1, 1) for sz in (-1, 1)]
    body = occ.fuse(shell, columns)[0]
    if hollow:
        holes = [(3, occ.addCylinder(sx * 25, WALL, sz * 15, 0, Y, 0, 2.4)) for sx in (-1, 1) for sz in (-1, 1)]
        body = occ.cut(body, holes)[0]
    occ.synchronize()
    return body


def main(target="."):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, hollow in (("demo_solid", False), ("demo_hollow", True)):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        box(hollow)
        gmsh.write(str(target / f"{name}.step"))
        gmsh.finalize()


if __name__ == "__main__":
    main(*sys.argv[1:])
