import importlib.util
import json
import math
import pathlib

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_angle_inset_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


# A one-cell-thick square ring. Its UV island is concave around a real hole,
# so a centroid-based shrink would point the inner corners the wrong way.
coordinates = tuple(
    (float(x), float(y), 0.0)
    for y in range(4)
    for x in range(4)
)


def vertex_index(x, y):
    return y * 4 + x


faces = []
for y in range(3):
    for x in range(3):
        if (x, y) == (1, 1):
            continue
        faces.append(
            (
                vertex_index(x, y),
                vertex_index(x + 1, y),
                vertex_index(x + 1, y + 1),
                vertex_index(x, y + 1),
            )
        )

mesh = bpy.data.meshes.new("UVAngleInsetRingMesh")
obj = bpy.data.objects.new("UVAngleInsetRing", mesh)
bpy.context.scene.collection.objects.link(obj)
mesh.from_pydata(coordinates, (), faces)
uv_map = mesh.uv_layers.new(name="UVMap")
for loop in mesh.loops:
    coordinate = coordinates[loop.vertex_index]
    uv_map.data[loop.index].uv = coordinate[:2]

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")

bm = bmesh.from_edit_mesh(mesh)
bm.verts.ensure_lookup_table()
uv_layer = bm.loops.layers.uv.active
before = {
    vert.index: (float(vert.co.x), float(vert.co.y))
    for vert in bm.verts
}

moved_nodes = addon._inset_uv_face_components(
    [set(bm.faces)],
    set(),
    uv_layer,
    separation=0.0,
)


def uv_at(coordinate):
    vert = next(
        vert
        for vert in bm.verts
        if abs(vert.co.x - coordinate[0]) < 1.0e-8
        and abs(vert.co.y - coordinate[1]) < 1.0e-8
    )
    values = {
        (
            float(loop[uv_layer].uv.x),
            float(loop[uv_layer].uv.y),
        )
        for loop in vert.link_loops
    }
    assert len(values) == 1, (coordinate, values)
    return next(iter(values))


inset = addon._MIN_UV_ISLAND_INSET
diagonal = inset / math.sqrt(2.0)
tolerance = inset * 0.15

outer_corner = uv_at((0.0, 0.0))
inner_hole_corner = uv_at((1.0, 1.0))
straight_outer_edge = uv_at((1.0, 0.0))

assert math.dist(outer_corner, (diagonal, diagonal)) <= tolerance
assert math.dist(
    inner_hole_corner,
    (1.0 - diagonal, 1.0 - diagonal),
) <= tolerance
assert math.dist(straight_outer_edge, (1.0, inset)) <= tolerance

displacements = []
for vert in bm.verts:
    current = uv_at(before[vert.index])
    displacements.append(math.dist(before[vert.index], current))

assert moved_nodes == 16, moved_nodes
assert min(displacements) >= inset * 0.85, displacements
assert max(displacements) <= inset * 1.15, displacements

summary = {
    "moved_nodes": moved_nodes,
    "outer_corner": outer_corner,
    "inner_hole_corner": inner_hole_corner,
    "straight_outer_edge": straight_outer_edge,
    "minimum_displacement": min(displacements),
    "maximum_displacement": max(displacements),
}
print("UV_ANGLE_INSET_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
