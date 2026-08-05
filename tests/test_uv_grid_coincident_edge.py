import importlib.util
import json
import pathlib

import bmesh
import bpy


root = pathlib.Path(__file__).resolve().parents[1]
addon_path = root / "__init__.py"
spec = importlib.util.spec_from_file_location(
    "uv_grid_coincident_edge_test",
    addon_path,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


# Two adjacent quads share the UV edge from (0, 1) to (1, 1). The grid's
# vertical internal line splits that edge first. Its following horizontal line
# then exactly overlaps the now multi-segment boundary and must be a no-op.
mesh = bpy.data.meshes.new("UVGridCoincidentEdgeMesh")
obj = bpy.data.objects.new("UVGridCoincidentEdge", mesh)
bpy.context.scene.collection.objects.link(obj)
vertices = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
)
mesh.from_pydata(
    vertices,
    (),
    (
        (0, 1, 3, 2),
        (2, 3, 5, 4),
    ),
)
uv_layer = mesh.uv_layers.new(name="UVMap")
for loop in mesh.loops:
    coordinate = vertices[loop.vertex_index]
    uv_layer.data[loop.index].uv = coordinate[:2]

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")

result = addon._cut_grid_object(
    obj,
    center=(0.5, 1.0),
    size=2.0,
    angle=0.0,
    subdivisions=2,
    target_mode="VISIBLE",
    split_uv_islands=False,
    separation=0.00002,
    mark_seams=True,
    sync_selection=False,
)

bm = bmesh.from_edit_mesh(mesh)
areas = sorted(face.calc_area() for face in bm.faces)
summary = {
    "operator_result": result,
    "verts": len(bm.verts),
    "edges": len(bm.edges),
    "faces": len(bm.faces),
    "areas": areas,
}

assert len(bm.faces) == 4, summary
assert min(areas) > 1.0e-10, summary
assert result["cut_edges"] == 2, summary

print("UV_GRID_COINCIDENT_EDGE_TEST_RESULT=" + json.dumps(summary, sort_keys=True))


# A face can already contain intermediate collinear vertices on its boundary
# (for example after preceding grid lines split a shared cube edge). A line
# coincident with that boundary must not connect the outer vertices across the
# existing edge chain, because that creates a zero-area face.
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.select_all(action="DESELECT")
chain_vertices = (
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
)
chain_mesh = bpy.data.meshes.new("UVGridBoundaryChainMesh")
chain_obj = bpy.data.objects.new("UVGridBoundaryChain", chain_mesh)
bpy.context.scene.collection.objects.link(chain_obj)
chain_mesh.from_pydata(chain_vertices, (), ((0, 1, 2, 3, 4),))
chain_uv_layer = chain_mesh.uv_layers.new(name="UVMap")
for loop in chain_mesh.loops:
    coordinate = chain_vertices[loop.vertex_index]
    chain_uv_layer.data[loop.index].uv = coordinate[:2]

bpy.context.view_layer.objects.active = chain_obj
chain_obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
chain_bm = bmesh.from_edit_mesh(chain_mesh)
chain_uv = chain_bm.loops.layers.uv.active
chain_result = addon._cut_face_set_with_line(
    chain_bm,
    chain_uv,
    set(chain_bm.faces),
    origin=(0.0, 5.0e-7),
    direction=(1.0, 0.0),
    extend_line=True,
)
chain_areas = sorted(face.calc_area() for face in chain_bm.faces)
chain_summary = {
    "cut_edges": len(chain_result["cut_edges"]),
    "new_vertices": chain_result["new_vertices"],
    "faces": len(chain_bm.faces),
    "areas": chain_areas,
}
assert chain_summary["cut_edges"] == 0, chain_summary
assert chain_summary["new_vertices"] == 0, chain_summary
assert chain_summary["faces"] == 1, chain_summary
assert min(chain_areas) > 1.0e-10, chain_summary

print(
    "UV_GRID_BOUNDARY_CHAIN_TEST_RESULT="
    + json.dumps(chain_summary, sort_keys=True)
)


# Regression case matching a cube cross: the 5 x 10 visible cells span two
# stacked UV faces and the center horizontal grid line is their exact 3D edge.
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.select_all(action="DESELECT")
cube_vertices = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
)
cube_faces = (
    (0, 3, 2, 1),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
    (4, 5, 6, 7),
)
cube_uvs = (
    (1.0, 0.0), (1.0, -1.0), (2.0, -1.0), (2.0, 0.0),
    (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0),
    (3.0, 1.0), (3.0, 2.0), (2.0, 2.0), (2.0, 1.0),
    (2.0, 3.0), (1.0, 3.0), (1.0, 2.0), (2.0, 2.0),
    (0.0, 2.0), (0.0, 1.0), (1.0, 1.0), (1.0, 2.0),
    (1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0),
)
cube_mesh = bpy.data.meshes.new("UVGridCoincidentCubeMesh")
cube_obj = bpy.data.objects.new("UVGridCoincidentCube", cube_mesh)
bpy.context.scene.collection.objects.link(cube_obj)
cube_mesh.from_pydata(cube_vertices, (), cube_faces)
cube_uv_layer = cube_mesh.uv_layers.new(name="UVMap")
for loop_index, uv in enumerate(cube_uvs):
    cube_uv_layer.data[loop_index].uv = uv

bpy.context.view_layer.objects.active = cube_obj
cube_obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
cube_result = addon._cut_grid_object(
    cube_obj,
    center=(1.5, 0.0),
    size=2.0,
    angle=0.0,
    subdivisions=10,
    target_mode="VISIBLE",
    split_uv_islands=False,
    separation=0.00002,
    mark_seams=True,
    sync_selection=False,
)
cube_bm = bmesh.from_edit_mesh(cube_mesh)
cube_areas = sorted(face.calc_area() for face in cube_bm.faces)
cube_summary = {
    "operator_result": cube_result,
    "verts": len(cube_bm.verts),
    "edges": len(cube_bm.edges),
    "faces": len(cube_bm.faces),
    "minimum_area": min(cube_areas),
    "zero_area_faces": sum(area <= 1.0e-10 for area in cube_areas),
}
assert cube_summary["zero_area_faces"] == 0, cube_summary

print(
    "UV_GRID_COINCIDENT_CUBE_TEST_RESULT="
    + json.dumps(cube_summary, sort_keys=True)
)
