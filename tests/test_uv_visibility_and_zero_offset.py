import importlib.util
import json
import pathlib

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_visibility_and_zero_offset_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


mesh = bpy.data.meshes.new("UVVisibilityTestMesh")
obj = bpy.data.objects.new("UVVisibilityTest", mesh)
bpy.context.scene.collection.objects.link(obj)
mesh.from_pydata(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
    ),
    (),
    ((0, 1, 2, 3), (4, 5, 6, 7)),
)
uv_map = mesh.uv_layers.new(name="UVMap")
uvs = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (2.0, 0.0),
    (3.0, 0.0),
    (3.0, 1.0),
    (2.0, 1.0),
)
for loop_index, uv in enumerate(uvs):
    uv_map.data[loop_index].uv = uv

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")

bm = bmesh.from_edit_mesh(mesh)
hidden_face = next(
    face
    for face in bm.faces
    if sum(vert.co.x for vert in face.verts) / len(face.verts) > 1.5
)
hidden_face.hide_set(True)
bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)

result = addon._cut_grid_object(
    obj,
    center=(0.5, 0.5),
    size=0.8,
    angle=0.0,
    subdivisions=4,
    target_mode="VISIBLE",
    split_uv_islands=True,
    separation=0.0,
    mark_seams=True,
    sync_selection=False,
)

bm = bmesh.from_edit_mesh(mesh)
uv_layer = bm.loops.layers.uv.get("UVMap")
hidden_faces = [face for face in bm.faces if face.hide]
visible_faces = [face for face in bm.faces if not face.hide]

assert hidden_face.is_valid
assert hidden_faces == [hidden_face], {
    "hidden_faces": len(hidden_faces),
    "total_faces": len(bm.faces),
    "result": result,
}
assert len(visible_faces) > 1, result

expected_coordinates = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
for face in visible_faces:
    for loop in face.loops:
        uv = loop[uv_layer].uv
        assert any(abs(uv.x - value) < 1.0e-6 for value in expected_coordinates), uv.x
        assert any(abs(uv.y - value) < 1.0e-6 for value in expected_coordinates), uv.y

assert all(not face.hide for face in visible_faces)
assert all(
    not edge.hide
    for edge in bm.edges
    if any(not face.hide for face in edge.link_faces)
)
assert all(
    not vert.hide
    for vert in bm.verts
    if any(not face.hide for face in vert.link_faces)
)

summary = {
    "result": result,
    "faces": len(bm.faces),
    "hidden_faces": len(hidden_faces),
    "visible_faces": len(visible_faces),
}
print(
    "UV_VISIBILITY_ZERO_OFFSET_TEST_RESULT="
    + json.dumps(summary, sort_keys=True)
)
