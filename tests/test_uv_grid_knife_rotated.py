import importlib.util
import json
import math
import pathlib

import bmesh
import bpy


root = pathlib.Path(__file__).resolve().parents[1]
addon_path = root / "__init__.py"
spec = importlib.util.spec_from_file_location("uv_grid_rotated_test", addon_path)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)

mesh = bpy.data.meshes.new("UVGridRotatedMesh")
obj = bpy.data.objects.new("UVGridRotated", mesh)
bpy.context.scene.collection.objects.link(obj)
mesh.from_pydata(
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
    (),
    ((0, 1, 2, 3),),
)
uv_layer = mesh.uv_layers.new(name="UVMap")
for loop_index, uv in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
    uv_layer.data[loop_index].uv = uv

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")

angle = math.radians(30.0)
result = addon._cut_grid_object(
    obj,
    center=(0.5, 0.5),
    size=0.6,
    angle=angle,
    subdivisions=3,
    target_mode="VISIBLE",
    split_uv_islands=False,
    separation=0.00002,
    mark_seams=True,
    sync_selection=False,
)

bm = bmesh.from_edit_mesh(mesh)
uv = bm.loops.layers.uv.active
axis_x, axis_y = addon._grid_axes(angle)
inside_faces = [
    face
    for face in bm.faces
    if addon._face_fully_inside_grid(
        face,
        uv,
        center=(0.5, 0.5),
        axis_x=axis_x,
        axis_y=axis_y,
        half_size=0.3,
    )
]
summary = {
    "operator_result": result,
    "inside_faces": len(inside_faces),
    "seams": sum(1 for edge in bm.edges if edge.seam),
}

assert result["cut_edges"] == 24, summary
assert len(inside_faces) == 9, summary
assert summary["seams"] == 24, summary

print("UV_GRID_ROTATED_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
