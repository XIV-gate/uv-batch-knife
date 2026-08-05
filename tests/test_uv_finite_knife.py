import importlib.util
import json
import pathlib

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_finite_knife_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


mesh = bpy.data.meshes.new("UVFiniteKnifeTestMesh")
obj = bpy.data.objects.new("UVFiniteKnifeTest", mesh)
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
uv_layer = mesh.uv_layers.new(name="UVMap")
quad_uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
for loop_index, uv in enumerate(quad_uvs + quad_uvs):
    uv_layer.data[loop_index].uv = uv

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")

result = addon._cut_polyline_object(
    obj,
    ((-0.2, 0.5), (1.2, 0.5)),
    target_mode="VISIBLE",
    split_uv_islands=True,
    separation=0.00002,
    mark_seams=True,
    sync_selection=False,
)

bm = bmesh.from_edit_mesh(mesh)
summary = {
    "operator_result": result,
    "verts": len(bm.verts),
    "edges": len(bm.edges),
    "faces": len(bm.faces),
    "seams": sum(1 for edge in bm.edges if edge.seam),
}

assert result["cut_edges"] == 2, summary
assert result["new_vertices"] == 4, summary
assert len(bm.faces) == 4, summary
assert summary["seams"] == 2, summary

print("UV_FINITE_KNIFE_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
