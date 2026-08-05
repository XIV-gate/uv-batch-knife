import importlib.util
import json
import pathlib

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_polyline_knife_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


mesh = bpy.data.meshes.new("UVPolylineKnifeTestMesh")
obj = bpy.data.objects.new("UVPolylineKnifeTest", mesh)
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
for loop_index, uv in enumerate(
    (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
):
    uv_layer.data[loop_index].uv = uv

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")

clicked_points = (
    (0.5, 0.5),
    (0.25, 0.75),
    (0.9, 0.6),
)
result = addon._cut_polyline_object(
    obj,
    clicked_points,
    target_mode="VISIBLE",
    split_uv_islands=True,
    separation=0.00002,
    mark_seams=True,
    sync_selection=False,
)

bm = bmesh.from_edit_mesh(mesh)
uv = bm.loops.layers.uv.active
seam_edges = [edge for edge in bm.edges if edge.seam]
discontinuous_seams = 0
for edge in seam_edges:
    if len(edge.link_faces) != 2:
        continue
    map_a = addon._edge_uvs_by_vertex(edge.link_faces[0], edge, uv)
    map_b = addon._edge_uvs_by_vertex(edge.link_faces[1], edge, uv)
    if any(
        addon._dist_sq(map_a[vert], map_b[vert]) > 1.0e-16
        for vert in edge.verts
    ):
        discontinuous_seams += 1
all_uvs = {
    (round(loop[uv].uv.x, 6), round(loop[uv].uv.y, 6))
    for face in bm.faces
    for loop in face.loops
}
summary = {
    "operator_result": result,
    "verts": len(bm.verts),
    "edges": len(bm.edges),
    "faces": len(bm.faces),
    "seams": len(seam_edges),
    "discontinuous_seams": discontinuous_seams,
    "clicked_uvs_present": all(point in all_uvs for point in clicked_points),
    "boundary_extensions": {
        "start": (0.5, 0.0) in all_uvs,
        "end": (1.0, 0.6) in all_uvs,
    },
}

assert result["cut_edges"] == 8, summary
assert result["new_vertices"] == 10, summary
assert len(bm.faces) == 4, summary
assert summary["seams"] == 8, summary
assert summary["discontinuous_seams"] == 8, summary
assert summary["clicked_uvs_present"], summary
assert all(summary["boundary_extensions"].values()), summary

print("UV_POLYLINE_KNIFE_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
