import importlib.util
import json
import pathlib

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location("uv_batch_knife_test", ADDON_PATH)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


def build_two_stacked_strips():
    mesh = bpy.data.meshes.new("UVBatchKnifeTestMesh")
    obj = bpy.data.objects.new("UVBatchKnifeTest", mesh)
    bpy.context.scene.collection.objects.link(obj)

    vertices = []
    faces = []
    face_uvs = []

    # Two disconnected 1x2 quad strips in 3D. Their UVs are exactly stacked.
    for strip_index, x_offset in enumerate((0.0, 3.0)):
        base = len(vertices)
        vertices.extend(
            (
                (x_offset + 0.0, 0.0, 0.0),
                (x_offset + 1.0, 0.0, 0.0),
                (x_offset + 0.0, 1.0, 0.0),
                (x_offset + 1.0, 1.0, 0.0),
                (x_offset + 0.0, 2.0, 0.0),
                (x_offset + 1.0, 2.0, 0.0),
            )
        )
        faces.extend(
            (
                (base + 0, base + 1, base + 3, base + 2),
                (base + 2, base + 3, base + 5, base + 4),
            )
        )
        face_uvs.extend(
            (
                ((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)),
                ((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)),
            )
        )

    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon, polygon_uvs in zip(mesh.polygons, face_uvs):
        for loop_index, uv in zip(polygon.loop_indices, polygon_uvs):
            uv_layer.data[loop_index].uv = uv

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    return obj


def uv_island_count(bm, uv_layer):
    faces = list(bm.faces)
    remaining = set(faces)
    islands = 0

    while remaining:
        islands += 1
        seed = remaining.pop()
        stack = [seed]
        while stack:
            face = stack.pop()
            for edge in face.edges:
                loop_a = addon._loop_for_edge(face, edge)
                if loop_a is None:
                    continue
                map_a = addon._edge_uvs_by_vertex(face, edge, uv_layer)
                for neighbor in edge.link_faces:
                    if neighbor is face or neighbor not in remaining:
                        continue
                    map_b = addon._edge_uvs_by_vertex(neighbor, edge, uv_layer)
                    if map_b is None:
                        continue
                    if all(
                        addon._dist_sq(map_a[vert], map_b[vert]) < 1.0e-16
                        for vert in edge.verts
                    ):
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return islands


obj = build_two_stacked_strips()
result = addon._cut_object(
    obj,
    origin=(0.5, -1.0),
    direction=(0.0, 3.0),
    target_mode="VISIBLE",
    extend_line=True,
    split_uv_islands=True,
    separation=0.00002,
    mark_seams=True,
    sync_selection=False,
)

bm = bmesh.from_edit_mesh(obj.data)
uv_layer = bm.loops.layers.uv.active
summary = {
    "operator_result": result,
    "verts": len(bm.verts),
    "edges": len(bm.edges),
    "faces": len(bm.faces),
    "seams": sum(1 for edge in bm.edges if edge.seam),
    "uv_islands": uv_island_count(bm, uv_layer),
}

assert result["cut_edges"] == 4, summary
assert result["new_vertices"] == 6, summary
assert len(bm.verts) == 18, summary
assert len(bm.faces) == 8, summary
assert summary["seams"] == 4, summary
assert summary["uv_islands"] == 4, summary

print("UV_BATCH_KNIFE_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
