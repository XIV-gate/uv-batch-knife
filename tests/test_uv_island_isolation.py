import importlib.util
import json
import pathlib

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_island_isolation_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


VERTICES = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (2.0, 0.0, 0.0),
    (2.0, 1.0, 0.0),
)
FACES = ((0, 1, 2, 3), (1, 4, 5, 2))
LEFT_UVS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
SEPARATE_RIGHT_UVS = (
    (2.0, 0.0),
    (3.0, 0.0),
    (3.0, 1.0),
    (2.0, 1.0),
)
CONTINUOUS_RIGHT_UVS = (
    (1.0, 0.0),
    (2.0, 0.0),
    (2.0, 1.0),
    (1.0, 1.0),
)


def _new_adjacent_quads(name, right_uvs, select_right=False):
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.from_pydata(VERTICES, (), FACES)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop_index, uv in enumerate(LEFT_UVS + right_uvs):
        uv_layer.data[loop_index].uv = uv

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    for face in bm.faces:
        face.select = False
    bm.faces[0].select = True
    bm.faces[1].select = select_right
    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    return obj, mesh


def _face_groups(bm, uv_layer):
    left_faces = {
        face
        for face in bm.faces
        if face.is_valid
        and addon._face_uv_centroid(face, uv_layer)[0] < 1.5
    }
    right_faces = {
        face
        for face in bm.faces
        if face.is_valid
        and addon._face_uv_centroid(face, uv_layer)[0] > 1.5
    }
    return left_faces, right_faces


def _connected_face_components(faces):
    remaining = set(faces)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            face = stack.pop()
            for edge in face.edges:
                for neighbor in edge.link_faces:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return components


def _run_separate_island_case(name, isolate):
    obj, mesh = _new_adjacent_quads(name, SEPARATE_RIGHT_UVS)
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    right_face = bm.faces[1]

    result = addon._cut_polyline_object(
        obj,
        ((0.5, 0.0), (1.0, 0.5)),
        target_mode="SELECTED_UV",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
        endpoint_extension_mode="NEAREST_CORNER",
        isolate_uv_islands=isolate,
    )

    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    left_faces, right_faces = _face_groups(bm, uv_layer)
    left_verts = {vert for face in left_faces for vert in face.verts}
    right_verts = {vert for face in right_faces for vert in face.verts}
    left_has_shared_cut_edge = any(
        len([face for face in edge.link_faces if face in left_faces]) == 2
        for edge in bm.edges
        if edge.is_valid
    )
    summary = {
        "operator_result": result,
        "right_face_vertices": len(right_face.verts),
        "left_faces": len(left_faces),
        "right_faces": len(right_faces),
        "groups_share_vertices": bool(left_verts & right_verts),
        "left_has_shared_cut_edge": left_has_shared_cut_edge,
    }
    bpy.ops.object.mode_set(mode="OBJECT")
    return summary


normal = _run_separate_island_case("UVIslandNormal", False)
isolated = _run_separate_island_case("UVIslandDetached", True)

assert normal["operator_result"]["isolated_islands"] == 0, normal
assert normal["right_face_vertices"] == 5, normal
assert normal["groups_share_vertices"], normal

assert isolated["operator_result"]["isolated_islands"] == 1, isolated
assert isolated["operator_result"]["isolation_edges"] == 1, isolated
assert isolated["right_face_vertices"] == 4, isolated
assert not isolated["groups_share_vertices"], isolated
assert isolated["left_has_shared_cut_edge"], isolated


obj, mesh = _new_adjacent_quads(
    "UVIslandInfiniteDetached",
    SEPARATE_RIGHT_UVS,
)
bm = bmesh.from_edit_mesh(mesh)
bm.faces.ensure_lookup_table()
infinite_right_face = bm.faces[1]
infinite_result = addon._cut_object(
    obj,
    origin=(0.5, 0.0),
    direction=(0.5, 0.5),
    target_mode="SELECTED_UV",
    extend_line=True,
    split_uv_islands=False,
    separation=0.0,
    mark_seams=True,
    sync_selection=False,
    isolate_uv_islands=True,
)
infinite_summary = {
    "operator_result": infinite_result,
    "right_face_vertices": len(infinite_right_face.verts),
}
assert infinite_result["isolated_islands"] == 1, infinite_summary
assert infinite_result["isolation_edges"] == 1, infinite_summary
assert infinite_summary["right_face_vertices"] == 4, infinite_summary


obj, mesh = _new_adjacent_quads("UVIslandGridDetached", SEPARATE_RIGHT_UVS)
bm = bmesh.from_edit_mesh(mesh)
bm.faces.ensure_lookup_table()
grid_right_face = bm.faces[1]
grid_result = addon._cut_grid_object(
    obj,
    center=(0.7, 0.5),
    size=0.8,
    angle=0.0,
    subdivisions=2,
    target_mode="SELECTED_UV",
    split_uv_islands=False,
    separation=0.0,
    mark_seams=True,
    sync_selection=False,
    isolate_uv_islands=True,
)
grid_summary = {
    "operator_result": grid_result,
    "right_face_vertices": len(grid_right_face.verts),
}
assert grid_result["isolated_islands"] == 1, grid_summary
assert grid_result["isolation_edges"] == 1, grid_summary
assert grid_summary["right_face_vertices"] == 4, grid_summary


obj, mesh = _new_adjacent_quads(
    "UVIslandContinuous",
    CONTINUOUS_RIGHT_UVS,
    select_right=True,
)
continuous_result = addon._cut_polyline_object(
    obj,
    ((-0.2, 0.5), (2.2, 0.5)),
    target_mode="SELECTED_UV",
    split_uv_islands=False,
    separation=0.0,
    mark_seams=True,
    sync_selection=False,
    endpoint_extension_mode="NEAREST_CORNER",
    isolate_uv_islands=True,
)
bm = bmesh.from_edit_mesh(mesh)
continuous_summary = {
    "operator_result": continuous_result,
    "face_components": _connected_face_components(bm.faces),
}
assert continuous_result["isolated_islands"] == 0, continuous_summary
assert continuous_summary["face_components"] == 1, continuous_summary

print(
    "UV_ISLAND_ISOLATION_TEST_RESULT="
    + json.dumps(
        {
            "normal": normal,
            "isolated": isolated,
            "infinite": infinite_summary,
            "grid": grid_summary,
            "continuous": continuous_summary,
        },
        sort_keys=True,
    )
)
