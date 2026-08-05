import importlib.util
import json
import pathlib

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_polygon_detach_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


def _leave_edit_mode():
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def _new_quad_strip(name, count, selected_indices):
    _leave_edit_mode()
    bpy.ops.object.select_all(action="DESELECT")

    vertices = []
    for row in (0.0, 1.0):
        vertices.extend((float(column), row, 0.0) for column in range(count + 1))
    faces = []
    row_width = count + 1
    for column in range(count):
        faces.append(
            (
                column,
                column + 1,
                row_width + column + 1,
                row_width + column,
            )
        )

    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.from_pydata(vertices, (), faces)
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
            uv_layer.data[loop_index].uv = (vertex.co.x, vertex.co.y)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    for face in bm.faces:
        face.select = face.index in selected_indices
    bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
    return obj, mesh


def _faces_in_uv_column(bm, uv_layer, column):
    return {
        face
        for face in bm.faces
        if face.is_valid
        and column < addon._face_uv_centroid(face, uv_layer)[0] < column + 1
    }


def _vertex_set(faces):
    return {vertex for face in faces for vertex in face.verts}


def _polyline_one_touched_polygon():
    obj, mesh = _new_quad_strip("UVPolygonDetachOne", 2, {0})
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    untouched_face = bm.faces[1]

    result = addon._cut_polyline_object(
        obj,
        ((-0.2, 0.5), (1.2, 0.5)),
        target_mode="SELECTED_UV",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
        detach_mode="POLYGONS",
    )

    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    left_faces = _faces_in_uv_column(bm, uv_layer, 0)
    right_faces = _faces_in_uv_column(bm, uv_layer, 1)
    summary = {
        "operator_result": result,
        "untouched_face_vertices": len(untouched_face.verts),
        "groups_share_vertices": bool(
            _vertex_set(left_faces) & _vertex_set(right_faces)
        ),
    }
    assert result["isolated_polygons"] == 1, summary
    assert result["isolation_edges"] == 1, summary
    assert result["cut_edges"] == 1, summary
    assert summary["untouched_face_vertices"] == 4, summary
    assert not summary["groups_share_vertices"], summary
    return summary


def _polyline_two_touched_polygons():
    obj, mesh = _new_quad_strip("UVPolygonDetachBoth", 2, {0, 1})
    result = addon._cut_polyline_object(
        obj,
        ((-0.2, 0.5), (2.2, 0.5)),
        target_mode="SELECTED_UV",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
        detach_mode="POLYGONS",
    )

    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    left_faces = _faces_in_uv_column(bm, uv_layer, 0)
    right_faces = _faces_in_uv_column(bm, uv_layer, 1)
    summary = {
        "operator_result": result,
        "groups_share_vertices": bool(
            _vertex_set(left_faces) & _vertex_set(right_faces)
        ),
    }
    assert result["isolated_polygons"] == 2, summary
    assert result["isolation_edges"] == 1, summary
    assert result["cut_edges"] == 2, summary
    assert not summary["groups_share_vertices"], summary
    return summary


def _grid_only_intersected_polygon():
    obj, mesh = _new_quad_strip("UVPolygonDetachGrid", 3, {0, 1, 2})
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    left_face = bm.faces[0]
    right_face = bm.faces[2]

    result = addon._cut_grid_object(
        obj,
        center=(1.5, 0.5),
        size=0.8,
        angle=0.0,
        subdivisions=2,
        target_mode="SELECTED_UV",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
        detach_mode="POLYGONS",
    )

    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    left_faces = _faces_in_uv_column(bm, uv_layer, 0)
    middle_faces = _faces_in_uv_column(bm, uv_layer, 1)
    right_faces = _faces_in_uv_column(bm, uv_layer, 2)
    middle_vertices = _vertex_set(middle_faces)
    summary = {
        "operator_result": result,
        "left_face_vertices": len(left_face.verts),
        "right_face_vertices": len(right_face.verts),
        "middle_shares_left": bool(middle_vertices & _vertex_set(left_faces)),
        "middle_shares_right": bool(middle_vertices & _vertex_set(right_faces)),
    }
    assert result["isolated_polygons"] == 1, summary
    assert result["isolation_edges"] == 2, summary
    assert result["cut_edges"] > 0, summary
    assert summary["left_face_vertices"] == 4, summary
    assert summary["right_face_vertices"] == 4, summary
    assert not summary["middle_shares_left"], summary
    assert not summary["middle_shares_right"], summary
    return summary


summary = {
    "polyline_one": _polyline_one_touched_polygon(),
    "polyline_both": _polyline_two_touched_polygons(),
    "grid_middle": _grid_only_intersected_polygon(),
}
print("UV_POLYGON_DETACH_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
