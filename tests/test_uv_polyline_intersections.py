import importlib.util
import json
import pathlib

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_polyline_intersections_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


QUAD_UVS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
)


def _new_quad(name):
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.from_pydata(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop_index, uv in enumerate(QUAD_UVS):
        uv_layer.data[loop_index].uv = uv

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    return obj, mesh


def _vert_uvs(vert, uv_layer):
    return {
        (round(loop[uv_layer].uv.x, 6), round(loop[uv_layer].uv.y, 6))
        for loop in vert.link_loops
    }


def _crossing_case():
    path = (
        (0.0, 0.2),
        (0.8, 0.8),
        (0.2, 0.8),
        (1.0, 0.2),
    )
    intersection = addon._polyline_segment_intersection(
        path[0], path[1], path[2], path[3]
    )[2]
    intersection_key = tuple(round(value, 6) for value in intersection)

    obj, mesh = _new_quad("UVPolylineCrossing")
    result = addon._cut_polyline_object(
        obj,
        path,
        target_mode="VISIBLE",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
    )
    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    intersection_verts = [
        vert for vert in bm.verts
        if intersection_key in _vert_uvs(vert, uv_layer)
    ]
    seam_degree = (
        sum(1 for edge in intersection_verts[0].link_edges if edge.seam)
        if len(intersection_verts) == 1
        else 0
    )
    summary = {
        "operator_result": result,
        "intersection": intersection_key,
        "intersection_vertices": len(intersection_verts),
        "intersection_seam_degree": seam_degree,
        "faces": len(bm.faces),
    }
    assert result["cut_edges"] == 5, summary
    assert result["new_vertices"] == 5, summary
    assert len(intersection_verts) == 1, summary
    assert seam_degree == 4, summary
    assert len(bm.faces) == 4, summary
    return summary


def _closed_contour_case():
    path = (
        (0.25, 0.25),
        (0.75, 0.25),
        (0.75, 0.75),
        (0.25, 0.75),
        (0.25, 0.25),
    )
    path_points = set(path[:-1])
    corner_points = set(QUAD_UVS)

    obj, mesh = _new_quad("UVPolylineClosedContour")
    result = addon._cut_polyline_object(
        obj,
        path,
        target_mode="VISIBLE",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
    )
    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    support_edges = []
    for edge in bm.edges:
        endpoint_uvs = []
        for vert in edge.verts:
            endpoint_uvs.append(next(iter(_vert_uvs(vert, uv_layer))))
        if (
            not edge.seam
            and any(point in path_points for point in endpoint_uvs)
            and any(point in corner_points for point in endpoint_uvs)
        ):
            support_edges.append(edge)

    center_faces = [
        face
        for face in bm.faces
        if all(
            (round(loop[uv_layer].uv.x, 6), round(loop[uv_layer].uv.y, 6))
            in path_points
            for loop in face.loops
        )
    ]
    summary = {
        "operator_result": result,
        "faces": len(bm.faces),
        "seams": sum(1 for edge in bm.edges if edge.seam),
        "support_edges": len(support_edges),
        "center_faces": len(center_faces),
    }
    assert result["cut_edges"] == 4, summary
    assert result["new_vertices"] == 4, summary
    assert len(bm.faces) == 3, summary
    assert summary["seams"] == 4, summary
    assert len(support_edges) == 2, summary
    assert len(center_faces) == 1, summary
    return summary


def _closed_contour_from_boundary_case():
    path = (
        (0.0, 0.0),
        (0.75, 0.25),
        (0.75, 0.75),
        (0.25, 0.75),
        (0.0, 0.0),
    )
    obj, mesh = _new_quad("UVPolylineClosedFromBoundary")
    result = addon._cut_polyline_object(
        obj,
        path,
        target_mode="VISIBLE",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
    )
    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    corner_verts = [
        vert for vert in bm.verts
        if (0.0, 0.0) in _vert_uvs(vert, uv_layer)
    ]
    summary = {
        "operator_result": result,
        "verts": len(bm.verts),
        "faces": len(bm.faces),
        "corner_vertices": len(corner_verts),
        "seams": sum(1 for edge in bm.edges if edge.seam),
    }
    assert result["cut_edges"] == 4, summary
    assert result["new_vertices"] == 3, summary
    assert len(bm.verts) == 7, summary
    assert len(bm.faces) == 3, summary
    assert len(corner_verts) == 1, summary
    assert summary["seams"] == 4, summary
    return summary


def _snapped_own_point_junction_case():
    junction = (0.4, 0.45)
    path = (
        (0.0, 0.2),
        junction,
        (0.75, 0.75),
        (0.75, 0.3),
        junction,
        (1.0, 0.65),
    )
    obj, mesh = _new_quad("UVPolylineOwnPointJunction")
    result = addon._cut_polyline_object(
        obj,
        path,
        target_mode="VISIBLE",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
    )
    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    junction_verts = [
        vert for vert in bm.verts
        if junction in _vert_uvs(vert, uv_layer)
    ]
    seam_degree = (
        sum(1 for edge in junction_verts[0].link_edges if edge.seam)
        if len(junction_verts) == 1
        else 0
    )
    wire_edges = [edge for edge in bm.edges if not edge.link_faces]
    summary = {
        "operator_result": result,
        "junction_vertices": len(junction_verts),
        "junction_seam_degree": seam_degree,
        "wire_edges": len(wire_edges),
        "faces": len(bm.faces),
    }
    assert len(junction_verts) == 1, summary
    assert seam_degree == 4, summary
    assert not wire_edges, summary
    assert len(bm.faces) == 4, summary
    return summary


def _two_snapped_junctions_case():
    junctions = ((0.4, 0.72), (0.55, 0.34))
    path = (
        (0.0, 0.3),
        junctions[0],
        (0.68, 0.9),
        (0.72, 0.58),
        junctions[0],
        junctions[1],
        (0.82, 0.25),
        (0.7, 0.08),
        junctions[1],
        (1.0, 0.5),
    )
    obj, mesh = _new_quad("UVPolylineTwoOwnPointJunctions")
    result = addon._cut_polyline_object(
        obj,
        path,
        target_mode="VISIBLE",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
    )
    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    junction_data = []
    for junction in junctions:
        vertices = [
            vert for vert in bm.verts
            if junction in _vert_uvs(vert, uv_layer)
        ]
        seam_degree = (
            sum(1 for edge in vertices[0].link_edges if edge.seam)
            if len(vertices) == 1
            else 0
        )
        junction_data.append((len(vertices), seam_degree))
    wire_edges = [edge for edge in bm.edges if not edge.link_faces]
    summary = {
        "operator_result": result,
        "junctions": junction_data,
        "wire_edges": len(wire_edges),
        "faces": len(bm.faces),
    }
    assert junction_data == [(1, 4), (1, 4)], summary
    assert not wire_edges, summary
    assert len(bm.faces) == 6, summary
    return summary


def _retraced_branch_junctions_case():
    junctions = ((0.4, 0.72), (0.55, 0.34))
    path = (
        (0.0, 0.3),
        junctions[0],
        (0.68, 0.9),
        junctions[0],
        junctions[1],
        (0.82, 0.2),
        junctions[1],
        (1.0, 0.5),
    )
    obj, mesh = _new_quad("UVPolylineRetracedBranches")
    result = addon._cut_polyline_object(
        obj,
        path,
        target_mode="VISIBLE",
        split_uv_islands=False,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
    )
    bm = bmesh.from_edit_mesh(mesh)
    uv_layer = bm.loops.layers.uv.active
    junction_data = []
    for junction in junctions:
        vertices = [
            vert for vert in bm.verts
            if junction in _vert_uvs(vert, uv_layer)
        ]
        junction_data.append(
            (
                len(vertices),
                len(vertices[0].link_edges) if len(vertices) == 1 else 0,
            )
        )
    wire_edges = [edge for edge in bm.edges if not edge.link_faces]
    summary = {
        "operator_result": result,
        "junctions": junction_data,
        "wire_edges": len(wire_edges),
        "faces": len(bm.faces),
    }
    assert junction_data == [(1, 3), (1, 3)], summary
    assert not wire_edges, summary
    assert result["cut_edges"] == 5, summary
    assert len(bm.faces) == 4, summary
    return summary


summary = {
    "crossing": _crossing_case(),
    "closed_contour": _closed_contour_case(),
    "closed_from_boundary": _closed_contour_from_boundary_case(),
    "own_point_junction": _snapped_own_point_junction_case(),
    "two_own_point_junctions": _two_snapped_junctions_case(),
    "retraced_branch_junctions": _retraced_branch_junctions_case(),
}
print(
    "UV_POLYLINE_INTERSECTIONS_TEST_RESULT="
    + json.dumps(summary, sort_keys=True)
)
