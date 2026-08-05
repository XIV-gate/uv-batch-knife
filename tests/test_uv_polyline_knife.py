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


CLICKED_POINTS = (
    (0.5, 0.5),
    (0.25, 0.75),
    (0.9, 0.6),
)
QUAD_UVS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
)
INSET_TOLERANCE = addon._MIN_UV_ISLAND_INSET * 1.5


def run_case(name, endpoint_extension_mode):
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
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        (),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop_index, uv in enumerate(QUAD_UVS + QUAD_UVS):
        uv_layer.data[loop_index].uv = uv

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")

    result = addon._cut_polyline_object(
        obj,
        CLICKED_POINTS,
        target_mode="VISIBLE",
        split_uv_islands=True,
        separation=0.0,
        mark_seams=True,
        sync_selection=False,
        endpoint_extension_mode=endpoint_extension_mode,
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
    all_uvs = [
        (float(loop[uv].uv.x), float(loop[uv].uv.y))
        for face in bm.faces
        for loop in face.loops
    ]

    def nearest_distance(point):
        return min(
            ((candidate[0] - point[0]) ** 2
             + (candidate[1] - point[1]) ** 2) ** 0.5
            for candidate in all_uvs
        )

    clicked_offsets = [
        nearest_distance(point) for point in CLICKED_POINTS
    ]
    summary = {
        "operator_result": result,
        "verts": len(bm.verts),
        "edges": len(bm.edges),
        "faces": len(bm.faces),
        "seams": len(seam_edges),
        "discontinuous_seams": discontinuous_seams,
        "clicked_uv_offsets": clicked_offsets,
        "edge_extension_points": {
            "start": nearest_distance((0.5, 0.0)) <= INSET_TOLERANCE,
            "end": nearest_distance((1.0, 0.6)) <= INSET_TOLERANCE,
        },
    }
    bpy.ops.object.mode_set(mode="OBJECT")
    return summary


corner_summary = run_case("UVPolylineCornerTest", "NEAREST_CORNER")
edge_summary = run_case("UVPolylineEdgeTest", "NEAREST_EDGE")

assert addon._nearest_visible_polygon_corner(
    CLICKED_POINTS[0],
    QUAD_UVS,
) == (0.0, 0.0)
assert addon._nearest_visible_polygon_corner(
    CLICKED_POINTS[-1],
    QUAD_UVS,
) == (1.0, 1.0)

for summary in (corner_summary, edge_summary):
    assert summary["operator_result"]["cut_edges"] == 8, summary
    assert summary["faces"] == 4, summary
    assert summary["seams"] == 8, summary
    assert summary["discontinuous_seams"] == 8, summary
    assert all(
        addon._UV_EPS < offset <= INSET_TOLERANCE
        for offset in summary["clicked_uv_offsets"]
    ), summary

assert corner_summary["operator_result"]["new_vertices"] == 6, corner_summary
assert not any(corner_summary["edge_extension_points"].values()), corner_summary
assert edge_summary["operator_result"]["new_vertices"] == 10, edge_summary
assert all(edge_summary["edge_extension_points"].values()), edge_summary

print(
    "UV_POLYLINE_KNIFE_TEST_RESULT="
    + json.dumps(
        {
            "nearest_corner": corner_summary,
            "nearest_edge": edge_summary,
        },
        sort_keys=True,
    )
)
