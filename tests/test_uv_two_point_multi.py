import importlib.util
import json
import pathlib
from types import SimpleNamespace

import bmesh
import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_two_point_multi_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)

assert addon._KNIFE_MODES == ("MULTI", "INFINITE", "GRID")
mode_state = SimpleNamespace(cut_mode="LINE", line_mode="MULTI")
for expected_mode in addon._KNIFE_MODES:
    addon.UV_OT_batch_knife._set_knife_mode(mode_state, expected_mode)
    actual_mode = addon.UV_OT_batch_knife._knife_mode(mode_state)
    assert actual_mode == expected_mode, (expected_mode, actual_mode)

grid_label_state = SimpleNamespace(
    grid_subdivisions=2,
    _grid_preview_size=0.0,
    _grid_preview_angle=0.0,
    _grid_step_active=False,
    _knife_mode_label=lambda: "C3 Grid Knife",
    _snap_mode_label=lambda _context: "S0 OFF",
    _detach_mode_label=lambda: "N0 OFF",
)
assert (
    addon._grid_cursor_label(grid_label_state, None)
    == "C3 Grid Knife | 2 x 2 | S0 OFF | N0 OFF"
)

detach_state = SimpleNamespace(detach_mode="OFF")
assert (
    addon.UV_OT_batch_knife._detach_mode_label(detach_state)
    == "N0 OFF"
)
detach_state.detach_mode = "ISLANDS"
assert (
    addon.UV_OT_batch_knife._detach_mode_label(detach_state)
    == "N1 ISLANDS"
)
detach_state.detach_mode = "POLYGONS"
assert (
    addon.UV_OT_batch_knife._detach_mode_label(detach_state)
    == "N2 POLYGONS"
)

endpoint_state = SimpleNamespace(endpoint_extension_mode="NEAREST_CORNER")
assert (
    addon.UV_OT_batch_knife._endpoint_mode_label(endpoint_state)
    == "E1 CORNER"
)
endpoint_state.endpoint_extension_mode = "NEAREST_EDGE"
assert (
    addon.UV_OT_batch_knife._endpoint_mode_label(endpoint_state)
    == "E2 EDGE"
)

modal_area = SimpleNamespace(
    as_pointer=lambda: 101,
    tag_redraw=lambda: None,
)
modal_scene = SimpleNamespace(
    uv_batch_knife_isolate_uv_islands=False,
    uv_batch_knife_detach_mode="OFF",
    uv_batch_knife_endpoint_extension_mode="NEAREST_CORNER",
)
modal_context = SimpleNamespace(area=modal_area, scene=modal_scene)
modal_state = SimpleNamespace(
    _area_pointer=101,
    isolate_uv_islands=False,
    detach_mode="OFF",
    endpoint_extension_mode="NEAREST_CORNER",
    _set_status_text=lambda _context: None,
)
modal_result = addon.UV_OT_batch_knife.modal(
    modal_state,
    modal_context,
    SimpleNamespace(type="N", value="PRESS"),
)
assert modal_result == {"RUNNING_MODAL"}
assert modal_state.isolate_uv_islands
assert modal_scene.uv_batch_knife_isolate_uv_islands
assert modal_state.detach_mode == "ISLANDS"
assert modal_scene.uv_batch_knife_detach_mode == "ISLANDS"

modal_result = addon.UV_OT_batch_knife.modal(
    modal_state,
    modal_context,
    SimpleNamespace(type="N", value="PRESS"),
)
assert modal_result == {"RUNNING_MODAL"}
assert not modal_state.isolate_uv_islands
assert not modal_scene.uv_batch_knife_isolate_uv_islands
assert modal_state.detach_mode == "POLYGONS"
assert modal_scene.uv_batch_knife_detach_mode == "POLYGONS"

modal_result = addon.UV_OT_batch_knife.modal(
    modal_state,
    modal_context,
    SimpleNamespace(type="N", value="PRESS"),
)
assert modal_result == {"RUNNING_MODAL"}
assert modal_state.detach_mode == "OFF"
assert modal_scene.uv_batch_knife_detach_mode == "OFF"

modal_result = addon.UV_OT_batch_knife.modal(
    modal_state,
    modal_context,
    SimpleNamespace(type="E", value="PRESS"),
)
assert modal_result == {"RUNNING_MODAL"}
assert modal_state.endpoint_extension_mode == "NEAREST_EDGE"
assert modal_scene.uv_batch_knife_endpoint_extension_mode == "NEAREST_EDGE"

addon.register()
try:
    operator_rna = bpy.ops.uv.batch_knife.get_rna_type()
    endpoint_property = operator_rna.properties["endpoint_extension_mode"]
    endpoint_modes = tuple(
        item.identifier for item in endpoint_property.enum_items
    )
    assert endpoint_modes == ("NEAREST_CORNER", "NEAREST_EDGE"), endpoint_modes
    assert endpoint_property.default == "NEAREST_CORNER"
    assert not operator_rna.properties["isolate_uv_islands"].default
    detach_property = operator_rna.properties["detach_mode"]
    assert tuple(
        item.identifier for item in detach_property.enum_items
    ) == ("OFF", "ISLANDS", "POLYGONS")
    assert detach_property.default == "OFF"
    assert abs(
        operator_rna.properties["separation"].default
        - addon._MIN_UV_ISLAND_INSET
    ) < 1.0e-10
    assert (
        bpy.context.scene.uv_batch_knife_endpoint_extension_mode
        == "NEAREST_CORNER"
    )
    assert not bpy.context.scene.uv_batch_knife_isolate_uv_islands
    assert bpy.context.scene.uv_batch_knife_detach_mode == "OFF"
    assert bpy.context.scene.uv_batch_knife_mode == "MULTI"
    assert bpy.context.scene.uv_batch_knife_snap_mode == "OFF"
    assert bpy.context.scene.uv_batch_knife_target_mode == "VISIBLE"
    assert bpy.context.scene.uv_batch_knife_split_uv_islands
    assert abs(
        bpy.context.scene.uv_batch_knife_separation
        - addon._MIN_UV_ISLAND_INSET
    ) < 1.0e-10
    assert bpy.context.scene.uv_batch_knife_mark_seams
    assert bpy.context.scene.uv_batch_knife_grid_size == 0.5
    assert bpy.context.scene.uv_batch_knife_grid_angle == 0.0
    assert bpy.context.scene.uv_batch_knife_grid_subdivisions == 2

    scene = bpy.context.scene
    scene.uv_batch_knife_mode = "GRID"
    scene.uv_batch_knife_snap_mode = "FACE_CENTERS"
    scene.uv_batch_knife_target_mode = "SELECTED_UV"
    scene.uv_batch_knife_endpoint_extension_mode = "NEAREST_EDGE"
    scene.uv_batch_knife_detach_mode = "POLYGONS"
    scene.uv_batch_knife_split_uv_islands = False
    scene.uv_batch_knife_separation = 0.00125
    scene.uv_batch_knife_mark_seams = False
    scene.uv_batch_knife_grid_size = 0.75
    scene.uv_batch_knife_grid_angle = 0.25
    scene.uv_batch_knife_grid_subdivisions = 7

    remembered = SimpleNamespace()
    remembered._set_knife_mode = lambda mode: (
        setattr(remembered, "cut_mode", "GRID")
        if mode == "GRID"
        else (
            setattr(remembered, "cut_mode", "LINE"),
            setattr(remembered, "line_mode", mode),
        )
    )
    addon.UV_OT_batch_knife._load_last_settings(
        remembered,
        SimpleNamespace(scene=scene),
    )
    assert remembered.cut_mode == "GRID"
    assert remembered._snap_mode == "FACE_CENTERS"
    assert remembered.target_mode == "SELECTED_UV"
    assert remembered.endpoint_extension_mode == "NEAREST_EDGE"
    assert remembered.detach_mode == "POLYGONS"
    assert not remembered.split_uv_islands
    assert abs(remembered.separation - 0.00125) < 1.0e-9
    assert not remembered.mark_seams
    assert abs(remembered.grid_size - 0.75) < 1.0e-9
    assert abs(remembered.grid_angle - 0.25) < 1.0e-9
    assert remembered.grid_subdivisions == 7

    remembered.cut_mode = "LINE"
    remembered.line_mode = "INFINITE"
    remembered._knife_mode = lambda: (
        "GRID" if remembered.cut_mode == "GRID" else remembered.line_mode
    )
    remembered._snap_mode = "UV_GRID"
    remembered.target_mode = "VISIBLE"
    remembered.endpoint_extension_mode = "NEAREST_CORNER"
    remembered.detach_mode = "ISLANDS"
    remembered.split_uv_islands = True
    remembered.separation = 0.0
    remembered.mark_seams = True
    remembered.grid_size = 1.25
    remembered.grid_angle = -0.5
    remembered.grid_subdivisions = 9
    addon.UV_OT_batch_knife._store_last_settings(
        remembered,
        SimpleNamespace(scene=scene),
    )
    assert scene.uv_batch_knife_mode == "INFINITE"
    assert scene.uv_batch_knife_snap_mode == "UV_GRID"
    assert scene.uv_batch_knife_target_mode == "VISIBLE"
    assert scene.uv_batch_knife_endpoint_extension_mode == "NEAREST_CORNER"
    assert scene.uv_batch_knife_detach_mode == "ISLANDS"
    assert scene.uv_batch_knife_isolate_uv_islands
    assert scene.uv_batch_knife_split_uv_islands
    assert scene.uv_batch_knife_separation == 0.0
    assert scene.uv_batch_knife_mark_seams
    assert abs(scene.uv_batch_knife_grid_size - 1.25) < 1.0e-9
    assert abs(scene.uv_batch_knife_grid_angle + 0.5) < 1.0e-9
    assert scene.uv_batch_knife_grid_subdivisions == 9
finally:
    addon.unregister()


mesh = bpy.data.meshes.new("UVTwoPointMultiTestMesh")
obj = bpy.data.objects.new("UVTwoPointMultiTest", mesh)
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

print("UV_TWO_POINT_MULTI_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
