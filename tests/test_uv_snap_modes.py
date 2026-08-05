import importlib.util
import json
import pathlib
from types import MethodType, SimpleNamespace

import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location("uv_snap_modes_test", ADDON_PATH)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


mesh = bpy.data.meshes.new("UVSnapModesTestMesh")
obj = bpy.data.objects.new("UVSnapModesTest", mesh)
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
for loop_index, uv in enumerate(
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
):
    uv_layer.data[loop_index].uv = uv

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")


class FakeView2D:
    @staticmethod
    def view_to_region(u, v, clip=False):
        return u * 100.0 + 500.0, v * 100.0 + 500.0

    @staticmethod
    def region_to_view(x, y):
        return (x - 500.0) / 100.0, (y - 500.0) / 100.0


region = SimpleNamespace(
    width=1000,
    height=1000,
    view2d=FakeView2D(),
)
uv_editor = SimpleNamespace(
    grid_shape_source="FIXED",
    custom_grid_subdivisions=(10, 20),
)
context = SimpleNamespace(
    region=region,
    scene=bpy.context.scene,
    objects_in_mode_unique_data=(obj,),
    edit_object=obj,
    space_data=SimpleNamespace(uv_editor=uv_editor, image=None),
)

operator = SimpleNamespace(
    target_mode="VISIBLE",
    cut_mode="LINE",
    line_mode="MULTI",
    _snap_caches={},
    _snap_tree=None,
    _snap_points=None,
    _snap_mode="OFF",
    _current_grid_snap_step=None,
    _pixel_start=None,
    _axis_lock=None,
    _active_axis=None,
    _snap_active=False,
    _point_snap_hit=False,
    _current_snap_uv=None,
    _path_pixel_points=[],
    _path_uv_points=[],
)
for method_name in (
    "_knife_mode",
    "_constrained_point",
    "_uv_grid_snap_steps",
    "_snap_mode_label",
    "_build_snap_cache",
    "_update_pointer",
):
    setattr(
        operator,
        method_name,
        MethodType(
            getattr(addon.UV_OT_batch_knife, method_name),
            operator,
        ),
    )

cache_results = {}
for snap_mode in ("POINTS", "EDGE_CENTERS", "FACE_CENTERS"):
    operator._snap_mode = snap_mode
    operator._build_snap_cache(context)
    cache_results[snap_mode] = sorted(
        tuple(round(value, 6) for value in uv)
        for _pixel, uv in operator._snap_points
    )

assert cache_results["POINTS"] == [
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (1.0, 1.0),
], cache_results
assert cache_results["EDGE_CENTERS"] == [
    (0.0, 0.5),
    (0.5, 0.0),
    (0.5, 1.0),
    (1.0, 0.5),
], cache_results
assert cache_results["FACE_CENTERS"] == [(0.5, 0.5)], cache_results

operator._snap_mode = "POINTS"
operator._build_snap_cache(context)
operator._path_uv_points = [(0.25, 0.25), (0.8, 0.8)]
operator._path_pixel_points = [
    region.view2d.view_to_region(*uv)
    for uv in operator._path_uv_points
]
own_point_result = operator._update_pointer(
    context,
    region.view2d.view_to_region(0.28, 0.27),
)
own_point_snapped_uv = tuple(operator._current_snap_uv)
assert own_point_result == operator._path_pixel_points[0]
assert own_point_snapped_uv == operator._path_uv_points[0]
assert operator._point_snap_hit

operator._snap_mode = "UV_GRID"
grid_steps = operator._uv_grid_snap_steps(context)
input_pixel = region.view2d.view_to_region(0.137, 0.263)
operator._update_pointer(context, input_pixel)
grid_snapped_uv = tuple(
    round(value, 6)
    for value in operator._current_snap_uv
)

assert grid_steps == (0.1, 0.05), grid_steps
assert grid_snapped_uv == (0.1, 0.25), grid_snapped_uv
assert addon._SNAP_MODES == (
    "OFF",
    "POINTS",
    "UV_GRID",
    "EDGE_CENTERS",
    "FACE_CENTERS",
)

summary = {
    "candidate_counts": {
        mode: len(values)
        for mode, values in cache_results.items()
    },
    "grid_steps": grid_steps,
    "grid_snapped_uv": grid_snapped_uv,
    "own_point_snapped_uv": own_point_snapped_uv,
    "snap_modes": addon._SNAP_MODES,
}
print("UV_SNAP_MODES_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
