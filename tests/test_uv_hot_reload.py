import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile

import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_FILES = ("__init__.py", "README.md", "blender_manifest.toml")


def load_temporary_package(parent, module_name):
    package_root = parent / module_name
    package_root.mkdir()
    for name in PACKAGE_FILES:
        shutil.copy2(ROOT / name, package_root / name)
    sys.path.insert(0, str(parent))
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, package_root


def write_new_version(package_root, old_version, new_version):
    init_path = package_root / "__init__.py"
    source = init_path.read_text(encoding="utf-8")
    source = source.replace(
        f'_ADDON_VERSION = "{old_version}"',
        f'_ADDON_VERSION = "{new_version}"',
        1,
    )
    if f'_ADDON_VERSION = "{new_version}"' not in source:
        raise AssertionError("Could not create the hot-reload test version")
    init_path.write_text(source, encoding="utf-8")

    manifest_path = package_root / "blender_manifest.toml"
    manifest = manifest_path.read_text(encoding="utf-8").replace(
        f'version = "{old_version}"',
        f'version = "{new_version}"',
        1,
    )
    manifest_path.write_text(manifest, encoding="utf-8")


def cleanup_module(module_name):
    module = sys.modules.get(module_name)
    if module is not None:
        try:
            module.unregister()
        except Exception:
            pass
        sys.modules.pop(module_name, None)


results = {}
with tempfile.TemporaryDirectory(prefix="uv_batch_knife_reload_test_") as directory:
    temporary_parent = pathlib.Path(directory)

    module_name = "uv_batch_knife_hot_reload_success"
    module, package_root = load_temporary_package(
        temporary_parent,
        module_name,
    )
    try:
        old_version = module._ADDON_VERSION
        new_version = "1.5.6"
        module.register()
        snapshot = module._capture_extension_snapshot(package_root)
        write_new_version(package_root, old_version, new_version)
        callback = module._make_extension_reload_callback(
            module_name,
            new_version,
            snapshot,
            package_root,
        )

        module._active_batch_knife_modals = 1
        deferred = callback()
        assert deferred == 0.25, deferred
        assert sys.modules[module_name]._ADDON_VERSION == old_version

        module._active_batch_knife_modals = 0
        finished = callback()
        reloaded = sys.modules[module_name]
        status = bpy.context.window_manager.uv_batch_knife_update_status
        assert finished is None
        assert reloaded._ADDON_VERSION == new_version
        assert hasattr(bpy.ops.uv, "batch_knife")
        assert "loaded in this Blender session" in status, status
        results["success"] = {
            "old_version": old_version,
            "new_version": reloaded._ADDON_VERSION,
            "deferred_while_modal": deferred,
            "status": status,
        }
    finally:
        cleanup_module(module_name)

    module_name = "uv_batch_knife_hot_reload_rollback"
    module, package_root = load_temporary_package(
        temporary_parent,
        module_name,
    )
    try:
        old_version = module._ADDON_VERSION
        module.register()
        snapshot = module._capture_extension_snapshot(package_root)
        write_new_version(package_root, old_version, "1.5.6")
        callback = module._make_extension_reload_callback(
            module_name,
            "9.9.9",
            snapshot,
            package_root,
        )
        finished = callback()
        restored = sys.modules[module_name]
        status = bpy.context.window_manager.uv_batch_knife_update_status
        assert finished is None
        assert restored._ADDON_VERSION == old_version
        assert hasattr(bpy.ops.uv, "batch_knife")
        assert f"restored {old_version}" in status, status
        results["rollback"] = {
            "restored_version": restored._ADDON_VERSION,
            "status": status,
        }
    finally:
        cleanup_module(module_name)

print("UV_HOT_RELOAD_TEST_RESULT=" + json.dumps(results, sort_keys=True))
