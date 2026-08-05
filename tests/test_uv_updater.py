import importlib.util
import json
import pathlib
import tempfile
import tomllib
import zipfile

import bpy


ROOT = pathlib.Path(__file__).resolve().parents[1]
ADDON_PATH = ROOT / "__init__.py"

spec = importlib.util.spec_from_file_location(
    "uv_updater_test",
    ADDON_PATH,
)
addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


assert addon._semantic_version("v1.5.0") == (1, 5, 0)
assert addon._normalized_version("1.05.0") == "1.5.0"
try:
    addon._semantic_version("1.5")
except addon._UpdateError:
    pass
else:
    raise AssertionError("Invalid version was accepted")

latest_payload = {
    "tag_name": "v1.6.0",
    "html_url": "https://github.com/XIV-gate/uv-batch-knife/releases/tag/v1.6.0",
    "assets": [
        {
            "name": "uv_batch_knife-1.6.0.zip",
            "url": (
                "https://api.github.com/repos/XIV-gate/uv-batch-knife/"
                "releases/assets/123456"
            ),
            "browser_download_url": (
                "https://github.com/XIV-gate/uv-batch-knife/releases/"
                "download/v1.6.0/uv_batch_knife-1.6.0.zip"
            ),
        },
    ],
}
update = addon._release_update_info(latest_payload)
assert update["is_newer"]
assert update["latest_version"] == "1.6.0"
assert update["asset_name"] == "uv_batch_knife-1.6.0.zip"
authenticated_update = addon._release_update_info(
    latest_payload,
    authenticated=True,
)
assert authenticated_update["asset_url"].startswith(
    "https://api.github.com/"
)

current_payload = {
    "tag_name": "v1.5.0",
    "assets": [],
}
assert not addon._release_update_info(current_payload)["is_newer"]
assert addon._trusted_update_url("https://api.github.com/test", api=True)
assert addon._trusted_update_url("https://release-assets.githubusercontent.com/test")
assert not addon._trusted_update_url("http://github.com/test")
assert not addon._trusted_update_url("https://github.com.example.org/test")


with tempfile.TemporaryDirectory(prefix="uv_batch_knife_test_") as directory:
    archive_path = pathlib.Path(directory) / "update.zip"
    manifest_text = """\
schema_version = "1.0.0"
id = "uv_batch_knife"
version = "1.6.0"
name = "UV Batch Knife"
tagline = "Test package"
maintainer = "XIVgate"
type = "add-on"
blender_version_min = "5.0.0"
license = ["SPDX:GPL-3.0-or-later"]
"""
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("blender_manifest.toml", manifest_text)
        archive.writestr("__init__.py", "")
    manifest = addon._validate_update_archive(archive_path, "1.6.0")
    assert manifest["id"] == "uv_batch_knife"
    assert manifest["version"] == "1.6.0"


manifest = tomllib.loads((ROOT / "blender_manifest.toml").read_text("utf-8"))
assert manifest["version"] == addon._ADDON_VERSION
assert manifest["permissions"]["network"]
assert manifest["permissions"]["files"]

addon.register()
try:
    assert hasattr(bpy.ops.uv, "batch_knife_update")
    assert hasattr(
        bpy.context.window_manager,
        "uv_batch_knife_update_status",
    )
    if not bpy.app.online_access:
        update_result = bpy.ops.uv.batch_knife_update()
        assert update_result == {"CANCELLED"}
        assert "Online access" in (
            bpy.context.window_manager.uv_batch_knife_update_status
        )
finally:
    addon.unregister()

summary = {
    "addon_version": addon._ADDON_VERSION,
    "latest_update": update,
    "online_access": bpy.app.online_access,
    "online_access_override": bpy.app.online_access_override,
}
print("UV_UPDATER_TEST_RESULT=" + json.dumps(summary, sort_keys=True))
