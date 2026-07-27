import time
from collections import deque

import bmesh
import bpy
import gpu
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty
from gpu_extras.batch import batch_for_shader


_UV_EPS = 1.0e-7
_FACTOR_EPS = 1.0e-6


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub2(a, b):
    return a[0] - b[0], a[1] - b[1]


def _dist_sq(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return x * x + y * y


def _signed_distance(point, origin, direction):
    return _cross2(direction, _sub2(point, origin))


def _point_in_polygon(point, polygon):
    """Return True for a point strictly inside a simple UV polygon."""
    inside = False
    px, py = point
    count = len(polygon)
    for index in range(count):
        ax, ay = polygon[index]
        bx, by = polygon[(index + 1) % count]

        edge_x = bx - ax
        edge_y = by - ay
        rel_x = px - ax
        rel_y = py - ay
        cross = edge_x * rel_y - edge_y * rel_x
        if abs(cross) <= _UV_EPS:
            dot = rel_x * edge_x + rel_y * edge_y
            edge_len_sq = edge_x * edge_x + edge_y * edge_y
            if -_UV_EPS <= dot <= edge_len_sq + _UV_EPS:
                return True

        if (ay > py) != (by > py):
            x_at_y = ax + (py - ay) * edge_x / (by - ay)
            if px < x_at_y:
                inside = not inside
    return inside


def _line_edge_intersection(origin, direction, edge_a, edge_b, extend_line):
    edge_direction = _sub2(edge_b, edge_a)
    denominator = _cross2(direction, edge_direction)
    if abs(denominator) <= _UV_EPS:
        return None

    offset = _sub2(edge_a, origin)
    line_factor = _cross2(offset, edge_direction) / denominator
    edge_factor = _cross2(offset, direction) / denominator

    if not extend_line and not (-_UV_EPS <= line_factor <= 1.0 + _UV_EPS):
        return None
    if not (-_UV_EPS <= edge_factor <= 1.0 + _UV_EPS):
        return None

    edge_factor = min(1.0, max(0.0, edge_factor))
    line_factor = min(1.0, max(0.0, line_factor)) if not extend_line else line_factor
    point = (
        edge_a[0] + edge_direction[0] * edge_factor,
        edge_a[1] + edge_direction[1] * edge_factor,
    )
    return line_factor, edge_factor, point


def _face_uv_centroid(face, uv_layer):
    total_x = 0.0
    total_y = 0.0
    count = 0
    for loop in face.loops:
        uv = loop[uv_layer].uv
        total_x += uv.x
        total_y += uv.y
        count += 1
    if not count:
        return 0.0, 0.0
    return total_x / count, total_y / count


def _loop_for_edge(face, edge):
    for loop in face.loops:
        if loop.edge is edge:
            return loop
    return None


def _edge_uvs_by_vertex(face, edge, uv_layer):
    loop = _loop_for_edge(face, edge)
    if loop is None:
        return None
    next_loop = loop.link_loop_next
    return {
        loop.vert: (loop[uv_layer].uv.x, loop[uv_layer].uv.y),
        next_loop.vert: (next_loop[uv_layer].uv.x, next_loop[uv_layer].uv.y),
    }


def _uv_continuous_across(face_a, face_b, edge, uv_layer):
    map_a = _edge_uvs_by_vertex(face_a, edge, uv_layer)
    map_b = _edge_uvs_by_vertex(face_b, edge, uv_layer)
    if map_a is None or map_b is None:
        return False
    tolerance_sq = _UV_EPS * _UV_EPS
    for vert in edge.verts:
        if vert not in map_a or vert not in map_b:
            return False
        if _dist_sq(map_a[vert], map_b[vert]) > tolerance_sq:
            return False
    return True


def _selected_mesh_face(face):
    return not face.hide and face.select


def _visible_uv_face(face, sync_selection):
    if face.hide:
        return False
    return True if sync_selection else face.select


def _edit_mesh_objects(context):
    objects = getattr(context, "objects_in_mode_unique_data", None)
    if objects is None:
        active = context.edit_object
        objects = [active] if active is not None else []
    return [obj for obj in objects if obj is not None and obj.type == "MESH"]


def _collect_face_cut(
    face,
    uv_layer,
    origin,
    direction,
    extend_line,
    edge_cut_buckets,
):
    loops = list(face.loops)
    polygon = [(loop[uv_layer].uv.x, loop[uv_layer].uv.y) for loop in loops]
    signed = [_signed_distance(point, origin, direction) for point in polygon]
    if min(signed) > _UV_EPS or max(signed) < -_UV_EPS:
        return None

    hits = []
    count = len(loops)
    for index, loop in enumerate(loops):
        uv_a = polygon[index]
        uv_b = polygon[(index + 1) % count]
        if _dist_sq(uv_a, uv_b) <= _UV_EPS * _UV_EPS:
            continue

        intersection = _line_edge_intersection(
            origin,
            direction,
            uv_a,
            uv_b,
            extend_line,
        )
        if intersection is None:
            continue

        line_factor, edge_factor, point = intersection
        next_loop = loop.link_loop_next
        descriptor = {
            "line_factor": line_factor,
            "point": point,
            "edge": None,
            "edge_factor": None,
            "vert": None,
        }

        if edge_factor <= _UV_EPS:
            descriptor["vert"] = loop.vert
        elif edge_factor >= 1.0 - _UV_EPS:
            descriptor["vert"] = next_loop.vert
        else:
            edge = loop.edge
            descriptor["edge"] = edge
            if loop.vert is edge.verts[0]:
                descriptor["edge_factor"] = edge_factor
            else:
                descriptor["edge_factor"] = 1.0 - edge_factor
        hits.append(descriptor)

    if len(hits) < 2:
        return None

    hits.sort(key=lambda item: item["line_factor"])
    unique_hits = []
    tolerance_sq = _UV_EPS * _UV_EPS
    for hit in hits:
        if unique_hits and _dist_sq(hit["point"], unique_hits[-1]["point"]) <= tolerance_sq:
            if unique_hits[-1]["vert"] is None and hit["vert"] is not None:
                unique_hits[-1] = hit
            continue
        unique_hits.append(hit)

    if len(unique_hits) < 2:
        return None

    pairs = []
    used_descriptors = set()
    for first, second in zip(unique_hits, unique_hits[1:]):
        if second["line_factor"] - first["line_factor"] <= _UV_EPS:
            continue
        midpoint = (
            (first["point"][0] + second["point"][0]) * 0.5,
            (first["point"][1] + second["point"][1]) * 0.5,
        )
        if _point_in_polygon(midpoint, polygon):
            pairs.append((first, second))
            used_descriptors.add(id(first))
            used_descriptors.add(id(second))

    if not pairs:
        return None

    for descriptor in unique_hits:
        if id(descriptor) not in used_descriptors or descriptor["edge"] is None:
            continue
        edge_cut_buckets.setdefault(descriptor["edge"], []).append(descriptor)

    return {"face": face, "pairs": pairs}


def _split_requested_edges(edge_cut_buckets):
    created_vertices = 0
    for edge, descriptors in edge_cut_buckets.items():
        if not edge.is_valid:
            continue

        descriptors.sort(key=lambda item: item["edge_factor"])
        groups = []
        for descriptor in descriptors:
            factor = descriptor["edge_factor"]
            if groups and abs(factor - groups[-1][0]) <= _FACTOR_EPS:
                groups[-1][1].append(descriptor)
            else:
                groups.append([factor, [descriptor]])

        original_start = edge.verts[0]
        current_edge = edge
        current_start = original_start
        previous_factor = 0.0

        for factor, group in groups:
            if not current_edge.is_valid:
                break
            remaining = 1.0 - previous_factor
            if remaining <= _FACTOR_EPS:
                break
            relative_factor = (factor - previous_factor) / remaining
            relative_factor = min(1.0 - _UV_EPS, max(_UV_EPS, relative_factor))
            _new_edge, new_vert = bmesh.utils.edge_split(
                current_edge,
                current_start,
                relative_factor,
            )
            new_vert.select = True
            for descriptor in group:
                descriptor["vert"] = new_vert
            current_start = new_vert
            previous_factor = factor
            created_vertices += 1

    return created_vertices


def _connect_face_pairs(
    face_records,
    uv_layer,
    all_target_faces,
    mark_seams,
):
    cut_edges = set()
    cut_faces = set()

    for record in face_records:
        original_face = record["face"]
        if not original_face.is_valid:
            continue
        descendants = {original_face}

        for first, second in record["pairs"]:
            vert_a = first["vert"]
            vert_b = second["vert"]
            if vert_a is None or vert_b is None or vert_a is vert_b:
                continue
            if not vert_a.is_valid or not vert_b.is_valid:
                continue

            candidate = None
            for face in tuple(descendants):
                if not face.is_valid:
                    descendants.discard(face)
                    continue
                if vert_a in face.verts and vert_b in face.verts:
                    candidate = face
                    break
            if candidate is None:
                continue

            existing_edge = None
            for edge in candidate.edges:
                if vert_a in edge.verts and vert_b in edge.verts:
                    existing_edge = edge
                    break
            if existing_edge is not None:
                continue

            selected = candidate.select
            result = bmesh.utils.face_split(
                candidate,
                vert_a,
                vert_b,
                use_exist=False,
            )
            if result is None:
                continue
            new_face, new_loop = result
            if new_face is None or new_loop is None:
                continue

            candidate.select = selected
            new_face.select = selected
            descendants.add(new_face)
            all_target_faces.add(new_face)
            cut_faces.add(candidate)
            cut_faces.add(new_face)

            cut_edge = new_loop.edge
            cut_edge.select = True
            vert_a.select = True
            vert_b.select = True
            if mark_seams:
                cut_edge.seam = True
            cut_edges.add(cut_edge)

    return cut_edges, cut_faces


def _positive_cut_side_faces(
    all_target_faces,
    cut_edges,
    uv_layer,
    origin,
    direction,
):
    if not cut_edges:
        return set()

    valid_target_faces = {face for face in all_target_faces if face.is_valid}
    sign_cache = {}

    def face_sign(face):
        cached = sign_cache.get(face)
        if cached is not None:
            return cached
        value = _signed_distance(_face_uv_centroid(face, uv_layer), origin, direction)
        sign_cache[face] = value
        return value

    seeds = set()
    for edge in cut_edges:
        for face in edge.link_faces:
            if face in valid_target_faces and face_sign(face) > _UV_EPS:
                seeds.add(face)

    result = set(seeds)
    queue = deque(seeds)
    while queue:
        face = queue.popleft()
        for edge in face.edges:
            if edge in cut_edges or edge.seam:
                continue
            for neighbor in edge.link_faces:
                if neighbor is face or neighbor in result:
                    continue
                if neighbor not in valid_target_faces:
                    continue
                if face_sign(neighbor) < -_UV_EPS:
                    continue
                if not _uv_continuous_across(face, neighbor, edge, uv_layer):
                    continue
                result.add(neighbor)
                queue.append(neighbor)
    return result


def _shift_uv_faces(faces, uv_layer, direction, separation):
    length = (direction[0] * direction[0] + direction[1] * direction[1]) ** 0.5
    if length <= _UV_EPS or separation <= 0.0:
        return
    shift_x = -direction[1] / length * separation
    shift_y = direction[0] / length * separation
    for face in faces:
        if not face.is_valid:
            continue
        for loop in face.loops:
            loop[uv_layer].uv.x += shift_x
            loop[uv_layer].uv.y += shift_y


def _cut_object(
    obj,
    origin,
    direction,
    *,
    target_mode,
    extend_line,
    split_uv_islands,
    separation,
    mark_seams,
    sync_selection,
):
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    active_uv = mesh.uv_layers.active
    if active_uv is None:
        return {
            "object": obj.name,
            "target_faces": 0,
            "cut_faces": 0,
            "cut_edges": 0,
            "new_vertices": 0,
        }

    uv_layer = bm.loops.layers.uv.get(active_uv.name)
    if uv_layer is None:
        return {
            "object": obj.name,
            "target_faces": 0,
            "cut_faces": 0,
            "cut_edges": 0,
            "new_vertices": 0,
        }

    if target_mode == "SELECTED_UV":
        target_faces = [
            face
            for face in bm.faces
            if _selected_mesh_face(face)
        ]
    else:
        target_faces = [
            face
            for face in bm.faces
            if _visible_uv_face(face, sync_selection)
        ]

    all_target_faces = set(target_faces)
    edge_cut_buckets = {}
    face_records = []
    for face in target_faces:
        if not face.is_valid or len(face.loops) < 3:
            continue
        record = _collect_face_cut(
            face,
            uv_layer,
            origin,
            direction,
            extend_line,
            edge_cut_buckets,
        )
        if record is not None:
            face_records.append(record)

    new_vertices = _split_requested_edges(edge_cut_buckets)
    cut_edges, cut_faces = _connect_face_pairs(
        face_records,
        uv_layer,
        all_target_faces,
        mark_seams,
    )

    if split_uv_islands and cut_edges and separation > 0.0:
        positive_faces = _positive_cut_side_faces(
            all_target_faces,
            cut_edges,
            uv_layer,
            origin,
            direction,
        )
        _shift_uv_faces(positive_faces, uv_layer, direction, separation)

    if cut_edges:
        bm.select_flush_mode()
        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)

    return {
        "object": obj.name,
        "target_faces": len(target_faces),
        "cut_faces": len(cut_faces),
        "cut_edges": len(cut_edges),
        "new_vertices": new_vertices,
    }


def _extended_preview_points(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1.0:
        return start, end
    dx /= length
    dy /= length
    extension = 10000.0
    return (
        start[0] - dx * extension,
        start[1] - dy * extension,
    ), (
        start[0] + dx * extension,
        start[1] + dy * extension,
    )


def _draw_batch_knife(operator):
    context = bpy.context
    if (
        context.area is None
        or context.area.as_pointer() != operator._area_pointer
    ):
        return

    start = operator._pixel_start
    end = operator._pixel_end
    if start is None:
        return

    if end is None:
        end = start
    line_start, line_end = (
        _extended_preview_points(start, end)
        if operator.extend_line
        else (start, end)
    )

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(2.0)
    line_batch = batch_for_shader(
        shader,
        "LINES",
        {"pos": (line_start, line_end)},
    )
    shader.bind()
    shader.uniform_float("color", (1.0, 0.25, 0.05, 0.95))
    line_batch.draw(shader)

    gpu.state.point_size_set(7.0)
    point_positions = [start]
    if operator._stage == 1:
        point_positions.append(end)
    point_batch = batch_for_shader(
        shader,
        "POINTS",
        {"pos": point_positions},
    )
    shader.uniform_float("color", (1.0, 0.7, 0.1, 1.0))
    point_batch.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)
    gpu.state.blend_set("NONE")


class UV_OT_batch_knife(bpy.types.Operator):
    bl_idname = "uv.batch_knife"
    bl_label = "UV Batch Knife"
    bl_description = (
        "Cut every intersected UV face with one UV-space line, creating real "
        "mesh vertices and edges"
    )
    bl_options = {"REGISTER", "UNDO"}

    target_mode: EnumProperty(
        name="Target Faces",
        description="Choose which UV faces can be cut",
        items=(
            (
                "VISIBLE",
                "Visible in UV Editor",
                "Cut faces visible in the UV Editor",
            ),
            (
                "SELECTED_UV",
                "Selected Mesh Faces",
                "Cut only selected mesh faces",
            ),
        ),
        default="VISIBLE",
    )
    extend_line: BoolProperty(
        name="Extend Line",
        description="Treat the drawn segment as an infinite line",
        default=True,
    )
    split_uv_islands: BoolProperty(
        name="Split UV Islands",
        description=(
            "Move one cut side by a tiny amount so the two sides become "
            "separate UV islands"
        ),
        default=True,
    )
    separation: FloatProperty(
        name="Separation",
        description="UV offset used to separate the two cut sides",
        default=0.00002,
        min=0.0,
        soft_max=0.01,
        precision=6,
    )
    mark_seams: BoolProperty(
        name="Mark New Edges as Seams",
        description="Mark the newly created cut edges as UV seams",
        default=True,
    )
    start_uv: FloatVectorProperty(
        name="Start UV",
        size=2,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    end_uv: FloatVectorProperty(
        name="End UV",
        size=2,
        options={"HIDDEN", "SKIP_SAVE"},
    )

    _draw_handle = None
    _area = None
    _area_pointer = 0
    _pixel_start = None
    _pixel_end = None
    _stage = 0

    @classmethod
    def poll(cls, context):
        area = context.area
        return (
            area is not None
            and area.type == "IMAGE_EDITOR"
            and area.ui_type == "UV"
            and context.region is not None
            and context.region.type == "WINDOW"
            and context.mode == "EDIT_MESH"
            and bool(_edit_mesh_objects(context))
        )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "target_mode")
        layout.prop(self, "extend_line")
        layout.separator()
        layout.prop(self, "split_uv_islands")
        sub = layout.column()
        sub.enabled = self.split_uv_islands
        sub.prop(self, "separation")
        layout.prop(self, "mark_seams")

    def _finish_modal(self, context):
        if self._draw_handle is not None:
            bpy.types.SpaceImageEditor.draw_handler_remove(
                self._draw_handle,
                "WINDOW",
            )
            self._draw_handle = None
        if context.area is not None:
            context.area.tag_redraw()
        if context.workspace is not None:
            context.workspace.status_text_set(None)

    def invoke(self, context, event):
        self._area = context.area
        self._area_pointer = context.area.as_pointer()
        self._pixel_start = None
        self._pixel_end = (event.mouse_region_x, event.mouse_region_y)
        self._stage = 0
        self._draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw_batch_knife,
            (self,),
            "WINDOW",
            "POST_PIXEL",
        )
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "UV Batch Knife: ЛКМ — первая и вторая точки; ПКМ/Esc — отмена"
        )
        context.area.tag_redraw()
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if (
            context.area is None
            or context.area.as_pointer() != self._area_pointer
        ):
            self._finish_modal(context)
            return {"CANCELLED"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._finish_modal(context)
            return {"CANCELLED"}

        if event.type == "MOUSEMOVE":
            self._pixel_end = (event.mouse_region_x, event.mouse_region_y)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            point = (event.mouse_region_x, event.mouse_region_y)
            if self._stage == 0:
                self._pixel_start = point
                self._pixel_end = point
                self._stage = 1
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}

            if _dist_sq(point, self._pixel_start) < 16.0:
                return {"RUNNING_MODAL"}

            self._pixel_end = point
            start_uv = context.region.view2d.region_to_view(*self._pixel_start)
            end_uv = context.region.view2d.region_to_view(*self._pixel_end)
            self.start_uv = start_uv
            self.end_uv = end_uv
            self._finish_modal(context)
            return self.execute(context)

        return {"RUNNING_MODAL"}

    def execute(self, context):
        origin = (float(self.start_uv[0]), float(self.start_uv[1]))
        end = (float(self.end_uv[0]), float(self.end_uv[1]))
        direction = _sub2(end, origin)
        if direction[0] * direction[0] + direction[1] * direction[1] <= _UV_EPS:
            self.report({"WARNING"}, "Линия UV Batch Knife слишком короткая")
            return {"CANCELLED"}

        started = time.perf_counter()
        sync_selection = context.scene.tool_settings.use_uv_select_sync
        results = []
        for obj in _edit_mesh_objects(context):
            results.append(
                _cut_object(
                    obj,
                    origin,
                    direction,
                    target_mode=self.target_mode,
                    extend_line=self.extend_line,
                    split_uv_islands=self.split_uv_islands,
                    separation=self.separation,
                    mark_seams=self.mark_seams,
                    sync_selection=sync_selection,
                )
            )

        cut_edges = sum(item["cut_edges"] for item in results)
        cut_faces = sum(item["cut_faces"] for item in results)
        new_vertices = sum(item["new_vertices"] for item in results)
        target_faces = sum(item["target_faces"] for item in results)
        elapsed = time.perf_counter() - started

        if cut_edges == 0:
            self.report(
                {"WARNING"},
                (
                    f"Линия не пересекла подходящие UV-грани "
                    f"(проверено: {target_faces})"
                ),
            )
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            (
                f"UV Batch Knife: рёбер {cut_edges}, вершин {new_vertices}, "
                f"затронуто граней {cut_faces}; {elapsed:.2f} с"
            ),
        )
        return {"FINISHED"}


class IMAGE_PT_uv_batch_knife(bpy.types.Panel):
    bl_label = "UV Batch Knife"
    bl_idname = "IMAGE_PT_uv_batch_knife"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "UV"

    @classmethod
    def poll(cls, context):
        area = context.area
        return (
            area is not None
            and area.ui_type == "UV"
            and context.mode == "EDIT_MESH"
        )

    def draw(self, context):
        column = self.layout.column(align=True)
        column.operator(UV_OT_batch_knife.bl_idname, text="Start UV Batch Knife")
        column.label(text="Shortcut: K")


def _menu_uv_batch_knife(self, context):
    self.layout.separator()
    self.layout.operator(UV_OT_batch_knife.bl_idname)


_classes = (
    UV_OT_batch_knife,
    IMAGE_PT_uv_batch_knife,
)
_keymaps = []


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.IMAGE_MT_uvs.append(_menu_uv_batch_knife)

    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is not None:
        keymap = keyconfig.keymaps.new(
            name="UV Editor",
            space_type="EMPTY",
            region_type="WINDOW",
        )
        item = keymap.keymap_items.new(
            UV_OT_batch_knife.bl_idname,
            "K",
            "PRESS",
        )
        _keymaps.append((keymap, item))


def unregister():
    for keymap, item in _keymaps:
        keymap.keymap_items.remove(item)
    _keymaps.clear()

    bpy.types.IMAGE_MT_uvs.remove(_menu_uv_batch_knife)

    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
