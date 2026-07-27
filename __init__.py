import math
import time
from collections import deque

import bmesh
import blf
import bpy
import gpu
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
)
from gpu_extras.batch import batch_for_shader
from mathutils.kdtree import KDTree


_UV_EPS = 1.0e-7
_FACTOR_EPS = 1.0e-6
_POINT_SNAP_RADIUS_PX = 18.0
_SNAP_MODES = (
    "OFF",
    "POINTS",
    "UV_GRID",
    "EDGE_CENTERS",
    "FACE_CENTERS",
)
_SNAP_MODE_LABELS = {
    "OFF": "S0 OFF",
    "POINTS": "S1 UV Points",
    "UV_GRID": "S2 UV Grid",
    "EDGE_CENTERS": "S3 Edge Centers",
    "FACE_CENTERS": "S4 Face Centers",
}


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub2(a, b):
    return a[0] - b[0], a[1] - b[1]


def _dist_sq(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return x * x + y * y


def _nice_grid_step(value):
    """Round a positive UV spacing to a readable 1/2/5 decade."""
    value = max(float(value), _UV_EPS)
    exponent = math.floor(math.log10(value))
    scale = 10.0 ** exponent
    normalized = value / scale
    if normalized <= 1.0:
        factor = 1.0
    elif normalized <= 2.0:
        factor = 2.0
    elif normalized <= 5.0:
        factor = 5.0
    else:
        factor = 10.0
    return factor * scale


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


def _grid_axes(angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (cosine, sine), (-sine, cosine)


def _grid_local(point, center, axis_x, axis_y):
    relative = _sub2(point, center)
    return (
        relative[0] * axis_x[0] + relative[1] * axis_x[1],
        relative[0] * axis_y[0] + relative[1] * axis_y[1],
    )


def _grid_world(local, center, axis_x, axis_y):
    return (
        center[0] + axis_x[0] * local[0] + axis_y[0] * local[1],
        center[1] + axis_x[1] * local[0] + axis_y[1] * local[1],
    )


def _orientation(a, b, c):
    return _cross2(_sub2(b, a), _sub2(c, a))


def _segments_intersect(a, b, c, d):
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return (
        min(ab_c, ab_d) <= _UV_EPS
        and max(ab_c, ab_d) >= -_UV_EPS
        and min(cd_a, cd_b) <= _UV_EPS
        and max(cd_a, cd_b) >= -_UV_EPS
    )


def _face_grid_local_polygon(face, uv_layer, center, axis_x, axis_y):
    return [
        _grid_local(
            (loop[uv_layer].uv.x, loop[uv_layer].uv.y),
            center,
            axis_x,
            axis_y,
        )
        for loop in face.loops
    ]


def _face_overlaps_grid(face, uv_layer, center, axis_x, axis_y, half_size):
    polygon = _face_grid_local_polygon(
        face,
        uv_layer,
        center,
        axis_x,
        axis_y,
    )
    if not polygon:
        return False

    minimum_x = min(point[0] for point in polygon)
    maximum_x = max(point[0] for point in polygon)
    minimum_y = min(point[1] for point in polygon)
    maximum_y = max(point[1] for point in polygon)
    if (
        maximum_x < -half_size
        or minimum_x > half_size
        or maximum_y < -half_size
        or minimum_y > half_size
    ):
        return False

    if any(
        -half_size - _UV_EPS <= point[0] <= half_size + _UV_EPS
        and -half_size - _UV_EPS <= point[1] <= half_size + _UV_EPS
        for point in polygon
    ):
        return True

    corners = (
        (-half_size, -half_size),
        (half_size, -half_size),
        (half_size, half_size),
        (-half_size, half_size),
    )
    if any(_point_in_polygon(corner, polygon) for corner in corners):
        return True

    square_edges = tuple(
        (corners[index], corners[(index + 1) % 4])
        for index in range(4)
    )
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        if any(
            _segments_intersect(point, next_point, edge_a, edge_b)
            for edge_a, edge_b in square_edges
        ):
            return True
    return False


def _face_inside_grid(face, uv_layer, center, axis_x, axis_y, half_size):
    centroid = _face_uv_centroid(face, uv_layer)
    local = _grid_local(centroid, center, axis_x, axis_y)
    return (
        -half_size - _UV_EPS <= local[0] <= half_size + _UV_EPS
        and -half_size - _UV_EPS <= local[1] <= half_size + _UV_EPS
    )


def _face_fully_inside_grid(
    face,
    uv_layer,
    center,
    axis_x,
    axis_y,
    half_size,
):
    polygon = _face_grid_local_polygon(
        face,
        uv_layer,
        center,
        axis_x,
        axis_y,
    )
    return bool(polygon) and all(
        -half_size - _UV_EPS <= point[0] <= half_size + _UV_EPS
        and -half_size - _UV_EPS <= point[1] <= half_size + _UV_EPS
        for point in polygon
    )


def _scope_faces(bm, uv_layer, target_mode, sync_selection):
    if target_mode == "SELECTED_UV":
        return [
            face
            for face in bm.faces
            if _selected_mesh_face(face)
        ]
    return [
        face
        for face in bm.faces
        if _visible_uv_face(face, sync_selection)
    ]


def _cut_face_set_with_line(
    bm,
    uv_layer,
    target_faces,
    origin,
    direction,
    *,
    extend_line,
):
    valid_faces = {
        face for face in target_faces
        if face.is_valid and len(face.loops) >= 3
    }
    edge_cut_buckets = {}
    face_records = []
    for face in valid_faces:
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
    all_target_faces = set(valid_faces)
    cut_edges, cut_faces = _connect_face_pairs(
        face_records,
        uv_layer,
        all_target_faces,
        mark_seams=False,
    )
    return {
        "cut_edges": cut_edges,
        "cut_faces": cut_faces,
        "new_vertices": new_vertices,
        "target_faces": all_target_faces,
    }


def _edge_uv_pair(edge, uv_layer):
    for face in edge.link_faces:
        loop = _loop_for_edge(face, edge)
        if loop is None:
            continue
        next_loop = loop.link_loop_next
        return (
            (loop[uv_layer].uv.x, loop[uv_layer].uv.y),
            (next_loop[uv_layer].uv.x, next_loop[uv_layer].uv.y),
        )
    return None


def _cut_grid_object(
    obj,
    center,
    size,
    angle,
    subdivisions,
    *,
    target_mode,
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

    tag_name = "_uv_batch_knife_grid"
    tag_layer = bm.edges.layers.int.get(tag_name)
    if tag_layer is None:
        tag_layer = bm.edges.layers.int.new(tag_name)
    for edge in bm.edges:
        edge[tag_layer] = 0

    axis_x, axis_y = _grid_axes(angle)
    half_size = size * 0.5
    target_faces = _scope_faces(
        bm,
        uv_layer,
        target_mode,
        sync_selection,
    )
    initial_target_count = len(target_faces)
    current_faces = {
        face
        for face in target_faces
        if _face_overlaps_grid(
            face,
            uv_layer,
            center,
            axis_x,
            axis_y,
            half_size,
        )
    }

    total_new_vertices = 0
    touched_faces = set()
    boundary_lines = (
        (
            _grid_world((-half_size, 0.0), center, axis_x, axis_y),
            axis_y,
        ),
        (
            _grid_world((half_size, 0.0), center, axis_x, axis_y),
            axis_y,
        ),
        (
            _grid_world((0.0, -half_size), center, axis_x, axis_y),
            axis_x,
        ),
        (
            _grid_world((0.0, half_size), center, axis_x, axis_y),
            axis_x,
        ),
    )

    for line_origin, line_direction in boundary_lines:
        candidates = {
            face
            for face in current_faces
            if face.is_valid
            and _face_overlaps_grid(
                face,
                uv_layer,
                center,
                axis_x,
                axis_y,
                half_size,
            )
        }
        result = _cut_face_set_with_line(
            bm,
            uv_layer,
            candidates,
            line_origin,
            line_direction,
            extend_line=True,
        )
        for edge in result["cut_edges"]:
            edge[tag_layer] = 1
        current_faces.update(result["target_faces"])
        touched_faces.update(result["cut_faces"])
        total_new_vertices += result["new_vertices"]

    tolerance = max(_UV_EPS * 10.0, size * 1.0e-6)
    outside_boundary_edges = []
    for edge in tuple(bm.edges):
        if not edge.is_valid or edge[tag_layer] != 1:
            continue
        uv_pair = _edge_uv_pair(edge, uv_layer)
        if uv_pair is None:
            continue
        local_a = _grid_local(uv_pair[0], center, axis_x, axis_y)
        local_b = _grid_local(uv_pair[1], center, axis_x, axis_y)
        midpoint = (
            (local_a[0] + local_b[0]) * 0.5,
            (local_a[1] + local_b[1]) * 0.5,
        )

        on_vertical = (
            (
                abs(local_a[0] + half_size) <= tolerance
                and abs(local_b[0] + half_size) <= tolerance
            )
            or (
                abs(local_a[0] - half_size) <= tolerance
                and abs(local_b[0] - half_size) <= tolerance
            )
        )
        on_horizontal = (
            (
                abs(local_a[1] + half_size) <= tolerance
                and abs(local_b[1] + half_size) <= tolerance
            )
            or (
                abs(local_a[1] - half_size) <= tolerance
                and abs(local_b[1] - half_size) <= tolerance
            )
        )
        inside_segment = (
            on_vertical
            and -half_size - tolerance <= midpoint[1] <= half_size + tolerance
        ) or (
            on_horizontal
            and -half_size - tolerance <= midpoint[0] <= half_size + tolerance
        )
        if (on_vertical or on_horizontal) and not inside_segment:
            outside_boundary_edges.append(edge)

    for edge in outside_boundary_edges:
        if edge.is_valid:
            edge[tag_layer] = 0
            bmesh.ops.dissolve_edges(
                bm,
                edges=[edge],
                use_verts=False,
                use_face_split=False,
            )

    current_faces = {
        face
        for face in _scope_faces(
            bm,
            uv_layer,
            target_mode,
            sync_selection,
        )
        if face.is_valid
        and _face_fully_inside_grid(
            face,
            uv_layer,
            center,
            axis_x,
            axis_y,
            half_size,
        )
    }

    cell_size = size / subdivisions
    internal_lines = []
    for index in range(1, subdivisions):
        offset = -half_size + cell_size * index
        internal_lines.append(
            (
                _grid_world((offset, 0.0), center, axis_x, axis_y),
                axis_y,
            )
        )
        internal_lines.append(
            (
                _grid_world((0.0, offset), center, axis_x, axis_y),
                axis_x,
            )
        )

    for line_origin, line_direction in internal_lines:
        candidates = {
            face
            for face in current_faces
            if face.is_valid
            and _face_fully_inside_grid(
                face,
                uv_layer,
                center,
                axis_x,
                axis_y,
                half_size,
            )
        }
        result = _cut_face_set_with_line(
            bm,
            uv_layer,
            candidates,
            line_origin,
            line_direction,
            extend_line=True,
        )
        for edge in result["cut_edges"]:
            edge[tag_layer] = 2
        current_faces.update(result["target_faces"])
        touched_faces.update(result["cut_faces"])
        total_new_vertices += result["new_vertices"]

    grid_edges = {
        edge
        for edge in bm.edges
        if edge.is_valid and edge[tag_layer] > 0
    }
    if mark_seams:
        for edge in grid_edges:
            edge.seam = True
            edge.select = True

    if split_uv_islands and grid_edges and separation > 0.0:
        inside_faces = [
            face
            for face in _scope_faces(
                bm,
                uv_layer,
                target_mode,
                sync_selection,
            )
            if face.is_valid
            and _face_fully_inside_grid(
                face,
                uv_layer,
                center,
                axis_x,
                axis_y,
                half_size,
            )
        ]
        for face in inside_faces:
            local = _grid_local(
                _face_uv_centroid(face, uv_layer),
                center,
                axis_x,
                axis_y,
            )
            column = min(
                subdivisions - 1,
                max(0, int((local[0] + half_size) / cell_size)),
            )
            row = min(
                subdivisions - 1,
                max(0, int((local[1] + half_size) / cell_size)),
            )
            shift_x = separation * (column + 1)
            shift_y = separation * (row + 1)
            world_shift = (
                axis_x[0] * shift_x + axis_y[0] * shift_y,
                axis_x[1] * shift_x + axis_y[1] * shift_y,
            )
            for loop in face.loops:
                loop[uv_layer].uv.x += world_shift[0]
                loop[uv_layer].uv.y += world_shift[1]

    cut_edge_count = len(grid_edges)
    cut_face_count = len({face for face in touched_faces if face.is_valid})
    bm.select_flush_mode()
    bm.edges.layers.int.remove(tag_layer)

    if cut_edge_count:
        bmesh.update_edit_mesh(mesh, loop_triangles=True, destructive=True)

    return {
        "object": obj.name,
        "target_faces": initial_target_count,
        "cut_faces": cut_face_count,
        "cut_edges": cut_edge_count,
        "new_vertices": total_new_vertices,
    }


def _grid_uv_segments(center, size, angle, subdivisions):
    axis_x, axis_y = _grid_axes(angle)
    half_size = size * 0.5
    segments = []
    for index in range(subdivisions + 1):
        offset = -half_size + size * index / subdivisions
        segments.append(
            (
                _grid_world((offset, -half_size), center, axis_x, axis_y),
                _grid_world((offset, half_size), center, axis_x, axis_y),
            )
        )
        segments.append(
            (
                _grid_world((-half_size, offset), center, axis_x, axis_y),
                _grid_world((half_size, offset), center, axis_x, axis_y),
            )
        )
    return segments


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

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(2.0)
    shader.bind()
    line_color = (
        (0.15, 1.0, 0.35, 1.0)
        if operator._point_snap_hit
        else (
            (0.1, 0.65, 1.0, 1.0)
            if operator._snap_active
            else (1.0, 0.25, 0.05, 0.95)
        )
    )

    if operator.cut_mode == "GRID":
        positions = []
        for segment_start, segment_end in operator._grid_segments_pixel:
            positions.extend((segment_start, segment_end))
        if positions:
            grid_batch = batch_for_shader(
                shader,
                "LINES",
                {"pos": positions},
            )
            shader.uniform_float("color", line_color)
            grid_batch.draw(shader)
    else:
        line_start, line_end = (
            _extended_preview_points(start, end)
            if operator.extend_line
            else (start, end)
        )
        line_batch = batch_for_shader(
            shader,
            "LINES",
            {"pos": (line_start, line_end)},
        )
        shader.uniform_float("color", line_color)
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

    if (
        operator.cut_mode == "GRID"
        and operator._grid_preview_size > _UV_EPS
    ):
        cell_size = (
            operator._grid_preview_size / operator.grid_subdivisions
        )
        angle_degrees = math.degrees(operator._grid_preview_angle)
        step_text = " | STEP 0.1" if operator._grid_step_active else ""
        label = (
            f"{operator.grid_subdivisions} x "
            f"{operator.grid_subdivisions}"
            f" | Size {operator._grid_preview_size:.4f} UV"
            f" | Cell {cell_size:.4f} UV"
            f" | Angle {angle_degrees:.1f}°"
            + step_text
            + f" | {operator._snap_mode_label(context)}"
        )
        blf.position(0, end[0] + 14.0, end[1] + 14.0, 0.0)
        blf.size(0, 14.0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, label)
    elif operator._stage == 1:
        blf.position(0, end[0] + 14.0, end[1] + 14.0, 0.0)
        blf.size(0, 14.0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(0, operator._snap_mode_label(context))

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
    cut_mode: EnumProperty(
        name="Cut Mode",
        description="Cut with one line or with a square grid",
        items=(
            ("LINE", "Line", "Cut with one line"),
            ("GRID", "Grid", "Cut everything inside a square grid"),
        ),
        default="LINE",
    )
    extend_line: BoolProperty(
        name="Extend Line",
        description="Treat the drawn segment as an infinite line",
        default=False,
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
    grid_center_uv: FloatVectorProperty(
        name="Grid Center",
        size=2,
        options={"HIDDEN", "SKIP_SAVE"},
    )
    grid_size: FloatProperty(
        name="Grid Size",
        description="Square side length in UV units",
        default=0.5,
        min=0.000001,
        soft_max=2.0,
        precision=5,
    )
    grid_angle: FloatProperty(
        name="Grid Angle",
        description="Grid rotation in UV space",
        default=0.0,
        subtype="ANGLE",
    )
    grid_subdivisions: IntProperty(
        name="Grid Subdivisions",
        description="Equal number of square cells horizontally and vertically",
        default=2,
        min=1,
        max=64,
    )

    _draw_handle = None
    _area = None
    _area_pointer = 0
    _pixel_start = None
    _pixel_end = None
    _pixel_raw_end = None
    _axis_lock = None
    _active_axis = None
    _snap_active = False
    _snap_mode = "OFF"
    _point_snap_hit = False
    _current_snap_uv = None
    _start_snap_uv = None
    _snap_tree = None
    _snap_points = None
    _snap_caches = None
    _current_grid_snap_step = None
    _grid_segments_pixel = ()
    _grid_preview_size = 0.0
    _grid_preview_angle = 0.0
    _grid_step_active = False
    _cursor_is_modal = False
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
        layout.prop(self, "cut_mode")
        layout.prop(self, "target_mode")
        if self.cut_mode == "GRID":
            layout.prop(self, "grid_size")
            layout.prop(self, "grid_angle")
            layout.prop(self, "grid_subdivisions")
        else:
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
        if self._cursor_is_modal and context.window is not None:
            context.window.cursor_modal_restore()
            self._cursor_is_modal = False
        self._snap_tree = None
        self._snap_points = None
        self._snap_caches = {}
        self._current_grid_snap_step = None
        if context.area is not None:
            context.area.tag_redraw()
        if context.workspace is not None:
            context.workspace.status_text_set(None)

    def _constrained_point(self, point, nearest_axis=False):
        if self._pixel_start is None:
            self._active_axis = None
            self._snap_active = False
            return point

        axis = self._axis_lock
        if axis is None and nearest_axis:
            delta_x = point[0] - self._pixel_start[0]
            delta_y = point[1] - self._pixel_start[1]
            axis = "X" if abs(delta_x) >= abs(delta_y) else "Y"

        self._active_axis = axis
        self._snap_active = axis is not None
        if axis == "X":
            return point[0], self._pixel_start[1]
        if axis == "Y":
            return self._pixel_start[0], point[1]
        return point

    def _uv_grid_snap_steps(self, context):
        uv_editor = getattr(context.space_data, "uv_editor", None)
        grid_source = (
            getattr(uv_editor, "grid_shape_source", "DYNAMIC")
            if uv_editor is not None
            else "DYNAMIC"
        )

        if grid_source == "FIXED":
            subdivisions = getattr(
                uv_editor,
                "custom_grid_subdivisions",
                (10, 10),
            )
            step_x = 1.0 / max(1, int(subdivisions[0]))
            step_y = 1.0 / max(1, int(subdivisions[1]))
            return step_x, step_y

        if grid_source == "PIXEL":
            image = getattr(context.space_data, "image", None)
            image_size = getattr(image, "size", None)
            if (
                image_size is not None
                and len(image_size) >= 2
                and image_size[0] > 0
                and image_size[1] > 0
            ):
                return 1.0 / image_size[0], 1.0 / image_size[1]

        region = context.region
        view2d = region.view2d
        bottom_left = view2d.region_to_view(0.0, 0.0)
        top_right = view2d.region_to_view(
            float(region.width),
            float(region.height),
        )
        uv_per_pixel = max(
            abs(top_right[0] - bottom_left[0]) / max(1, region.width),
            abs(top_right[1] - bottom_left[1]) / max(1, region.height),
        )
        dynamic_step = _nice_grid_step(uv_per_pixel * 32.0)
        return dynamic_step, dynamic_step

    def _snap_mode_label(self, context):
        label = _SNAP_MODE_LABELS.get(self._snap_mode, "S0 OFF")
        if self._snap_mode != "UV_GRID":
            return label
        steps = self._current_grid_snap_step
        if steps is None:
            steps = self._uv_grid_snap_steps(context)
        if abs(steps[0] - steps[1]) <= _UV_EPS:
            return f"{label} ({steps[0]:.6g} UV)"
        return f"{label} ({steps[0]:.6g} x {steps[1]:.6g} UV)"

    def _build_snap_cache(self, context):
        self._snap_tree = None
        self._snap_points = []
        if self._snap_mode not in {
            "POINTS",
            "EDGE_CENTERS",
            "FACE_CENTERS",
        }:
            return

        if self._snap_caches is None:
            self._snap_caches = {}
        cached = self._snap_caches.get(self._snap_mode)
        if cached is not None:
            self._snap_tree, self._snap_points = cached
            return

        seen = set()
        region = context.region
        radius = _POINT_SNAP_RADIUS_PX
        sync_selection = context.scene.tool_settings.use_uv_select_sync

        for obj in _edit_mesh_objects(context):
            mesh = obj.data
            active_uv = mesh.uv_layers.active
            if active_uv is None:
                continue
            bm = bmesh.from_edit_mesh(mesh)
            uv_layer = bm.loops.layers.uv.get(active_uv.name)
            if uv_layer is None:
                continue

            if self.target_mode == "SELECTED_UV":
                faces = (
                    face for face in bm.faces
                    if _selected_mesh_face(face)
                )
            else:
                faces = (
                    face for face in bm.faces
                    if _visible_uv_face(face, sync_selection)
                )

            for face in faces:
                if self._snap_mode == "POINTS":
                    candidates = (
                        (
                            float(loop[uv_layer].uv.x),
                            float(loop[uv_layer].uv.y),
                        )
                        for loop in face.loops
                    )
                elif self._snap_mode == "EDGE_CENTERS":
                    candidates = (
                        (
                            (
                                float(loop[uv_layer].uv.x)
                                + float(loop.link_loop_next[uv_layer].uv.x)
                            )
                            * 0.5,
                            (
                                float(loop[uv_layer].uv.y)
                                + float(loop.link_loop_next[uv_layer].uv.y)
                            )
                            * 0.5,
                        )
                        for loop in face.loops
                    )
                else:
                    candidates = (_face_uv_centroid(face, uv_layer),)

                for uv in candidates:
                    key = (round(uv[0], 8), round(uv[1], 8))
                    if key in seen:
                        continue
                    pixel = region.view2d.view_to_region(
                        uv[0],
                        uv[1],
                        clip=False,
                    )
                    if (
                        pixel[0] < -radius
                        or pixel[0] > region.width + radius
                        or pixel[1] < -radius
                        or pixel[1] > region.height + radius
                    ):
                        continue
                    seen.add(key)
                    self._snap_points.append(
                        ((float(pixel[0]), float(pixel[1])), uv)
                    )

        if self._snap_points:
            tree = KDTree(len(self._snap_points))
            for index, (pixel, _uv) in enumerate(self._snap_points):
                tree.insert((pixel[0], pixel[1], 0.0), index)
            tree.balance()
            self._snap_tree = tree
        self._snap_caches[self._snap_mode] = (
            self._snap_tree,
            self._snap_points,
        )

    def _update_pointer(self, context, point, nearest_axis=False):
        constrained = self._constrained_point(
            point,
            nearest_axis=nearest_axis,
        )
        self._point_snap_hit = False
        self._current_snap_uv = None

        if self._snap_mode == "UV_GRID" and self._active_axis is None:
            step_x, step_y = self._uv_grid_snap_steps(context)
            self._current_grid_snap_step = (step_x, step_y)
            uv = context.region.view2d.region_to_view(*constrained)
            snapped_uv = (
                round(uv[0] / step_x) * step_x,
                round(uv[1] / step_y) * step_y,
            )
            pixel = context.region.view2d.view_to_region(
                snapped_uv[0],
                snapped_uv[1],
                clip=False,
            )
            self._point_snap_hit = True
            self._snap_active = True
            self._current_snap_uv = snapped_uv
            return float(pixel[0]), float(pixel[1])

        if (
            self._snap_mode in {"POINTS", "EDGE_CENTERS", "FACE_CENTERS"}
            and self._active_axis is None
            and self._snap_tree is not None
        ):
            _coordinate, index, distance = self._snap_tree.find(
                (constrained[0], constrained[1], 0.0)
            )
            if distance <= _POINT_SNAP_RADIUS_PX:
                pixel, uv = self._snap_points[index]
                self._point_snap_hit = True
                self._snap_active = True
                self._current_snap_uv = uv
                return pixel
        return constrained

    def _update_grid_preview(
        self,
        context,
        point,
        *,
        snap_angle=False,
        step_size=False,
    ):
        snapped_pixel = self._update_pointer(
            context,
            point,
            nearest_axis=False,
        )
        if self._pixel_start is None:
            self._pixel_end = snapped_pixel
            return

        center_uv = (
            self._start_snap_uv
            if self._start_snap_uv is not None
            else context.region.view2d.region_to_view(*self._pixel_start)
        )
        corner_uv = (
            self._current_snap_uv
            if self._current_snap_uv is not None
            else context.region.view2d.region_to_view(*snapped_pixel)
        )
        delta = _sub2(corner_uv, center_uv)
        radius = math.hypot(delta[0], delta[1])
        if radius <= _UV_EPS:
            self._grid_preview_size = 0.0
            self._grid_segments_pixel = ()
            self._pixel_end = snapped_pixel
            return

        corner_angle = math.atan2(delta[1], delta[0])
        grid_angle = corner_angle - math.pi * 0.25
        if snap_angle:
            angle_step = math.radians(15.0)
            grid_angle = round(grid_angle / angle_step) * angle_step

        size = radius * math.sqrt(2.0)
        if step_size:
            size = max(0.1, round(size / 0.1) * 0.1)
        size = max(size, 0.000001)

        final_corner_angle = grid_angle + math.pi * 0.25
        final_radius = size / math.sqrt(2.0)
        final_corner_uv = (
            center_uv[0] + math.cos(final_corner_angle) * final_radius,
            center_uv[1] + math.sin(final_corner_angle) * final_radius,
        )
        final_corner_pixel = context.region.view2d.view_to_region(
            final_corner_uv[0],
            final_corner_uv[1],
            clip=False,
        )
        self._pixel_end = (
            float(final_corner_pixel[0]),
            float(final_corner_pixel[1]),
        )
        self._grid_preview_size = size
        self._grid_preview_angle = grid_angle
        self._grid_step_active = step_size
        if snap_angle or step_size:
            self._point_snap_hit = False

        segments = _grid_uv_segments(
            center_uv,
            size,
            grid_angle,
            self.grid_subdivisions,
        )
        self._grid_segments_pixel = tuple(
            (
                tuple(
                    float(value)
                    for value in context.region.view2d.view_to_region(
                        segment_start[0],
                        segment_start[1],
                        clip=False,
                    )
                ),
                tuple(
                    float(value)
                    for value in context.region.view2d.view_to_region(
                        segment_end[0],
                        segment_end[1],
                        clip=False,
                    )
                ),
            )
            for segment_start, segment_end in segments
        )

    def _set_status_text(self, context):
        snap_text = self._snap_mode_label(context)
        if self.cut_mode == "GRID":
            context.workspace.status_text_set(
                f"UV Grid Knife: {self.grid_subdivisions} x "
                f"{self.grid_subdivisions}; колесо — подразделения; "
                f"Ctrl — угол 15°; Alt — шаг размера 0.1 UV; "
                f"S — режим снапа: {snap_text}; "
                f"G — режим линии; ПКМ/Esc — отмена"
            )
            return

        lock_text = ""
        if self._axis_lock == "X":
            lock_text = " | X: горизонталь зафиксирована"
        elif self._axis_lock == "Y":
            lock_text = " | Y: вертикаль зафиксирована"
        line_text = "бесконечная" if self.extend_line else "от точки до точки"
        context.workspace.status_text_set(
            f"UV Batch Knife: линия {line_text}; "
            f"S — режим снапа: {snap_text}; "
            "C — тип линии; Ctrl — ближайшая ось; X/Y — фиксация оси; "
            "G — режим грида; ПКМ/Esc — отмена"
            + lock_text
        )

    def invoke(self, context, event):
        self._area = context.area
        self._area_pointer = context.area.as_pointer()
        self.cut_mode = "LINE"
        self.extend_line = False
        self.grid_subdivisions = 2
        self._pixel_start = None
        self._pixel_end = (event.mouse_region_x, event.mouse_region_y)
        self._pixel_raw_end = self._pixel_end
        self._axis_lock = None
        self._active_axis = None
        self._snap_active = False
        self._snap_mode = "OFF"
        self._point_snap_hit = False
        self._current_snap_uv = None
        self._start_snap_uv = None
        self._snap_tree = None
        self._snap_points = None
        self._snap_caches = {}
        self._current_grid_snap_step = None
        self._grid_segments_pixel = ()
        self._grid_preview_size = 0.0
        self._grid_preview_angle = 0.0
        self._grid_step_active = False
        self._cursor_is_modal = False
        self._stage = 0
        self._draw_handle = bpy.types.SpaceImageEditor.draw_handler_add(
            _draw_batch_knife,
            (self,),
            "WINDOW",
            "POST_PIXEL",
        )
        context.window.cursor_modal_set("KNIFE")
        self._cursor_is_modal = True
        context.window_manager.modal_handler_add(self)
        self._set_status_text(context)
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
            self._pixel_raw_end = (event.mouse_region_x, event.mouse_region_y)
            if self.cut_mode == "GRID" and self._stage == 1:
                self._update_grid_preview(
                    context,
                    self._pixel_raw_end,
                    snap_angle=event.ctrl,
                    step_size=event.alt,
                )
            else:
                self._pixel_end = self._update_pointer(
                    context,
                    self._pixel_raw_end,
                    nearest_axis=event.ctrl,
                )
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "S" and event.value == "PRESS":
            snap_index = _SNAP_MODES.index(self._snap_mode)
            self._snap_mode = _SNAP_MODES[
                (snap_index + 1) % len(_SNAP_MODES)
            ]
            self._snap_tree = None
            self._snap_points = None
            self._current_grid_snap_step = None
            if self._snap_mode in {
                "POINTS",
                "EDGE_CENTERS",
                "FACE_CENTERS",
            }:
                self._build_snap_cache(context)
            if self.cut_mode == "GRID" and self._stage == 1:
                self._update_grid_preview(
                    context,
                    self._pixel_raw_end,
                    snap_angle=event.ctrl,
                    step_size=event.alt,
                )
            else:
                self._pixel_end = self._update_pointer(
                    context,
                    self._pixel_raw_end,
                    nearest_axis=False,
                )
            self._set_status_text(context)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if (
            self.cut_mode == "LINE"
            and event.type == "C"
            and event.value == "PRESS"
        ):
            self.extend_line = not self.extend_line
            self._set_status_text(context)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "G" and event.value == "PRESS":
            self.cut_mode = "GRID" if self.cut_mode == "LINE" else "LINE"
            self._axis_lock = None
            self._active_axis = None
            self._grid_segments_pixel = ()
            if self._stage == 1:
                if self.cut_mode == "GRID":
                    self._update_grid_preview(
                        context,
                        self._pixel_raw_end,
                        snap_angle=event.ctrl,
                        step_size=event.alt,
                    )
                else:
                    self._pixel_end = self._update_pointer(
                        context,
                        self._pixel_raw_end,
                        nearest_axis=False,
                    )
            self._set_status_text(context)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if (
            self.cut_mode == "GRID"
            and self._stage == 1
            and event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}
        ):
            increment = 1 if event.type == "WHEELUPMOUSE" else -1
            self.grid_subdivisions = min(
                64,
                max(1, self.grid_subdivisions + increment),
            )
            self._update_grid_preview(
                context,
                self._pixel_raw_end,
                snap_angle=event.ctrl,
                step_size=event.alt,
            )
            self._set_status_text(context)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if (
            self.cut_mode == "LINE"
            and
            self._stage == 1
            and event.type in {"X", "Y"}
            and event.value == "PRESS"
        ):
            self._axis_lock = (
                None if self._axis_lock == event.type else event.type
            )
            self._pixel_end = self._update_pointer(
                context,
                self._pixel_raw_end,
                nearest_axis=False,
            )
            self._set_status_text(context)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            raw_point = (event.mouse_region_x, event.mouse_region_y)
            if self._stage == 0:
                self._pixel_raw_end = raw_point
                point = self._update_pointer(
                    context,
                    raw_point,
                    nearest_axis=False,
                )
                self._pixel_start = point
                self._pixel_end = point
                self._start_snap_uv = self._current_snap_uv
                self._stage = 1
                if self.cut_mode == "GRID":
                    self._grid_preview_size = 0.0
                    self._grid_segments_pixel = ()
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}

            self._pixel_raw_end = raw_point
            if self.cut_mode == "GRID":
                self._update_grid_preview(
                    context,
                    raw_point,
                    snap_angle=event.ctrl,
                    step_size=event.alt,
                )
                if self._grid_preview_size <= _UV_EPS:
                    return {"RUNNING_MODAL"}
                center_uv = (
                    self._start_snap_uv
                    if self._start_snap_uv is not None
                    else context.region.view2d.region_to_view(
                        *self._pixel_start
                    )
                )
                self.grid_center_uv = center_uv
                self.grid_size = self._grid_preview_size
                self.grid_angle = self._grid_preview_angle
                self._finish_modal(context)
                return self.execute(context)

            self._pixel_end = self._update_pointer(
                context,
                raw_point,
                nearest_axis=event.ctrl,
            )
            if _dist_sq(self._pixel_end, self._pixel_start) < 16.0:
                return {"RUNNING_MODAL"}

            start_uv = (
                self._start_snap_uv
                if self._start_snap_uv is not None
                else context.region.view2d.region_to_view(*self._pixel_start)
            )
            end_uv = (
                self._current_snap_uv
                if self._current_snap_uv is not None
                else context.region.view2d.region_to_view(*self._pixel_end)
            )
            if self._active_axis == "X":
                end_uv = (end_uv[0], start_uv[1])
            elif self._active_axis == "Y":
                end_uv = (start_uv[0], end_uv[1])
            self.start_uv = start_uv
            self.end_uv = end_uv
            self._finish_modal(context)
            return self.execute(context)

        return {"RUNNING_MODAL"}

    def execute(self, context):
        started = time.perf_counter()
        sync_selection = context.scene.tool_settings.use_uv_select_sync
        results = []
        if self.cut_mode == "GRID":
            center = (
                float(self.grid_center_uv[0]),
                float(self.grid_center_uv[1]),
            )
            if self.grid_size <= _UV_EPS:
                self.report({"WARNING"}, "UV Grid Knife слишком маленький")
                return {"CANCELLED"}
            for obj in _edit_mesh_objects(context):
                results.append(
                    _cut_grid_object(
                        obj,
                        center,
                        self.grid_size,
                        self.grid_angle,
                        self.grid_subdivisions,
                        target_mode=self.target_mode,
                        split_uv_islands=self.split_uv_islands,
                        separation=self.separation,
                        mark_seams=self.mark_seams,
                        sync_selection=sync_selection,
                    )
                )
        else:
            origin = (float(self.start_uv[0]), float(self.start_uv[1]))
            end = (float(self.end_uv[0]), float(self.end_uv[1]))
            direction = _sub2(end, origin)
            if (
                direction[0] * direction[0]
                + direction[1] * direction[1]
                <= _UV_EPS
            ):
                self.report(
                    {"WARNING"},
                    "Линия UV Batch Knife слишком короткая",
                )
                return {"CANCELLED"}
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
            tool_name = (
                "UV Grid Knife"
                if self.cut_mode == "GRID"
                else "Линия"
            )
            self.report(
                {"WARNING"},
                (
                    f"{tool_name} не пересёк подходящие UV-грани "
                    f"(проверено: {target_faces})"
                ),
            )
            return {"CANCELLED"}

        tool_name = (
            "UV Grid Knife"
            if self.cut_mode == "GRID"
            else "UV Batch Knife"
        )
        self.report(
            {"INFO"},
            (
                f"{tool_name}: рёбер {cut_edges}, вершин {new_vertices}, "
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
