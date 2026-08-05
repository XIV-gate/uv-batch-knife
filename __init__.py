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
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
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
_LINE_MODES = ("FINITE", "INFINITE", "MULTI")


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


def _project_point_to_segment(point, edge_a, edge_b):
    edge = _sub2(edge_b, edge_a)
    length_sq = edge[0] * edge[0] + edge[1] * edge[1]
    if length_sq <= _UV_EPS * _UV_EPS:
        return edge_a, 0.0, _dist_sq(point, edge_a)
    relative = _sub2(point, edge_a)
    factor = (
        relative[0] * edge[0] + relative[1] * edge[1]
    ) / length_sq
    factor = min(1.0, max(0.0, factor))
    projected = (
        edge_a[0] + edge[0] * factor,
        edge_a[1] + edge[1] * factor,
    )
    return projected, factor, _dist_sq(point, projected)


def _face_boundary_descriptor(
    face,
    uv_layer,
    point,
    *,
    nearest=False,
):
    best = None
    for loop in face.loops:
        next_loop = loop.link_loop_next
        uv_a = (loop[uv_layer].uv.x, loop[uv_layer].uv.y)
        uv_b = (
            next_loop[uv_layer].uv.x,
            next_loop[uv_layer].uv.y,
        )
        projected, factor, distance_sq = _project_point_to_segment(
            point,
            uv_a,
            uv_b,
        )
        if best is None or distance_sq < best[0]:
            best = (distance_sq, loop, factor, projected)

    if best is None:
        return None
    tolerance = max(_UV_EPS * 10.0, 1.0e-6)
    if not nearest and best[0] > tolerance * tolerance:
        return None

    _distance_sq, loop, factor, projected = best
    next_loop = loop.link_loop_next
    descriptor = {
        "point": projected,
        "edge": None,
        "edge_factor": None,
        "vert": None,
        "loop": loop,
    }
    if factor <= _UV_EPS:
        descriptor["vert"] = loop.vert
    elif factor >= 1.0 - _UV_EPS:
        descriptor["vert"] = next_loop.vert
    else:
        edge = loop.edge
        descriptor["edge"] = edge
        descriptor["edge_factor"] = (
            factor
            if loop.vert is edge.verts[0]
            else 1.0 - factor
        )
    return descriptor


def _point_on_polygon_boundary(point, polygon):
    tolerance_sq = max(_UV_EPS * 10.0, 1.0e-6) ** 2
    for index, edge_a in enumerate(polygon):
        edge_b = polygon[(index + 1) % len(polygon)]
        _projected, _factor, distance_sq = _project_point_to_segment(
            point,
            edge_a,
            edge_b,
        )
        if distance_sq <= tolerance_sq:
            return True
    return False


def _segment_inside_polygon_intervals(start, end, polygon):
    direction = _sub2(end, start)
    if _dist_sq(start, end) <= _UV_EPS * _UV_EPS:
        return []

    factors = [0.0, 1.0]
    for index, edge_a in enumerate(polygon):
        edge_b = polygon[(index + 1) % len(polygon)]
        intersection = _line_edge_intersection(
            start,
            direction,
            edge_a,
            edge_b,
            False,
        )
        if intersection is not None:
            factors.append(intersection[0])

    factors.sort()
    unique_factors = []
    for factor in factors:
        if (
            not unique_factors
            or abs(factor - unique_factors[-1]) > _FACTOR_EPS
        ):
            unique_factors.append(factor)

    intervals = []
    for factor_a, factor_b in zip(
        unique_factors,
        unique_factors[1:],
    ):
        if factor_b - factor_a <= _FACTOR_EPS:
            continue
        midpoint_factor = (factor_a + factor_b) * 0.5
        midpoint = (
            start[0] + direction[0] * midpoint_factor,
            start[1] + direction[1] * midpoint_factor,
        )
        if not _point_in_polygon(midpoint, polygon):
            continue
        if _point_on_polygon_boundary(midpoint, polygon):
            continue
        intervals.append(
            (
                (
                    start[0] + direction[0] * factor_a,
                    start[1] + direction[1] * factor_a,
                ),
                (
                    start[0] + direction[0] * factor_b,
                    start[1] + direction[1] * factor_b,
                ),
            )
        )
    return intervals


def _deduplicate_polyline(points):
    result = []
    tolerance_sq = _FACTOR_EPS * _FACTOR_EPS
    for point in points:
        normalized = (float(point[0]), float(point[1]))
        if result and _dist_sq(normalized, result[-1]) <= tolerance_sq:
            continue
        result.append(normalized)
    return result


def _polyline_face_point_chains(face, uv_layer, points):
    polygon = [
        (loop[uv_layer].uv.x, loop[uv_layer].uv.y)
        for loop in face.loops
    ]
    working = _deduplicate_polyline(points)
    if len(working) < 2 or len(polygon) < 3:
        return []

    start_boundary = _face_boundary_descriptor(
        face,
        uv_layer,
        working[0],
    )
    if _point_in_polygon(working[0], polygon) and start_boundary is None:
        nearest = _face_boundary_descriptor(
            face,
            uv_layer,
            working[0],
            nearest=True,
        )
        if nearest is not None:
            working.insert(0, nearest["point"])

    end_boundary = _face_boundary_descriptor(
        face,
        uv_layer,
        working[-1],
    )
    if _point_in_polygon(working[-1], polygon) and end_boundary is None:
        nearest = _face_boundary_descriptor(
            face,
            uv_layer,
            working[-1],
            nearest=True,
        )
        if nearest is not None:
            working.append(nearest["point"])

    raw_chains = []
    current = []
    tolerance_sq = _FACTOR_EPS * _FACTOR_EPS
    for segment_start, segment_end in zip(working, working[1:]):
        intervals = _segment_inside_polygon_intervals(
            segment_start,
            segment_end,
            polygon,
        )
        if not intervals:
            if len(current) >= 2:
                raw_chains.append(current)
            current = []
            continue
        for interval_start, interval_end in intervals:
            if (
                current
                and _dist_sq(current[-1], interval_start) <= tolerance_sq
            ):
                if _dist_sq(current[-1], interval_end) > tolerance_sq:
                    current.append(interval_end)
            else:
                if len(current) >= 2:
                    raw_chains.append(current)
                current = [interval_start, interval_end]
    if len(current) >= 2:
        raw_chains.append(current)

    chains = []
    for raw_chain in raw_chains:
        chain = _deduplicate_polyline(raw_chain)
        section_start = 0
        for index in range(1, len(chain) - 1):
            if _face_boundary_descriptor(
                face,
                uv_layer,
                chain[index],
            ) is None:
                continue
            section = chain[section_start:index + 1]
            if len(section) >= 2:
                chains.append(section)
            section_start = index
        section = chain[section_start:]
        if len(section) >= 2:
            chains.append(section)

    return [
        chain
        for chain in chains
        if _face_boundary_descriptor(
            face,
            uv_layer,
            chain[0],
        ) is not None
        and _face_boundary_descriptor(
            face,
            uv_layer,
            chain[-1],
        ) is not None
        and _dist_sq(chain[0], chain[-1]) > tolerance_sq
    ]


def _barycentric_weights_2d(point, a, b, c):
    denominator = (
        (b[1] - c[1]) * (a[0] - c[0])
        + (c[0] - b[0]) * (a[1] - c[1])
    )
    if abs(denominator) <= _UV_EPS:
        return None
    weight_a = (
        (b[1] - c[1]) * (point[0] - c[0])
        + (c[0] - b[0]) * (point[1] - c[1])
    ) / denominator
    weight_b = (
        (c[1] - a[1]) * (point[0] - c[0])
        + (a[0] - c[0]) * (point[1] - c[1])
    ) / denominator
    weight_c = 1.0 - weight_a - weight_b
    return weight_a, weight_b, weight_c


def _face_uv_to_3d(face, uv_layer, point):
    loops = list(face.loops)
    uv_points = [
        (loop[uv_layer].uv.x, loop[uv_layer].uv.y)
        for loop in loops
    ]
    for loop, uv in zip(loops, uv_points):
        if _dist_sq(point, uv) <= _UV_EPS * _UV_EPS:
            return loop.vert.co.copy()

    uv_vectors = [Vector((uv[0], uv[1], 0.0)) for uv in uv_points]
    try:
        triangles = tessellate_polygon([uv_vectors])
    except (RuntimeError, ValueError):
        triangles = ()

    for triangle in triangles:
        if all(isinstance(value, int) for value in triangle):
            indices = list(triangle)
        else:
            indices = []
            remaining = set(range(len(uv_vectors)))
            for vector in triangle:
                index = min(
                    remaining,
                    key=lambda candidate: (
                        uv_vectors[candidate] - vector
                    ).length_squared,
                )
                indices.append(index)
                remaining.discard(index)
        weights = _barycentric_weights_2d(
            point,
            uv_points[indices[0]],
            uv_points[indices[1]],
            uv_points[indices[2]],
        )
        if weights is None or min(weights) < -_FACTOR_EPS:
            continue
        coordinate = Vector((0.0, 0.0, 0.0))
        for weight, index in zip(weights, indices):
            coordinate += loops[index].vert.co * weight
        return coordinate

    nearest = _face_boundary_descriptor(
        face,
        uv_layer,
        point,
        nearest=True,
    )
    if nearest is not None:
        loop = nearest["loop"]
        projected, factor, _distance_sq = _project_point_to_segment(
            point,
            (loop[uv_layer].uv.x, loop[uv_layer].uv.y),
            (
                loop.link_loop_next[uv_layer].uv.x,
                loop.link_loop_next[uv_layer].uv.y,
            ),
        )
        del projected
        return loop.vert.co.lerp(loop.link_loop_next.vert.co, factor)

    coordinate = Vector((0.0, 0.0, 0.0))
    for loop in loops:
        coordinate += loop.vert.co
    return coordinate / max(1, len(loops))


def _collect_polyline_face_cut(
    face,
    uv_layer,
    points,
    edge_cut_buckets,
):
    point_chains = _polyline_face_point_chains(
        face,
        uv_layer,
        points,
    )
    chains = []
    for point_chain in point_chains:
        nodes = []
        valid = True
        for index, point in enumerate(point_chain):
            boundary = None
            if index in {0, len(point_chain) - 1}:
                boundary = _face_boundary_descriptor(
                    face,
                    uv_layer,
                    point,
                )
                if boundary is None:
                    valid = False
                    break
            if boundary is not None:
                node = boundary
                if node["edge"] is not None:
                    edge_cut_buckets.setdefault(
                        node["edge"],
                        [],
                    ).append(node)
            else:
                node = {
                    "point": point,
                    "edge": None,
                    "edge_factor": None,
                    "vert": None,
                    "co": _face_uv_to_3d(face, uv_layer, point),
                }
            nodes.append(node)
        if valid and len(nodes) >= 2:
            chains.append(nodes)
    if not chains:
        return None
    return {"face": face, "chains": chains}


def _connect_face_chains(
    bm,
    face_records,
    uv_layer,
    all_target_faces,
    mark_seams,
):
    cut_edges = set()
    cut_faces = set()
    edge_directions = {}
    created_vertices = 0

    for record in face_records:
        original_face = record["face"]
        if not original_face.is_valid:
            continue
        descendants = {original_face}

        for chain in record["chains"]:
            vert_a = chain[0]["vert"]
            vert_b = chain[-1]["vert"]
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

            coordinates = [
                node["co"]
                for node in chain[1:-1]
            ]
            old_edges = set(candidate.edges)
            old_verts = set(candidate.verts)
            selected = candidate.select
            try:
                split_result = bmesh.utils.face_split(
                    candidate,
                    vert_a,
                    vert_b,
                    coords=coordinates,
                    use_exist=False,
                )
            except (RuntimeError, ValueError):
                continue
            if split_result is None:
                continue
            new_face, new_loop = split_result
            if new_face is None or new_loop is None:
                continue

            candidate.select = selected
            new_face.select = selected
            descendants.add(new_face)
            all_target_faces.add(new_face)
            cut_faces.add(candidate)
            cut_faces.add(new_face)

            split_faces = {candidate, new_face}
            new_edges = {
                edge
                for face in split_faces
                for edge in face.edges
                if edge not in old_edges
            }
            new_verts = {
                vert
                for face in split_faces
                for vert in face.verts
                if vert not in old_verts
            }
            created_vertices += len(new_verts)

            available_verts = set(new_verts)
            for node in chain[1:-1]:
                if not available_verts:
                    break
                vert = min(
                    available_verts,
                    key=lambda candidate_vert: (
                        candidate_vert.co - node["co"]
                    ).length_squared,
                )
                node["vert"] = vert
                available_verts.remove(vert)
                for loop in vert.link_loops:
                    if loop.face in descendants:
                        loop[uv_layer].uv = node["point"]

            chain_verts = [node["vert"] for node in chain]
            for index, (path_vert_a, path_vert_b) in enumerate(
                zip(chain_verts, chain_verts[1:])
            ):
                if path_vert_a is None or path_vert_b is None:
                    continue
                edge = bm.edges.get((path_vert_a, path_vert_b))
                if edge is None or edge not in new_edges:
                    continue
                edge.select = True
                path_vert_a.select = True
                path_vert_b.select = True
                if mark_seams:
                    edge.seam = True
                cut_edges.add(edge)
                edge_directions[edge] = (
                    chain[index]["point"],
                    chain[index + 1]["point"],
                )

    return (
        cut_edges,
        cut_faces,
        created_vertices,
        edge_directions,
    )


def _positive_polyline_side_faces(
    all_target_faces,
    cut_edges,
    edge_directions,
    uv_layer,
):
    valid_faces = {face for face in all_target_faces if face.is_valid}
    boundary_faces = {
        face
        for edge in cut_edges
        for face in edge.link_faces
        if face in valid_faces
    }
    components = []
    face_components = {}
    for seed in boundary_faces:
        if seed in face_components:
            continue
        component_index = len(components)
        component = {seed}
        face_components[seed] = component_index
        queue = deque([seed])
        while queue:
            face = queue.popleft()
            for edge in face.edges:
                if edge in cut_edges or edge.seam:
                    continue
                for neighbor in edge.link_faces:
                    if neighbor is face or neighbor not in valid_faces:
                        continue
                    if neighbor in face_components:
                        continue
                    if not _uv_continuous_across(
                        face,
                        neighbor,
                        edge,
                        uv_layer,
                    ):
                        continue
                    face_components[neighbor] = component_index
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    adjacency = {index: set() for index in range(len(components))}
    positive_votes = [0 for _component in components]
    for edge in cut_edges:
        direction_points = edge_directions.get(edge)
        if direction_points is None:
            continue
        linked_components = {
            face_components[face]
            for face in edge.link_faces
            if face in face_components
        }
        for component_a in linked_components:
            adjacency[component_a].update(
                component_b
                for component_b in linked_components
                if component_b != component_a
            )

        origin, end = direction_points
        direction = _sub2(end, origin)
        for face in edge.link_faces:
            if face not in face_components:
                continue
            loop = _loop_for_edge(face, edge)
            if loop is None:
                continue
            side = 0.0
            neighbor_loops = (
                loop.link_loop_prev,
                loop.link_loop_next.link_loop_next,
            )
            for neighbor_loop in neighbor_loops:
                neighbor_uv = neighbor_loop[uv_layer].uv
                side = _signed_distance(
                    (neighbor_uv.x, neighbor_uv.y),
                    origin,
                    direction,
                )
                if abs(side) > _UV_EPS:
                    break
            if side > _UV_EPS:
                positive_votes[face_components[face]] += 1

    result = set()
    visited_components = set()
    for root in range(len(components)):
        if root in visited_components or not adjacency[root]:
            continue
        colors = {root: 0}
        group_queue = deque([root])
        visited_components.add(root)
        while group_queue:
            component = group_queue.popleft()
            for neighbor in adjacency[component]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[component]
                    visited_components.add(neighbor)
                    group_queue.append(neighbor)
        vote_totals = [0, 0]
        for component, color in colors.items():
            vote_totals[color] += positive_votes[component]
        selected_color = 0 if vote_totals[0] >= vote_totals[1] else 1
        for component, color in colors.items():
            if color == selected_color:
                result.update(components[component])
    return result


def _cut_polyline_object(
    obj,
    points,
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

    normalized_points = _deduplicate_polyline(points)
    target_faces = _scope_faces(
        bm,
        uv_layer,
        target_mode,
        sync_selection,
    )
    if len(normalized_points) < 2:
        return {
            "object": obj.name,
            "target_faces": len(target_faces),
            "cut_faces": 0,
            "cut_edges": 0,
            "new_vertices": 0,
        }

    all_target_faces = set(target_faces)
    edge_cut_buckets = {}
    face_records = []
    for face in target_faces:
        if not face.is_valid or len(face.loops) < 3:
            continue
        record = _collect_polyline_face_cut(
            face,
            uv_layer,
            normalized_points,
            edge_cut_buckets,
        )
        if record is not None:
            face_records.append(record)

    boundary_vertices = _split_requested_edges(edge_cut_buckets)
    (
        cut_edges,
        cut_faces,
        interior_vertices,
        edge_directions,
    ) = _connect_face_chains(
        bm,
        face_records,
        uv_layer,
        all_target_faces,
        mark_seams,
    )

    if split_uv_islands and cut_edges and separation > 0.0:
        positive_faces = _positive_polyline_side_faces(
            all_target_faces,
            cut_edges,
            edge_directions,
            uv_layer,
        )
        direction = None
        for point_a, point_b in zip(
            normalized_points,
            normalized_points[1:],
        ):
            candidate = _sub2(point_b, point_a)
            if candidate[0] * candidate[0] + candidate[1] * candidate[1] > _UV_EPS:
                direction = candidate
                break
        if direction is not None:
            _shift_uv_faces(
                positive_faces,
                uv_layer,
                direction,
                separation,
            )

    new_vertices = boundary_vertices + interior_vertices
    if cut_edges or new_vertices:
        bm.select_flush_mode()
        bmesh.update_edit_mesh(
            mesh,
            loop_triangles=True,
            destructive=True,
        )

    return {
        "object": obj.name,
        "target_faces": len(target_faces),
        "cut_faces": len({face for face in cut_faces if face.is_valid}),
        "cut_edges": len({edge for edge in cut_edges if edge.is_valid}),
        "new_vertices": new_vertices,
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

    end = operator._pixel_end
    if end is None:
        return
    start = operator._pixel_start if operator._pixel_start is not None else end

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

    if operator.cut_mode == "GRID" and operator._stage == 1:
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
    elif operator.cut_mode == "LINE" and operator._stage == 1:
        if operator.line_mode == "MULTI":
            positions = []
            committed = list(operator._path_pixel_points)
            for point_a, point_b in zip(committed, committed[1:]):
                positions.extend((point_a, point_b))
            if committed and _dist_sq(committed[-1], end) > 1.0:
                positions.extend((committed[-1], end))
            if positions:
                line_batch = batch_for_shader(
                    shader,
                    "LINES",
                    {"pos": positions},
                )
                shader.uniform_float("color", line_color)
                line_batch.draw(shader)
        else:
            line_start, line_end = (
                _extended_preview_points(start, end)
                if operator.line_mode == "INFINITE"
                else (start, end)
            )
            line_batch = batch_for_shader(
                shader,
                "LINES",
                {"pos": (line_start, line_end)},
            )
            shader.uniform_float("color", line_color)
            line_batch.draw(shader)

    fixed_points = []
    if operator._stage == 1:
        if operator.cut_mode == "LINE" and operator.line_mode == "MULTI":
            fixed_points = list(operator._path_pixel_points)
        else:
            fixed_points = [start]
    if fixed_points:
        gpu.state.point_size_set(7.0)
        point_batch = batch_for_shader(
            shader,
            "POINTS",
            {"pos": fixed_points},
        )
        shader.uniform_float("color", (1.0, 0.7, 0.1, 1.0))
        point_batch.draw(shader)

    gpu.state.point_size_set(9.0 if operator._stage == 0 else 7.0)
    cursor_batch = batch_for_shader(
        shader,
        "POINTS",
        {"pos": (end,)},
    )
    cursor_color = (
        (0.15, 1.0, 0.35, 1.0)
        if operator._point_snap_hit
        else (1.0, 0.7, 0.1, 1.0)
    )
    shader.uniform_float("color", cursor_color)
    cursor_batch.draw(shader)

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
    elif operator.cut_mode == "LINE":
        point_count = (
            f" | Points {len(operator._path_uv_points)}"
            if operator.line_mode == "MULTI"
            else ""
        )
        blf.position(0, end[0] + 14.0, end[1] + 14.0, 0.0)
        blf.size(0, 14.0)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.draw(
            0,
            f"{operator._line_mode_label()} | "
            f"{operator._snap_mode_label(context)}{point_count}",
        )
    else:
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
    line_mode: EnumProperty(
        name="Line Mode",
        description="Choose a finite, infinite, or multi-point cut",
        items=(
            (
                "FINITE",
                "Point to Point",
                "Cut only along the segment between two points",
            ),
            (
                "INFINITE",
                "Infinite Line",
                "Extend the line infinitely in both directions",
            ),
            (
                "MULTI",
                "Multi-Point",
                "Place a polyline of Knife points and confirm with Enter",
            ),
        ),
        default="FINITE",
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
    _path_pixel_points = ()
    _path_uv_points = ()
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
            layout.prop(self, "line_mode")
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

    def _line_mode_label(self):
        return {
            "FINITE": "C1 Point to Point",
            "INFINITE": "C2 Infinite Line",
            "MULTI": "C3 Multi-Point",
        }.get(self.line_mode, "C1 Point to Point")

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
        line_text = self._line_mode_label()
        multi_text = (
            " Enter — завершить; Backspace — удалить точку;"
            if self.line_mode == "MULTI"
            else ""
        )
        context.workspace.status_text_set(
            f"UV Batch Knife: {line_text}; "
            f"S — режим снапа: {snap_text}; "
            f"C — тип линии;{multi_text} "
            "Ctrl — ближайшая ось; X/Y — фиксация оси; "
            "G — режим грида; ПКМ/Esc — отмена"
            + lock_text
        )

    def invoke(self, context, event):
        self._area = context.area
        self._area_pointer = context.area.as_pointer()
        self.cut_mode = "LINE"
        self.line_mode = "FINITE"
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
        self._path_pixel_points = []
        self._path_uv_points = []
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
            previous_mode = self.line_mode
            mode_index = _LINE_MODES.index(self.line_mode)
            self.line_mode = _LINE_MODES[
                (mode_index + 1) % len(_LINE_MODES)
            ]
            self.extend_line = self.line_mode == "INFINITE"
            if (
                previous_mode == "MULTI"
                and self.line_mode != "MULTI"
                and self._path_pixel_points
            ):
                self._path_pixel_points = [self._path_pixel_points[0]]
                self._path_uv_points = [self._path_uv_points[0]]
                self._pixel_start = self._path_pixel_points[0]
            self._set_status_text(context)
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}

        if event.type == "G" and event.value == "PRESS":
            self.cut_mode = "GRID" if self.cut_mode == "LINE" else "LINE"
            self._axis_lock = None
            self._active_axis = None
            self._grid_segments_pixel = ()
            if self._path_pixel_points:
                self._path_pixel_points = [self._path_pixel_points[0]]
                self._path_uv_points = [self._path_uv_points[0]]
                self._pixel_start = self._path_pixel_points[0]
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
            self.cut_mode == "LINE"
            and self.line_mode == "MULTI"
            and self._stage == 1
            and event.type in {"RET", "NUMPAD_ENTER"}
            and event.value == "PRESS"
        ):
            if len(self._path_uv_points) < 2:
                return {"RUNNING_MODAL"}
            self._finish_modal(context)
            return self.execute(context)

        if (
            self.cut_mode == "LINE"
            and self.line_mode == "MULTI"
            and self._stage == 1
            and event.type == "BACK_SPACE"
            and event.value == "PRESS"
        ):
            if len(self._path_pixel_points) > 1:
                self._path_pixel_points.pop()
                self._path_uv_points.pop()
                self._pixel_start = self._path_pixel_points[-1]
                self._start_snap_uv = self._path_uv_points[-1]
                self._pixel_end = self._update_pointer(
                    context,
                    self._pixel_raw_end,
                    nearest_axis=False,
                )
            else:
                self._path_pixel_points = []
                self._path_uv_points = []
                self._pixel_start = None
                self._start_snap_uv = None
                self._stage = 0
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
                point_uv = (
                    self._current_snap_uv
                    if self._current_snap_uv is not None
                    else context.region.view2d.region_to_view(*point)
                )
                self._pixel_start = point
                self._pixel_end = point
                self._start_snap_uv = self._current_snap_uv
                self._path_pixel_points = [point]
                self._path_uv_points = [
                    (float(point_uv[0]), float(point_uv[1]))
                ]
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

            if self.line_mode == "MULTI":
                committed_pixel = (
                    float(self._pixel_end[0]),
                    float(self._pixel_end[1]),
                )
                committed_uv = (float(end_uv[0]), float(end_uv[1]))
                self._path_pixel_points.append(committed_pixel)
                self._path_uv_points.append(committed_uv)
                self._pixel_start = committed_pixel
                self._start_snap_uv = committed_uv
                self._pixel_end = committed_pixel
                self._active_axis = None
                self._snap_active = self._point_snap_hit
                self._set_status_text(context)
                context.area.tag_redraw()
                return {"RUNNING_MODAL"}

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
            if self.line_mode == "MULTI":
                path_points = _deduplicate_polyline(
                    self._path_uv_points
                )
            else:
                path_points = [
                    (float(self.start_uv[0]), float(self.start_uv[1])),
                    (float(self.end_uv[0]), float(self.end_uv[1])),
                ]
            if len(path_points) < 2:
                self.report(
                    {"WARNING"},
                    "Линия UV Batch Knife слишком короткая",
                )
                return {"CANCELLED"}

            origin = path_points[0]
            direction = _sub2(path_points[-1], origin)
            for obj in _edit_mesh_objects(context):
                if self.line_mode == "INFINITE":
                    results.append(
                        _cut_object(
                            obj,
                            origin,
                            direction,
                            target_mode=self.target_mode,
                            extend_line=True,
                            split_uv_islands=self.split_uv_islands,
                            separation=self.separation,
                            mark_seams=self.mark_seams,
                            sync_selection=sync_selection,
                        )
                    )
                else:
                    results.append(
                        _cut_polyline_object(
                            obj,
                            path_points,
                            target_mode=self.target_mode,
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
