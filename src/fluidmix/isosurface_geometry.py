"""Pure geometry checks for generated PhysX isosurface evidence."""

from __future__ import annotations

from collections.abc import Sequence
import math


_AREA_EPSILON_M2 = 1.0e-18


def analyze_triangle_mesh_geometry(
    points: Sequence[Sequence[float]],
    face_vertex_counts: Sequence[int],
    face_vertex_indices: Sequence[int],
) -> dict[str, object]:
    """Return JSON-safe evidence for whether mesh bytes form a real surface."""

    point_rows: list[tuple[float, float, float]] = []
    point_shape_valid = True
    for row in points:
        if len(row) != 3:
            point_shape_valid = False
            continue
        point_rows.append(tuple(float(value) for value in row))
    counts = [int(value) for value in face_vertex_counts]
    indices = [int(value) for value in face_vertex_indices]

    finite_points = bool(point_shape_valid) and all(
        math.isfinite(value) for row in point_rows for value in row
    )
    distinct_points = len(set(point_rows)) if finite_points else 0
    topology_consistent = bool(
        all(count >= 3 for count in counts) and sum(counts) == len(indices)
    )
    indices_valid = all(0 <= index < len(point_rows) for index in indices)

    if point_rows and finite_points:
        extent = [
            max(row[axis] for row in point_rows)
            - min(row[axis] for row in point_rows)
            for axis in range(3)
        ]
    else:
        extent = [None, None, None]
    extent_nonempty = bool(
        finite_points
        and all(value is not None and value > 0.0 for value in extent)
    )

    nonzero_area_faces = 0
    if topology_consistent and indices_valid and finite_points:
        cursor = 0
        for count in counts:
            face = indices[cursor : cursor + count]
            cursor += count
            anchor = point_rows[face[0]]
            face_has_area = False
            for offset in range(1, len(face) - 1):
                left = point_rows[face[offset]]
                right = point_rows[face[offset + 1]]
                ab = tuple(left[axis] - anchor[axis] for axis in range(3))
                ac = tuple(right[axis] - anchor[axis] for axis in range(3))
                cross = (
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                )
                if math.sqrt(sum(value * value for value in cross)) > _AREA_EPSILON_M2:
                    face_has_area = True
                    break
            nonzero_area_faces += int(face_has_area)
    all_faces_nonzero_area = bool(
        counts and nonzero_area_faces == len(counts)
    )

    failures: list[str] = []
    if not finite_points:
        failures.append("nonfinite_points")
    if distinct_points < 3:
        failures.append("fewer_than_three_distinct_points")
    if not topology_consistent:
        failures.append("topology_corner_count_mismatch")
    if not indices_valid:
        failures.append("invalid_face_vertex_index")
    if not all_faces_nonzero_area:
        failures.append("zero_area_face")
    if not extent_nonempty:
        failures.append("empty_three_dimensional_extent")

    return {
        "mesh_point_count": len(point_rows),
        "mesh_distinct_point_count": distinct_points,
        "mesh_face_count": len(counts),
        "mesh_corner_count": len(indices),
        "mesh_finite_points": finite_points,
        "mesh_topology_consistent": topology_consistent,
        "mesh_indices_valid": indices_valid,
        "mesh_nonzero_area_face_count": nonzero_area_faces,
        "mesh_all_faces_nonzero_area": all_faces_nonzero_area,
        "mesh_extent_m": extent,
        "mesh_extent_nonempty": extent_nonempty,
        "mesh_geometry_valid": not failures,
        "mesh_geometry_failures": failures,
    }
