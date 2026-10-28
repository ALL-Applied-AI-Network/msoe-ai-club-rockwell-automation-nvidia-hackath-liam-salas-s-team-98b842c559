"""CPU contracts for substantive generated-isosurface geometry evidence."""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fluidmix.isosurface_geometry import analyze_triangle_mesh_geometry


def test_tetrahedron_is_substantive_three_dimensional_surface():
    evidence = analyze_triangle_mesh_geometry(
        points=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        face_vertex_counts=[3, 3, 3, 3],
        face_vertex_indices=[0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3],
    )

    assert evidence == {
        "mesh_point_count": 4,
        "mesh_distinct_point_count": 4,
        "mesh_face_count": 4,
        "mesh_corner_count": 12,
        "mesh_finite_points": True,
        "mesh_topology_consistent": True,
        "mesh_indices_valid": True,
        "mesh_nonzero_area_face_count": 4,
        "mesh_all_faces_nonzero_area": True,
        "mesh_extent_m": [1.0, 1.0, 1.0],
        "mesh_extent_nonempty": True,
        "mesh_geometry_valid": True,
        "mesh_geometry_failures": [],
    }


def test_one_point_self_indexed_placeholder_is_rejected():
    evidence = analyze_triangle_mesh_geometry(
        points=[(0.0, 0.0, 0.0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 0, 0],
    )

    assert evidence["mesh_point_count"] == 1
    assert evidence["mesh_distinct_point_count"] == 1
    assert evidence["mesh_nonzero_area_face_count"] == 0
    assert evidence["mesh_extent_m"] == [0.0, 0.0, 0.0]
    assert evidence["mesh_geometry_valid"] is False
    assert evidence["mesh_geometry_failures"] == [
        "fewer_than_three_distinct_points",
        "zero_area_face",
        "empty_three_dimensional_extent",
    ]


def test_invalid_index_and_nonfinite_point_fail_closed():
    evidence = analyze_triangle_mesh_geometry(
        points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, math.nan, 1.0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 1, 9],
    )

    assert evidence["mesh_finite_points"] is False
    assert evidence["mesh_indices_valid"] is False
    assert evidence["mesh_geometry_valid"] is False
    assert "nonfinite_points" in evidence["mesh_geometry_failures"]
    assert "invalid_face_vertex_index" in evidence["mesh_geometry_failures"]


def test_corner_count_must_match_face_counts():
    evidence = analyze_triangle_mesh_geometry(
        points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0)],
        face_vertex_counts=[3],
        face_vertex_indices=[0, 1],
    )

    assert evidence["mesh_topology_consistent"] is False
    assert evidence["mesh_geometry_valid"] is False
    assert "topology_corner_count_mismatch" in evidence["mesh_geometry_failures"]
