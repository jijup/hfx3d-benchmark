"""Simple Geometry-Guided Functional Refinement rules.

Non-learning pipeline. For every instance I and every applicable attribute a:

    P = semantic prior (purity-adjusted, neutral 0.5)
    G = geometry evidence
    C = graph-context evidence
    confidence = odds_fusion(P, G, C)

All evidence is continuous, in [0,1], with 0.5 = neutral/unknown. Missing
evidence (no neighbor found, too few points, unreliable measurement) returns
0.5 and is never treated as strong negative evidence. There are no
per-class/per-attribute fusion weights, no binary thresholds, and no
condition-specific threshold banks. A small set of shared continuous
normalization scales lives in ``scoring``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial import cKDTree

from hfx3d_core import (
    AssignmentResult,
    InstanceRecord,
    LabelMap,
    Ontology,
    clip01,
    deterministic_direction,
    extract_instances,
    validate_score_arrays,
)
from scoring import (
    NEUTRAL,
    angle_inverse_ramp,
    angle_ramp,
    availability_aware_geometric_mean,
    availability_aware_max_merge,
    building_scale_normalize,
    clip01 as _clip01,
    geometric_mean,
    odds_fusion,
    overlap_min_extent,
    proximity,
    rank_01,
    vertical_gap_score,
)

# ---------------------------------------------------------------------------
# Local coordinate frame
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalFrame:
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    z_axis: np.ndarray
    ground_z: float
    source: str

    @property
    def rotation(self) -> np.ndarray:
        return np.stack([self.u_axis, self.v_axis, self.z_axis], axis=1)

    def project(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - self.origin) @ self.rotation


def estimate_local_frame(instances: list[InstanceRecord], config: dict[str, Any]) -> LocalFrame:
    if not instances:
        raise ValueError("Cannot estimate a local frame without valid instances")
    frame_config = config["local_frame"]
    facade_class = str(frame_config["facade_reference_class"])
    candidates = [record.sample_points for record in instances if record.semantic_class == facade_class and len(record.sample_points)]
    source = facade_class
    if not candidates:
        candidates = [record.sample_points for record in instances if len(record.sample_points)]
        source = "all_valid_instances_fallback"
    points = np.concatenate(candidates, axis=0)
    maximum = int(frame_config["max_reference_points"])
    if len(points) > maximum:
        points = points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64)]
    origin = np.median(points, axis=0)
    centered_xy = points[:, :2] - origin[:2]
    covariance = centered_xy.T @ centered_xy / max(len(centered_xy) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    horizontal = deterministic_direction(np.r_[eigenvectors[:, np.argmax(eigenvalues)], 0.0])
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    depth = np.cross(z_axis, horizontal)
    horizontal = deterministic_direction(np.cross(depth, z_axis))
    depth = np.cross(z_axis, horizontal)
    all_z = np.concatenate([record.sample_points[:, 2] for record in instances if len(record.sample_points)])
    ground_z = float(np.percentile(all_z, float(frame_config["ground_percentile"])))
    return LocalFrame(origin, horizontal, depth, z_axis, ground_z, source)


# ---------------------------------------------------------------------------
# Geometry descriptors (base, reused infra)
# ---------------------------------------------------------------------------


def fit_plane_pca(points: np.ndarray, epsilon: float) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return {
            "normal": np.array([0.0, 0.0, 1.0]),
            "centroid": points.mean(axis=0) if len(points) else np.zeros(3),
            "residual": 0.0,
            "eigenvalues": np.zeros(3),
            "directions": np.eye(3),
            "status": "insufficient_points",
        }
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    directions = eigenvectors[:, order]
    normal = deterministic_direction(directions[:, -1], epsilon)
    residual = float(np.sqrt(np.mean(np.square(centered @ normal))))
    return {
        "normal": normal,
        "centroid": centroid,
        "residual": residual,
        "eigenvalues": eigenvalues,
        "directions": directions,
        "status": "ok",
    }


def local_surface_statistics(points: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    plane_config = config["plane_extraction"]
    minimum = int(plane_config["minimum_neighbors"])
    if len(points) < minimum:
        return {
            "local_curvature": 0.0,
            "local_normal_dispersion": 0.0,
            "local_plane_angle_p90_degrees": 0.0,
            "local_plane_count_proxy": 1.0,
        }
    maximum_queries = int(plane_config["max_normal_queries"])
    query_indices = np.linspace(0, len(points) - 1, min(maximum_queries, len(points)), dtype=np.int64)
    query = points[query_indices]
    tree = cKDTree(points)
    k = min(int(plane_config["normal_neighbors"]), len(points))
    _, neighbor_indices = tree.query(query, k=k)
    normals: list[np.ndarray] = []
    curvatures: list[float] = []
    epsilon = float(config["numerical"]["epsilon"])
    for indices in np.atleast_2d(neighbor_indices):
        local = points[indices]
        fit = fit_plane_pca(local, epsilon)
        eigenvalues = fit["eigenvalues"]
        curvatures.append(float(eigenvalues[-1] / max(float(eigenvalues.sum()), epsilon)))
        normal = fit["normal"]
        if normals and np.dot(normal, normals[0]) < 0:
            normal = -normal
        normals.append(normal)
    normal_array = np.asarray(normals)
    reference = deterministic_direction(normal_array.mean(axis=0), epsilon)
    angles = np.degrees(np.arccos(np.clip(np.abs(normal_array @ reference), 0.0, 1.0)))
    dispersion = float(np.mean(np.linalg.norm(normal_array - reference, axis=1)))
    angle_threshold = float(plane_config["distinct_plane_angle_degrees"])
    plane_count_proxy = 1.0 + float(np.count_nonzero(angles > angle_threshold) > len(angles) * 0.15)
    return {
        "local_curvature": float(np.median(curvatures)),
        "local_normal_dispersion": dispersion,
        "local_plane_angle_p90_degrees": float(np.percentile(angles, 90)),
        "local_plane_count_proxy": plane_count_proxy,
    }


def _estimated_spacing(points: np.ndarray, maximum: int, epsilon: float) -> float:
    if len(points) < 2:
        return 0.0
    subset = points[np.linspace(0, len(points) - 1, min(maximum, len(points)), dtype=np.int64)]
    distances, _ = cKDTree(subset).query(subset, k=2)
    return float(np.median(np.maximum(distances[:, 1], epsilon)))


def _convex_hull_features(points_2d: np.ndarray, epsilon: float) -> tuple[float, int]:
    if len(points_2d) < 3:
        return 0.0, len(points_2d)
    try:
        from scipy.spatial import ConvexHull, QhullError
        hull = ConvexHull(points_2d)
        return float(hull.volume), int(len(hull.vertices))
    except Exception:
        extents = np.ptp(points_2d, axis=0)
        return float(np.prod(np.maximum(extents, epsilon))), 0


def _top_surface(local: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    geometry_config = config["geometry"]
    epsilon = float(config["numerical"]["epsilon"])
    empty = {
        "top_surface_reliable": False,
        "top_surface_slope_degrees": 0.0,
        "top_surface_fit_score": 0.0,
        "top_surface_boundary_score": 0.0,
        "top_surface_direction_u": 0.0,
        "top_surface_direction_v": 0.0,
    }
    if len(local) < int(geometry_config["minimum_top_surface_points"]):
        return empty
    cutoff = np.percentile(local[:, 2], float(geometry_config["top_surface_percentile"]))
    top = local[local[:, 2] >= cutoff]
    if len(top) < 3:
        return empty
    design = np.c_[top[:, 0], top[:, 1], np.ones(len(top))]
    coefficients, _, _, _ = np.linalg.lstsq(design, top[:, 2], rcond=None)
    predicted = design @ coefficients
    residual = float(np.sqrt(np.mean(np.square(top[:, 2] - predicted))))
    spread = float(np.std(top[:, 2]))
    fit_score = float(clip01(1.0 - residual / max(spread, epsilon)))
    gradient = coefficients[:2]
    magnitude = float(np.linalg.norm(gradient))
    downhill = -gradient / max(magnitude, epsilon)
    slope_degrees = float(np.degrees(np.arctan(magnitude)))
    low = top[top[:, 2] <= np.percentile(top[:, 2], float(geometry_config["top_low_percentile"]))]
    bounds_min = local[:, :2].min(axis=0)
    bounds_max = local[:, :2].max(axis=0)
    span = np.maximum(bounds_max - bounds_min, epsilon)
    normalized = (low[:, :2] - bounds_min) / span
    edge_distance = np.min(np.c_[normalized, 1.0 - normalized], axis=1)
    boundary_score = float(clip01(1.0 - np.median(edge_distance) / float(geometry_config["boundary_band_fraction"])))
    return {
        "top_surface_reliable": True,
        "top_surface_slope_degrees": slope_degrees,
        "top_surface_fit_score": fit_score,
        "top_surface_boundary_score": boundary_score,
        "top_surface_direction_u": float(downhill[0]),
        "top_surface_direction_v": float(downhill[1]),
    }


def _shape_proxy(local: np.ndarray, epsilon: float) -> dict[str, float]:
    if len(local) < 10:
        return {"enlarged_top_ratio": 1.0, "opening_score": 0.0, "open_boundary_score": 0.5}
    z_min, z_max = np.min(local[:, 2]), np.max(local[:, 2])
    z_span = max(float(z_max - z_min), epsilon)
    normalized_z = (local[:, 2] - z_min) / z_span

    def area_proxy(mask: np.ndarray) -> float:
        selection = local[mask, :2]
        if len(selection) < 3:
            return epsilon
        return float(np.prod(np.maximum(np.ptp(selection, axis=0), epsilon)))

    top_area = area_proxy(normalized_z >= 0.8)
    middle_area = area_proxy((normalized_z >= 0.35) & (normalized_z <= 0.65))
    enlarged_top_ratio = float(top_area / max(middle_area, epsilon))
    u, z = local[:, 0], local[:, 2]
    u_span = max(float(np.ptp(u)), epsilon)
    grid_size = 24
    ui = np.clip(((u - u.min()) / u_span * (grid_size - 1)).astype(int), 0, grid_size - 1)
    zi = np.clip(((z - z_min) / z_span * (grid_size - 1)).astype(int), 0, grid_size - 1)
    occupancy = np.zeros((grid_size, grid_size), dtype=bool)
    occupancy[ui, zi] = True
    interior = occupancy[7:17, 2:15].mean()
    sides = np.r_[occupancy[2:7, 2:18].ravel(), occupancy[17:22, 2:18].ravel()]
    crown = occupancy[5:19, 15:23]
    opening_score = float(clip01((0.5 * sides.mean() + 0.5 * crown.mean()) * (1.0 - interior)))
    edge = (ui <= 2) | (ui >= grid_size - 3)
    center = (ui >= 8) & (ui <= 15)
    edge_vertical = float(np.ptp(z[edge]) / z_span) if np.any(edge) else 0.0
    center_vertical = float(np.ptp(z[center]) / z_span) if np.any(center) else 0.0
    return {
        "enlarged_top_ratio": enlarged_top_ratio,
        "opening_score": opening_score,
        "open_boundary_score": float(clip01(1.0 - min(edge_vertical, center_vertical))),
    }


@dataclass
class InstanceGeometry:
    instance_id: int
    features: dict[str, Any]


def compute_instance_geometry(record: InstanceRecord, frame: LocalFrame, config: dict[str, Any]) -> InstanceGeometry:
    points = record.sample_points
    epsilon = float(config["numerical"]["epsilon"])
    geometry_config = config["geometry"]
    if not len(points):
        raise ValueError(f"Instance {record.instance_id} has no deterministic geometry sample")
    local = frame.project(points)
    plane = fit_plane_pca(points, epsilon)
    eigenvalues = plane["eigenvalues"]
    directions = plane["directions"]
    l1, l2, l3 = (float(v) for v in eigenvalues)
    normal = plane["normal"]
    local_min, local_max = local.min(axis=0), local.max(axis=0)
    local_extent = local_max - local_min
    footprint_area, boundary_count = _convex_hull_features(local[:, :2], epsilon)
    facade_area, facade_boundary_count = _convex_hull_features(local[:, [0, 2]], epsilon)
    spacing = _estimated_spacing(points, int(geometry_config["spacing_sample_points"]), epsilon)
    depth_quantiles = np.percentile(local[:, 1], [5, 25, 50, 75, 95])
    depth_iqr = float(depth_quantiles[3] - depth_quantiles[1])
    depth_range = max(float(depth_quantiles[4] - depth_quantiles[0]), epsilon)
    obb_local = (points - plane["centroid"]) @ directions
    features: dict[str, Any] = {
        "point_count": record.point_count,
        "centroid": record.centroid,
        "xyz_min": record.xyz_min,
        "xyz_max": record.xyz_max,
        "aabb_extent": record.xyz_max - record.xyz_min,
        "obb_center": plane["centroid"],
        "obb_extents": np.ptp(obb_local, axis=0),
        "obb_directions": directions,
        "width": float(local_extent[0]),
        "depth": float(local_extent[1]),
        "height": float(local_extent[2]),
        "diagonal": float(np.linalg.norm(record.xyz_max - record.xyz_min)),
        "pca_eigenvalues": eigenvalues,
        "pca_principal_directions": directions,
        "linearity": (l1 - l2) / max(l1, epsilon),
        "planarity": (l2 - l3) / max(l1, epsilon),
        "scattering": l3 / max(l1, epsilon),
        "verticality": float(1.0 - abs(float(normal[2]))),
        "dominant_plane_normal": normal,
        "plane_fit_reliable": plane["status"] == "ok",
        "plane_fitting_residual": plane["residual"],
        "top_elevation": float(record.xyz_max[2]),
        "bottom_elevation": float(record.xyz_min[2]),
        "height_above_estimated_ground": float(record.xyz_min[2] - frame.ground_z),
        "horizontal_extent": float(np.linalg.norm(local_extent[:2])),
        "vertical_extent": float(local_extent[2]),
        "surface_slope_degrees": float(np.degrees(np.arccos(np.clip(abs(float(normal[2])), 0.0, 1.0)))),
        "projected_footprint_area": footprint_area,
        "projected_facade_area": facade_area,
        "projected_extent": local_extent[[0, 2]],
        "boundary_point_count": boundary_count,
        "facade_boundary_point_count": facade_boundary_count,
        "depth_distribution": depth_quantiles,
        "depth_iqr": depth_iqr,
        "multi_depth_layer_score": float(clip01(depth_iqr / depth_range)),
        "point_density": float(record.point_count / max(footprint_area, facade_area, epsilon)),
        "estimated_point_spacing": spacing,
        "local_bounds_min": local_min,
        "local_bounds_max": local_max,
        "touches_ground": bool(record.xyz_min[2] <= frame.ground_z + float(config["scoring"]["ground_gap_spacing_multiplier"]) * max(spacing, epsilon)),
        "sample_point_count": len(points),
    }
    features.update(local_surface_statistics(points, config))
    features.update(_top_surface(local, config))
    features.update(_shape_proxy(local, epsilon))
    return InstanceGeometry(record.instance_id, features)


def compute_all_geometry(records: list[InstanceRecord], frame: LocalFrame, config: dict[str, Any]) -> dict[int, InstanceGeometry]:
    return {record.instance_id: compute_instance_geometry(record, frame, config) for record in records}


# ---------------------------------------------------------------------------
# Instance graph (base, reused infra)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphEdge:
    source_instance_id: int
    target_instance_id: int
    target_semantic_class: str
    metrics: dict[str, Any]


@dataclass
class InstanceGraph:
    neighbors: dict[int, list[GraphEdge]]

    def edges_from(self, instance_id: int, semantic_classes: set[str] | None = None) -> list[GraphEdge]:
        edges = self.neighbors.get(instance_id, [])
        if semantic_classes is None:
            return edges
        return [edge for edge in edges if edge.target_semantic_class in semantic_classes]


def _intersection(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, min(left_max, right_max) - max(left_min, right_min))


def _gap(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, max(left_min, right_min) - min(left_max, right_max))


def _directed_metrics(
    source: InstanceRecord,
    target: InstanceRecord,
    source_geometry: InstanceGeometry,
    target_geometry: InstanceGeometry,
    config: dict[str, Any],
    sampled_minimum_distance: float | None = None,
) -> dict[str, Any]:
    epsilon = float(config["numerical"]["epsilon"])
    source_min = np.asarray(source_geometry.features["local_bounds_min"])
    source_max = np.asarray(source_geometry.features["local_bounds_max"])
    target_min = np.asarray(target_geometry.features["local_bounds_min"])
    target_max = np.asarray(target_geometry.features["local_bounds_max"])
    source_extent = np.maximum(source_max - source_min, epsilon)
    target_extent = np.maximum(target_max - target_min, epsilon)
    overlaps = np.asarray([
        _intersection(source_min[axis], source_max[axis], target_min[axis], target_max[axis])
        for axis in range(3)
    ])
    gaps = np.asarray([
        _gap(source_min[axis], source_max[axis], target_min[axis], target_max[axis])
        for axis in range(3)
    ])
    aabb_minimum_distance = float(np.linalg.norm(gaps))
    minimum_distance = aabb_minimum_distance if sampled_minimum_distance is None else float(sampled_minimum_distance)
    centroid_distance = float(np.linalg.norm(source.centroid - target.centroid))
    source_facade_area = float(source_extent[0] * source_extent[2])
    target_facade_area = float(target_extent[0] * target_extent[2])
    facade_intersection_area = float(overlaps[0] * overlaps[2])
    horizontal_overlap = float(overlaps[0] / min(source_extent[0], target_extent[0]))
    vertical_overlap = float(overlaps[2] / min(source_extent[2], target_extent[2]))
    depth_overlap = float(overlaps[1] / min(source_extent[1], target_extent[1]))
    facade_overlap_source = float(facade_intersection_area / max(source_facade_area, epsilon))
    facade_overlap_target = float(facade_intersection_area / max(target_facade_area, epsilon))
    relation_tolerance = 0.0
    above = bool(source_min[2] >= target_max[2] - relation_tolerance)
    below = bool(source_max[2] <= target_min[2] + relation_tolerance)
    source_normal = np.asarray(source_geometry.features["dominant_plane_normal"])
    target_normal = np.asarray(target_geometry.features["dominant_plane_normal"])
    plane_parallelism = float(abs(np.dot(source_normal, target_normal)))
    return {
        "minimum_distance": minimum_distance,
        "aabb_minimum_distance": aabb_minimum_distance,
        "centroid_distance": centroid_distance,
        "horizontal_overlap": horizontal_overlap,
        "vertical_overlap": vertical_overlap,
        "depth_overlap": depth_overlap,
        "facade_projection_overlap": facade_overlap_source,
        "facade_projection_overlap_target": facade_overlap_target,
        "above": above,
        "below": below,
        "plane_parallelism": plane_parallelism,
    }


def build_instance_graph(
    records: list[InstanceRecord],
    geometry: dict[int, InstanceGeometry],
    config: dict[str, Any],
) -> InstanceGraph:
    neighbors = {record.instance_id: [] for record in records}
    if len(records) < 2:
        return InstanceGraph(neighbors)
    graph_config = config["instance_graph"]
    distance_maximum = int(graph_config["distance_sample_points"])
    sample_clouds = []
    sample_trees = []
    for record in records:
        points = record.sample_points
        if len(points) > distance_maximum:
            points = points[np.linspace(0, len(points) - 1, distance_maximum, dtype=np.int64)]
        sample_clouds.append(points)
        sample_trees.append(cKDTree(points))

    def sampled_distance(left_index: int, right_index: int) -> float:
        left_to_right = sample_trees[right_index].query(sample_clouds[left_index], k=1)[0]
        right_to_left = sample_trees[left_index].query(sample_clouds[right_index], k=1)[0]
        return float(min(np.min(left_to_right), np.min(right_to_left)))

    centroids = np.stack([record.centroid for record in records])
    centroid_distances = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    np.fill_diagonal(centroid_distances, np.inf)
    nearest_k = min(int(graph_config["centroid_nearest_k"]), len(records) - 1)
    nearest_sets = [set(np.argpartition(centroid_distances[row], nearest_k - 1)[:nearest_k]) for row in range(len(records))]
    candidate_distance = float(graph_config["candidate_minimum_distance"])
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            left, right = records[left_index], records[right_index]
            left_metrics = _directed_metrics(left, right, geometry[left.instance_id], geometry[right.instance_id], config)
            is_nearest = right_index in nearest_sets[left_index] or left_index in nearest_sets[right_index]
            if left_metrics["minimum_distance"] > candidate_distance and not is_nearest:
                continue
            point_distance = sampled_distance(left_index, right_index)
            left_metrics = _directed_metrics(left, right, geometry[left.instance_id], geometry[right.instance_id], config, point_distance)
            right_metrics = _directed_metrics(right, left, geometry[right.instance_id], geometry[left.instance_id], config, point_distance)
            neighbors[left.instance_id].append(GraphEdge(left.instance_id, right.instance_id, right.semantic_class, left_metrics))
            neighbors[right.instance_id].append(GraphEdge(right.instance_id, left.instance_id, left.semantic_class, right_metrics))
    for edges in neighbors.values():
        edges.sort(key=lambda edge: (edge.metrics["minimum_distance"], edge.target_instance_id))
    return InstanceGraph(neighbors)


# ---------------------------------------------------------------------------
# Semantic prior (continuous purity adjustment toward neutral 0.5)
# ---------------------------------------------------------------------------


def semantic_prior(semantic_class: str, attribute: str, semantic_purity: float, config: dict[str, Any]) -> float:
    base = float(config["semantic_priors"]["by_class"][semantic_class][attribute])
    return float(_clip01(0.5 + semantic_purity * (base - 0.5)))


# ---------------------------------------------------------------------------
# Building context
# ---------------------------------------------------------------------------


@dataclass
class BuildingContext:
    building_id: str
    frame: LocalFrame
    graph: InstanceGraph
    geometry: dict[int, InstanceGeometry]
    records_by_id: dict[int, InstanceRecord]
    class_rank_pools: dict[str, dict[str, list[float]]]
    building_height: float
    structural_base_z: float
    building_local_min: np.ndarray
    building_local_max: np.ndarray
    boundary_tree: tuple[Any, np.ndarray] | None
    config: dict[str, Any]
    per_instance_results: dict[int, dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class EvaluatorResult:
    geometry_score: float
    context_score: float
    evidence: dict[str, Any]
    evaluator_status: str
    informative: bool


Evaluator = Callable[[InstanceRecord, InstanceGeometry, BuildingContext, dict[str, Any]], EvaluatorResult]


def _role(config: dict[str, Any], name: str) -> set[str]:
    return set(config["semantic_roles"][name])


def _pair_spacing(ctx: BuildingContext, current_id: int, edge: GraphEdge) -> float:
    return max(
        float(ctx.geometry[current_id].features["estimated_point_spacing"]),
        float(ctx.geometry[edge.target_instance_id].features["estimated_point_spacing"]),
        1e-9,
    )


def _prox(ctx: BuildingContext, current_id: int, edge: GraphEdge) -> float:
    eps = float(ctx.config["numerical"]["epsilon"])
    return proximity(edge.metrics["minimum_distance"], _pair_spacing(ctx, current_id, edge), eps)


def _xy_overlap(ctx: BuildingContext, current_id: int, edge: GraphEdge) -> float:
    eps = float(ctx.config["numerical"]["epsilon"])
    src_min = ctx.geometry[current_id].features["local_bounds_min"]
    src_max = ctx.geometry[current_id].features["local_bounds_max"]
    tgt_min = ctx.geometry[edge.target_instance_id].features["local_bounds_min"]
    tgt_max = ctx.geometry[edge.target_instance_id].features["local_bounds_max"]
    return overlap_min_extent(src_min, src_max, tgt_min, tgt_max, (0, 1), eps)


def _best_composite(edges: list[GraphEdge], scorer: Callable[[GraphEdge], float]) -> tuple[float, GraphEdge | None]:
    """Best candidate evidence over measured edges.

    Distinguishes the two cases explicitly:
    * no candidate at all (no edges / nothing measurable) -> neutral 0.5
    * candidates exist and the relation was successfully measured -> the
      actual best score. A weak measured relation (e.g. 0.25) is negative
      contextual evidence, not unknown, and is therefore preserved.
    """
    if not edges:
        return NEUTRAL, None
    best = 0.0
    best_edge: GraphEdge | None = None
    for edge in edges:
        score = float(scorer(edge))
        if best_edge is None or score > best:
            best = score
            best_edge = edge
    return best, best_edge


def _vertical_direction_score(current_center_z: float, target_center_z: float, scale: float) -> float:
    """Continuous direction of the target relative to the current instance.

    target above current -> >0.5 ; same level -> 0.5 ; below -> <0.5.
    scale is the vertical size used to normalize the offset.
    """
    return float(_clip01(0.5 + (target_center_z - current_center_z) / max(scale, 1e-9)))


def _related(edges: list[GraphEdge]) -> tuple[list[int], list[str]]:
    return [edge.target_instance_id for edge in edges], [edge.target_semantic_class for edge in edges]


def _result(
    geometry: float,
    context: float,
    evidence: dict[str, Any],
    status: str,
    informative: bool,
) -> EvaluatorResult:
    """Return measured evidence or an explicit neutral/unknown result."""
    if not informative:
        geometry = NEUTRAL
        context = NEUTRAL
    return EvaluatorResult(
        float(_clip01(geometry)), float(_clip01(context)), evidence, status, bool(informative)
    )


def _geometry_available(geom: InstanceGeometry, config: dict[str, Any], *, plane: bool = False) -> bool:
    """Shared conservative availability check for instance measurements."""
    minimum = int(config["plane_extraction"]["minimum_neighbors"])
    enough_points = int(geom.features.get("sample_point_count", 0)) >= minimum
    spacing = float(geom.features.get("estimated_point_spacing", 0.0))
    finite_spacing = np.isfinite(spacing) and spacing > float(config["numerical"]["epsilon"])
    plane_ok = (not plane) or bool(geom.features.get("plane_fit_reliable", False))
    return bool(enough_points and finite_spacing and plane_ok)


def _base_evidence(record: InstanceRecord, ctx: BuildingContext, measured: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "measured_values": measured,
        "reason": reason,
        "related_instance_ids": [],
        "related_semantic_classes": [],
    }


# ---------------------------------------------------------------------------
# Shared continuous evidence helpers
# ---------------------------------------------------------------------------


def _ground_continuity(geom: InstanceGeometry, ctx: BuildingContext) -> float:
    spacing = max(float(geom.features["estimated_point_spacing"]), 1e-9)
    spacing_scale = spacing * float(ctx.config["scoring"]["ground_gap_spacing_multiplier"])
    building_scale = ctx.building_height * float(ctx.config["scoring"]["ground_gap_building_fraction"])
    scale = max(spacing_scale, building_scale, 1e-9)
    bottom = float(geom.features["xyz_min"][2])
    return float(_clip01(1.0 - max(0.0, bottom - ctx.structural_base_z) / scale))


def _depth_variation(geom: InstanceGeometry, ctx: BuildingContext) -> float:
    """Relative depth variation: percentile rank within the same building/class."""
    semantic_class = ctx.records_by_id[geom.instance_id].semantic_class
    values = ctx.class_rank_pools.get(semantic_class, {}).get("depth_iqr", [])
    return rank_01(float(geom.features["depth_iqr"]), values)


def _articulation(geom: InstanceGeometry, ctx: BuildingContext) -> float:
    """Relative surface articulation: percentile rank of normal dispersion in class."""
    semantic_class = ctx.records_by_id[geom.instance_id].semantic_class
    values = ctx.class_rank_pools.get(semantic_class, {}).get("normal_dispersion", [])
    return rank_01(float(geom.features["local_normal_dispersion"]), values)


def _host_wall_association(ctx: BuildingContext, current_id: int) -> tuple[float, GraphEdge | None]:
    edges = ctx.graph.edges_from(current_id, _role(ctx.config, "host_surface"))
    return _best_composite(edges, lambda edge: geometric_mean([
        _prox(ctx, current_id, edge),
        float(edge.metrics["facade_projection_overlap"]),
        float(edge.metrics["plane_parallelism"]),
    ]))


def _vegetation_side_score(ctx: BuildingContext, current_id: int, edge: GraphEdge) -> float:
    host_geom = ctx.geometry[current_id].features
    veg_geom = ctx.geometry[edge.target_instance_id].features
    host_bottom = float(host_geom["local_bounds_min"][2])
    host_height = max(float(host_geom["height"]), 1e-9)
    veg_bottom = float(veg_geom["local_bounds_min"][2])
    top_alignment = _clip01((veg_bottom - host_bottom) / host_height)
    side_separation = 1.0 - top_alignment
    return geometric_mean([
        _prox(ctx, current_id, edge),
        float(edge.metrics["facade_projection_overlap"]),
        float(edge.metrics["vertical_overlap"]),
        side_separation,
    ])


def _vegetation_top_score(ctx: BuildingContext, current_id: int, edge: GraphEdge) -> float:
    host_geom = ctx.geometry[current_id].features
    veg_geom = ctx.geometry[edge.target_instance_id].features
    host_top = float(host_geom["local_bounds_max"][2])
    host_bottom = float(host_geom["local_bounds_min"][2])
    host_height = max(float(host_geom["height"]), 1e-9)
    veg_bottom = float(veg_geom["local_bounds_min"][2])
    gap = max(0.0, veg_bottom - host_top)
    eps = float(ctx.config["numerical"]["epsilon"])
    top_alignment = _clip01((veg_bottom - host_bottom) / host_height)
    return geometric_mean([
        vertical_gap_score(gap, _pair_spacing(ctx, current_id, edge), eps),
        _xy_overlap(ctx, current_id, edge),
        top_alignment,
    ])


def _vegetation_context(ctx: BuildingContext, current_id: int, side: bool) -> tuple[float, GraphEdge | None]:
    edges = ctx.graph.edges_from(current_id, _role(ctx.config, "vegetation"))
    scorer = _vegetation_side_score if side else _vegetation_top_score
    return _best_composite(edges, lambda edge: scorer(ctx, current_id, edge))


def _upper_support_context(ctx: BuildingContext, current_id: int) -> tuple[float, GraphEdge | None]:
    """Supporting structure located ABOVE the current instance (load path).

    Directional relation: the candidate must be physically above the current
    instance (target center above current center). Evidence combines the
    vertical direction, the vertical gap between current top and target
    bottom, the XY footprint overlap and the surface proximity. A target
    below the current instance yields a low (measured negative) score, not a
    neutral one.
    """
    edges = ctx.graph.edges_from(current_id, _role(ctx.config, "upper_building"))

    def scorer(edge: GraphEdge) -> float:
        cur = ctx.geometry[current_id].features
        tgt = ctx.geometry[edge.target_instance_id].features
        cur_top = float(cur["local_bounds_max"][2])
        cur_bottom = float(cur["local_bounds_min"][2])
        tgt_bottom = float(tgt["local_bounds_min"][2])
        cur_center_z = 0.5 * (cur_top + cur_bottom)
        tgt_center_z = 0.5 * (float(tgt["local_bounds_max"][2]) + tgt_bottom)
        direction = _vertical_direction_score(cur_center_z, tgt_center_z, float(cur["height"]))
        gap = max(0.0, tgt_bottom - cur_top)
        eps = float(ctx.config["numerical"]["epsilon"])
        gap_score = vertical_gap_score(gap, _pair_spacing(ctx, current_id, edge), eps)
        return geometric_mean([
            direction,
            gap_score,
            _xy_overlap(ctx, current_id, edge),
            _prox(ctx, current_id, edge),
        ])

    return _best_composite(edges, scorer)


def _lower_support_context(ctx: BuildingContext, current_id: int) -> tuple[float, GraphEdge | None]:
    """Support instance located BELOW the current one (roof/balcony load path).

    Directional relation: the candidate must be physically below the current
    instance (target center below current center). Evidence combines the
    vertical direction, the vertical gap between the current bottom and the
    target top, the XY footprint overlap and the surface proximity. A target
    above the current instance yields a low (measured negative) score.
    """
    edges = ctx.graph.edges_from(current_id, _role(ctx.config, "lower_support"))

    def scorer(edge: GraphEdge) -> float:
        cur = ctx.geometry[current_id].features
        tgt = ctx.geometry[edge.target_instance_id].features
        cur_bottom = float(cur["local_bounds_min"][2])
        cur_top = float(cur["local_bounds_max"][2])
        tgt_top = float(tgt["local_bounds_max"][2])
        cur_center_z = 0.5 * (cur_top + cur_bottom)
        tgt_center_z = 0.5 * (tgt_top + float(tgt["local_bounds_min"][2]))
        scale = max(float(tgt["height"]), float(cur["height"]))
        direction = _vertical_direction_score(tgt_center_z, cur_center_z, scale)
        gap = max(0.0, cur_bottom - tgt_top)
        eps = float(ctx.config["numerical"]["epsilon"])
        gap_score = vertical_gap_score(gap, _pair_spacing(ctx, current_id, edge), eps)
        return geometric_mean([
            direction,
            gap_score,
            _xy_overlap(ctx, current_id, edge),
            _prox(ctx, current_id, edge),
        ])

    return _best_composite(edges, scorer)


def _exposure_fraction(geom: InstanceGeometry, ctx: BuildingContext) -> float:
    """Fraction of the instance boundary not covered by another instance (open edge)."""
    if ctx.boundary_tree is None:
        return NEUTRAL
    tree, labels = ctx.boundary_tree
    record = ctx.records_by_id[geom.instance_id]
    points = record.sample_points
    if len(points) < 3:
        return NEUTRAL
    if len(points) > 2000:
        points = points[np.linspace(0, len(points) - 1, 2000, dtype=np.int64)]
    k = min(256, len(labels))
    distances, indices = tree.query(points, k=k)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    labels_k = labels[indices]
    mismatch = labels_k != geom.instance_id
    row_min = np.where(mismatch, distances, np.inf).min(axis=1)
    spacing = max(float(geom.features["estimated_point_spacing"]), 1e-9)
    radius = spacing * float(ctx.config["scoring"]["contact_radius_multiplier"])
    if not len(row_min):
        return NEUTRAL
    return float(_clip01(float(np.mean(row_min > radius))))


def _barrier_context(ctx: BuildingContext, current_id: int) -> tuple[float, GraphEdge | None]:
    """Aligned vertical barrier (wall/column/railing-like) along the current instance."""
    edges = ctx.graph.edges_from(current_id, _role(ctx.config, "barrier_proxy"))

    def scorer(edge: GraphEdge) -> float:
        target_verticality = float(ctx.geometry[edge.target_instance_id].features["verticality"])
        verticality_score = _clip01(target_verticality / max(float(ctx.config["scoring"]["barrier_min_verticality"]), 1e-9))
        return geometric_mean([
            _prox(ctx, current_id, edge),
            float(edge.metrics["vertical_overlap"]),
            verticality_score,
        ])

    return _best_composite(edges, scorer)


def _elevation_fraction(geom: InstanceGeometry, ctx: BuildingContext) -> float:
    bottom = float(geom.features["xyz_min"][2])
    return building_scale_normalize(max(0.0, bottom - ctx.structural_base_z), ctx.building_height, 1e-9)


def _protruding_clusters(
    points: np.ndarray,
    plane_center: np.ndarray,
    plane_normal: np.ndarray,
    spacing: float,
    cluster_radius: float,
    epsilon: float,
) -> list[dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 5:
        return []
    distances = np.abs((points - plane_center) @ plane_normal)
    noise = max(spacing, epsilon)
    protruding = points[distances > noise * 1.5]
    if len(protruding) < 3:
        return []
    tree = cKDTree(protruding)
    visited = np.zeros(len(protruding), dtype=bool)
    clusters: list[np.ndarray] = []
    for start in range(len(protruding)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        members: list[int] = []
        while stack:
            idx = stack.pop()
            members.append(idx)
            for neighbor in tree.query_ball_point(protruding[idx], cluster_radius):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        clusters.append(protruding[members])
    result: list[dict[str, Any]] = []
    for cluster in clusters:
        fit = fit_plane_pca(cluster, epsilon)
        extents = np.ptp(cluster, axis=0)
        centroid = cluster.mean(axis=0)
        outward = centroid - plane_center
        outward_norm = float(np.linalg.norm(outward))
        outward_direction = outward / outward_norm if outward_norm > epsilon else np.zeros(3)
        protrusion = float(np.median(np.abs((cluster - plane_center) @ plane_normal)))
        result.append({
            "point_count": int(len(cluster)),
            "extent": float(max(extents)),
            "extents": extents.tolist(),
            "centroid": centroid.tolist(),
            "normal": fit["normal"].tolist(),
            "outward_direction": outward_direction.tolist(),
            "protrusion": protrusion,
            "residual": float(fit["residual"]),
        })
    result.sort(key=lambda item: (-item["point_count"], item["extent"], item["centroid"][0]))
    return result


def _cluster_shape_scores(cluster: dict[str, Any], instance_diagonal: float) -> tuple[float, float]:
    """(smallness, compactness) of a protruding cluster relative to its host."""
    size_ratio = float(cluster["extent"]) / max(instance_diagonal, 1e-9)
    small = _clip01(1.0 - size_ratio)
    extents = np.asarray(cluster["extents"], dtype=np.float64)
    compact = _clip01(float(extents.min()) / max(float(extents.max()), 1e-9))
    return small, compact


def _window_operable_evidence(ctx: BuildingContext, current_id: int) -> dict[str, Any] | None:
    return ctx.per_instance_results.get(current_id, {}).get("operable")


# ---------------------------------------------------------------------------
# Condition evaluators
# ---------------------------------------------------------------------------


def evaluate_always(record, geom, ctx, config):
    evidence = _base_evidence(record, ctx, {}, "pure semantic prior; no geometry claim")
    return _result(NEUTRAL, NEUTRAL, evidence, "always_semantic_prior", False)


def evaluate_wall_ground_upper_support(record, geom, ctx, config):
    geometry_available = _geometry_available(geom, config)
    ground = _ground_continuity(geom, ctx) if geometry_available else NEUTRAL
    thickness = rank_01(
        float(geom.features["depth"]),
        ctx.class_rank_pools.get("wall", {}).get("depth", []),
    )
    # Primary geometry evidence is ground/base continuity; the relative wall
    # thickness is kept as a diagnostic (weak auxiliary cue) and does not
    # gate the fusion.
    geometry = ground
    context, best = _upper_support_context(ctx, record.instance_id)
    measured = {
        "touches_ground": bool(geom.features["touches_ground"]),
        "ground_continuity": ground,
        "thickness_rank": thickness,
        "wall_depth": float(geom.features["depth"]),
        "upper_support_score": context,
    }
    evidence = _base_evidence(record, ctx, measured, "wall grounded and carries upper structure")
    evidence["related_instance_ids"], evidence["related_semantic_classes"] = (
        ([best.target_instance_id], [best.target_semantic_class]) if best is not None else ([], [])
    )
    status = "ok" if best is not None else "no_upper_contact"
    informative = geometry_available or best is not None
    return _result(geometry, context, evidence, status, informative)


def evaluate_vegetation_wall_contact(record, geom, ctx, config):
    context, best = _vegetation_context(ctx, record.instance_id, side=True)
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "vegetation"))
    evidence = _base_evidence(record, ctx, {"side_contact_score": context},
                              "vegetation climbing / side-contact on wall")
    evidence["related_instance_ids"], evidence["related_semantic_classes"] = _related(edges if best is not None else [])
    return _result(NEUTRAL, context, evidence, "ok" if best is not None else "no_vegetation_neighbor", best is not None)


def evaluate_vegetation_wall_rests_on_top(record, geom, ctx, config):
    context, best = _vegetation_context(ctx, record.instance_id, side=False)
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "vegetation"))
    evidence = _base_evidence(record, ctx, {"rests_on_top_score": context},
                              "vegetation resting on wall upper surface")
    evidence["related_instance_ids"], evidence["related_semantic_classes"] = _related(edges if best is not None else [])
    return _result(NEUTRAL, context, evidence, "ok" if best is not None else "no_vegetation_neighbor", best is not None)


def evaluate_window_articulation(record, geom, ctx, config):
    reliable = _geometry_available(geom, config, plane=True)
    depth_variation = _depth_variation(geom, ctx)
    articulation = _articulation(geom, ctx)
    geometry = geometric_mean([depth_variation, articulation]) if reliable else NEUTRAL
    # Host-wall association is NOT operability evidence: a well-hosted window
    # may still be fixed. It is kept as a diagnostic in measured_values only.
    host, _ = _host_wall_association(ctx, record.instance_id)
    context = NEUTRAL
    measured = {
        "depth_variation": depth_variation,
        "articulation": articulation,
        "depth_iqr": geom.features["depth_iqr"],
        "normal_dispersion": geom.features["local_normal_dispersion"],
        "host_association": host,
    }
    evidence = _base_evidence(record, ctx, measured, "window surface articulation and depth variation")
    status = "ok" if reliable else "insufficient_or_unreliable_surface_geometry"
    return _result(geometry, context, evidence, status, reliable)


def evaluate_window_handle(record, geom, ctx, config):
    spacing = max(float(geom.features["estimated_point_spacing"]), 1e-9)
    radius = spacing * float(config["scoring"]["cluster_radius_multiplier"])
    clusters = _protruding_clusters(
        record.sample_points,
        np.asarray(geom.features["obb_center"]),
        np.asarray(geom.features["dominant_plane_normal"]),
        spacing,
        radius,
        float(config["numerical"]["epsilon"]),
    )
    minimum_points = int(config["scoring"]["handle_min_points"])
    reliable = [cluster for cluster in clusters if cluster["point_count"] >= minimum_points]
    candidate = None
    if reliable:
        diagonal = float(geom.features["diagonal"])
        scores = []
        for cluster in reliable:
            small, compact = _cluster_shape_scores(cluster, diagonal)
            protrusion = _clip01(cluster["protrusion"] / max(spacing * float(config["scoring"]["spacing_multiplier_protrusion"]), 1e-9))
            # The location of the cluster inside the window is a diagnostic
            # only (no assumption that handles sit at the edge or the center);
            # the handle evidence uses smallness, compactness and protrusion.
            total = geometric_mean([small, compact, protrusion])
            scores.append((total, cluster, small, compact, protrusion))
        scores.sort(key=lambda item: (-item[0], -item[1]["point_count"]))
        total, candidate, small, compact, protrusion = scores[0]
    else:
        total = NEUTRAL
    # A missing handle is neutral (occlusion), never negative; host-wall
    # association is a diagnostic only and does not support operability.
    context = NEUTRAL
    host, _ = _host_wall_association(ctx, record.instance_id)
    measured = {
        "candidate_detected": candidate is not None,
        "candidate_point_count": candidate["point_count"] if candidate is not None else 0,
        "candidate_relative_size": None if candidate is None else float(candidate["extent"]) / max(float(geom.features["diagonal"]), 1e-9),
        "candidate_protrusion": candidate["protrusion"] if candidate is not None else None,
        "candidate_compactness": None if candidate is None else float(np.asarray(candidate["extents"]).min()) / max(float(np.asarray(candidate["extents"]).max()), 1e-9),
        "candidate_location": None if candidate is None else candidate["centroid"],
        "cluster_radius": radius,
        "host_association": host,
    }
    evidence = _base_evidence(record, ctx, measured, "handle-like small protrusion proxy on window")
    evidence["handle_candidate"] = candidate
    status = "handle_candidate_found" if candidate is not None else "no_handle_candidate"
    return _result(total, context, evidence, status, candidate is not None)


def evaluate_window_solar_shading(record, geom, ctx, config):
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "blind"))

    def scorer(edge: GraphEdge) -> float:
        return geometric_mean([
            _prox(ctx, record.instance_id, edge),
            float(edge.metrics["facade_projection_overlap"]),
            float(edge.metrics["vertical_overlap"]),
        ])

    context, best = _best_composite(edges, scorer)
    evidence = _base_evidence(record, ctx, {"blind_association": context}, "blind/shading device projected over window")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_blind_neighbor"
    return _result(NEUTRAL, context, evidence, status, best is not None)


def evaluate_window_ventilation(record, geom, ctx, config):
    # Explicit functional dependency: ventilation evidence is the final
    # operable confidence of the SAME window instance (G), with neutral
    # context. operable > 0.5 strengthens the ventilation prior, < 0.5
    # weakens it, = 0.5 leaves it unchanged. No airflow claim is made from
    # static XYZ.
    operable = _window_operable_evidence(ctx, record.instance_id)
    if operable is None:
        evidence = _base_evidence(record, ctx, {"dependency_attribute": "operable"}, "operable evidence unavailable")
        return _result(NEUTRAL, NEUTRAL, evidence, "operable_result_missing", False)
    operable_confidence = float(operable["confidence"])
    measured = {
        "dependency_attribute": "operable",
        "operable_confidence": operable_confidence,
        "operable_geometry_score": operable.get("geometry_score"),
        "operable_context_score": operable.get("context_score"),
    }
    evidence = _base_evidence(record, ctx, measured,
                              "ventilation derived from the same window instance final operable confidence")
    informative = bool(operable.get("informative", False))
    return _result(operable_confidence, NEUTRAL, evidence, "depends_on_operable", informative)


def evaluate_door_exterior_wall(record, geom, ctx, config):
    center = 0.5 * (geom.features["local_bounds_min"] + geom.features["local_bounds_max"])
    bmin = ctx.building_local_min
    bmax = ctx.building_local_max
    span = np.maximum(bmax - bmin, 1e-9)
    distance_to_edge = float(np.min(np.min(np.c_[center - bmin, bmax - center], axis=1)[:2]))
    building_span = float(max(span[0], span[1]))
    exterior = _clip01(1.0 - distance_to_edge / max(building_span, 1e-9))

    edges = ctx.graph.edges_from(record.instance_id, _role(config, "host_surface"))

    def scorer(edge: GraphEdge) -> float:
        return geometric_mean([
            _prox(ctx, record.instance_id, edge),
            float(edge.metrics["facade_projection_overlap"]),
            float(edge.metrics["depth_overlap"]),
            float(edge.metrics["plane_parallelism"]),
        ])

    geometry_available = _geometry_available(geom, config)
    if not geometry_available:
        exterior = NEUTRAL
    context, best = _best_composite(edges, scorer)
    measured = {
        "exterior_boundary_score": exterior,
        "distance_to_building_edge": distance_to_edge,
        "wall_association": context,
    }
    evidence = _base_evidence(record, ctx, measured, "door on exterior boundary with host-wall association")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_host_wall"
    informative = geometry_available or best is not None
    return _result(exterior, context, evidence, status, informative)


def evaluate_balcony_structural(record, geom, ctx, config):
    wall_edges = ctx.graph.edges_from(record.instance_id, {"wall"})

    def wall_scorer(edge: GraphEdge) -> float:
        return geometric_mean([
            _prox(ctx, record.instance_id, edge),
            float(edge.metrics["facade_projection_overlap"]),
            float(edge.metrics["depth_overlap"]),
            float(edge.metrics["plane_parallelism"]),
        ])

    c_wall, wall_best = _best_composite(wall_edges, wall_scorer)
    c_lower, lower_best = _lower_support_context(ctx, record.instance_id)
    wall_available = wall_best is not None
    lower_available = lower_best is not None
    context, informative = availability_aware_max_merge([
        (c_wall, wall_available), (c_lower, lower_available),
    ])
    if lower_available and (not wall_available or c_lower > c_wall):
        best = lower_best
        support_kind = "lower_support"
    else:
        best = wall_best
        support_kind = "wall_attachment"
    measured = {"wall_attachment_score": c_wall, "lower_support_score": c_lower, "support_kind": support_kind}
    evidence = _base_evidence(record, ctx, measured, "balcony attached to wall or supported from below")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_support_found"
    return _result(NEUTRAL, context, evidence, status, informative)


def evaluate_vegetation_balcony_contact(record, geom, ctx, config):
    context, best = _vegetation_context(ctx, record.instance_id, side=True)
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "vegetation"))
    evidence = _base_evidence(record, ctx, {"side_contact_score": context},
                              "vegetation climbing / side-contact on balcony")
    evidence["related_instance_ids"], evidence["related_semantic_classes"] = _related(edges if best is not None else [])
    return _result(NEUTRAL, context, evidence, "ok" if best is not None else "no_vegetation_neighbor", best is not None)


def evaluate_vegetation_balcony_rests_on_top(record, geom, ctx, config):
    context, best = _vegetation_context(ctx, record.instance_id, side=False)
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "vegetation"))
    evidence = _base_evidence(record, ctx, {"rests_on_top_score": context},
                              "vegetation resting on balcony upper surface")
    evidence["related_instance_ids"], evidence["related_semantic_classes"] = _related(edges if best is not None else [])
    return _result(NEUTRAL, context, evidence, "ok" if best is not None else "no_vegetation_neighbor", best is not None)


def evaluate_stair_fall_protection(record, geom, ctx, config):
    geometry_available = _geometry_available(geom, config)
    elevation = _elevation_fraction(geom, ctx) if geometry_available else NEUTRAL
    exposure = _exposure_fraction(geom, ctx) if geometry_available else NEUTRAL
    geometry = geometric_mean([elevation, exposure]) if geometry_available else NEUTRAL
    context, best = _barrier_context(ctx, record.instance_id)
    measured = {
        "elevation_fraction": elevation,
        "open_boundary_exposure": exposure,
        "barrier_score": context,
    }
    evidence = _base_evidence(record, ctx, measured,
                              "elevated stair with exposed boundary and aligned vertical barrier")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_barrier"
    return _result(geometry, context, evidence, status, geometry_available or best is not None)


def evaluate_terrain_circulation(record, geom, ctx, config):
    top_reliable = bool(geom.features.get("top_surface_reliable", False))
    plane_reliable = bool(geom.features.get("plane_fit_reliable", False))
    if top_reliable:
        slope = float(geom.features["top_surface_slope_degrees"])
    elif plane_reliable:
        slope = float(geom.features["surface_slope_degrees"])
    else:
        slope = None
    walkable = (NEUTRAL if slope is None else
                angle_inverse_ramp(slope, float(config["scoring"]["slope_full_scale_degrees"])))
    # A low-slope surface is a necessary precondition, not sufficient evidence:
    # by itself it contributes at most neutral support for the circulation
    # function. The access connection (context) is what raises confidence.
    geometry = NEUTRAL if slope is None else NEUTRAL * walkable
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "access_connector"))

    def scorer(edge: GraphEdge) -> float:
        return geometric_mean([
            _prox(ctx, record.instance_id, edge),
            _xy_overlap(ctx, record.instance_id, edge),
        ])

    context, best = _best_composite(edges, scorer)
    measured = {"slope_degrees": slope, "slope_reliable": slope is not None,
                "walkable_score": walkable, "access_connection": context}
    evidence = _base_evidence(record, ctx, measured, "low-slope terrain connected to door or stairs")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_access_connector"
    informative = slope is not None or best is not None
    return _result(geometry, context, evidence, status, informative)


def evaluate_roof_structural(record, geom, ctx, config):
    context, best = _lower_support_context(ctx, record.instance_id)
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "lower_support"))
    evidence = _base_evidence(record, ctx, {"lower_support_score": context},
                              "roof supported by wall or column below")
    evidence["related_instance_ids"], evidence["related_semantic_classes"] = _related(edges if best is not None else [])
    status = "ok" if best is not None else "no_lower_support"
    return _result(NEUTRAL, context, evidence, status, best is not None)


def evaluate_roof_drainage(record, geom, ctx, config):
    top_reliable = bool(geom.features.get("top_surface_reliable", False))
    slope = float(geom.features.get("top_surface_slope_degrees", 0.0)) if top_reliable else None
    slope_score = (angle_ramp(slope, float(config["scoring"]["slope_full_scale_degrees"]))
                   if slope is not None else NEUTRAL)
    boundary = float(geom.features.get("top_surface_boundary_score", 0.0)) if top_reliable else NEUTRAL
    geometry = geometric_mean([slope_score, boundary]) if top_reliable else NEUTRAL

    downhill = np.array([
        float(geom.features.get("top_surface_direction_u", 0.0)),
        float(geom.features.get("top_surface_direction_v", 0.0)),
    ])
    downhill_norm = np.linalg.norm(downhill)

    def scorer(edge: GraphEdge) -> float:
        proximity_value = _prox(ctx, record.instance_id, edge)
        if downhill_norm <= 1e-9:
            return proximity_value
        target_center = 0.5 * (ctx.geometry[edge.target_instance_id].features["local_bounds_min"]
                               + ctx.geometry[edge.target_instance_id].features["local_bounds_max"])[:2]
        current_center = 0.5 * (geom.features["local_bounds_min"] + geom.features["local_bounds_max"])[:2]
        direction = target_center - current_center
        norm = np.linalg.norm(direction)
        if norm <= 1e-9:
            alignment = 0.5
        else:
            alignment = 0.5 + 0.5 * float(np.dot(downhill / downhill_norm, direction / norm))
        return geometric_mean([proximity_value, alignment])

    edges = ctx.graph.edges_from(record.instance_id, _role(config, "drainage_neighbor"))
    context, best = _best_composite(edges, scorer)
    measured = {"slope_degrees": slope, "top_surface_reliable": top_reliable,
                "slope_score": slope_score, "low_boundary_score": boundary,
                "gutter_relation": context}
    evidence = _base_evidence(record, ctx, measured, "roof slope and downhill drainage-edge relation")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_drainage_neighbor"
    return _result(geometry, context, evidence, status, top_reliable or best is not None)


def evaluate_roof_fall_protection(record, geom, ctx, config):
    top_reliable = bool(geom.features.get("top_surface_reliable", False))
    plane_reliable = bool(geom.features.get("plane_fit_reliable", False))
    if top_reliable:
        slope = float(geom.features["top_surface_slope_degrees"])
    elif plane_reliable:
        slope = float(geom.features["surface_slope_degrees"])
    else:
        slope = None
    accessible = (NEUTRAL if slope is None else
                  angle_inverse_ramp(slope, float(config["scoring"]["slope_full_scale_degrees"])))
    exposure = _exposure_fraction(geom, ctx)
    geometry = geometric_mean([accessible, exposure])
    context, best = _barrier_context(ctx, record.instance_id)
    measured = {"slope_degrees": slope, "slope_reliable": slope is not None,
                "accessible_roof_score": accessible, "open_boundary_exposure": exposure,
                "barrier_score": context}
    evidence = _base_evidence(record, ctx, measured, "accessible roof boundary with protective barrier")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_barrier"
    return _result(geometry, context, evidence, status, slope is not None or best is not None)


def evaluate_blind_operable(record, geom, ctx, config):
    geometry_available = _geometry_available(geom, config, plane=True)
    depth_variation = _depth_variation(geom, ctx)
    articulation = _articulation(geom, ctx)
    geometry = geometric_mean([depth_variation, articulation]) if geometry_available else NEUTRAL
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "window"))

    def scorer(edge: GraphEdge) -> float:
        return geometric_mean([
            _prox(ctx, record.instance_id, edge),
            float(edge.metrics["facade_projection_overlap"]),
        ])

    context, best = _best_composite(edges, scorer)
    measured = {"depth_variation": depth_variation, "articulation": articulation, "window_association": context}
    evidence = _base_evidence(record, ctx, measured, "multi-depth articulated blind geometry relative to window")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_window_neighbor"
    return _result(geometry, context, evidence, status, geometry_available or best is not None)


def evaluate_blind_privacy(record, geom, ctx, config):
    edges = ctx.graph.edges_from(record.instance_id, _role(config, "window"))

    def scorer(edge: GraphEdge) -> float:
        coverage = float(edge.metrics["facade_projection_overlap_target"])
        return geometric_mean([
            _prox(ctx, record.instance_id, edge),
            coverage,
        ])

    context, best = _best_composite(edges, scorer)
    measured = {"window_projection_coverage": context}
    evidence = _base_evidence(record, ctx, measured, "blind projected coverage of associated window")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_window_neighbor"
    return _result(NEUTRAL, context, evidence, status, best is not None)


def evaluate_column_structural(record, geom, ctx, config):
    geometry_available = _geometry_available(geom, config, plane=True)
    verticality = float(geom.features["verticality"]) if geometry_available else NEUTRAL
    ground = _ground_continuity(geom, ctx) if geometry_available else NEUTRAL
    geometry = geometric_mean([verticality, ground]) if geometry_available else NEUTRAL
    context, best = _upper_support_context(ctx, record.instance_id)
    measured = {"verticality": verticality, "ground_continuity": ground, "upper_support_score": context}
    evidence = _base_evidence(record, ctx, measured, "vertical column grounded and carrying upper structure")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_upper_contact"
    return _result(geometry, context, evidence, status, geometry_available or best is not None)


def evaluate_column_aesthetic(record, geom, ctx, config):
    top_rank = rank_01(
        float(geom.features["enlarged_top_ratio"]),
        ctx.class_rank_pools.get("column", {}).get("enlarged_top_ratio", []),
    )
    measured = {
        "enlarged_top_ratio": float(geom.features["enlarged_top_ratio"]),
        "top_enlargement_rank": top_rank,
    }
    evidence = _base_evidence(record, ctx, measured, "enlarged / articulated column-top relative to shaft")
    informative = _geometry_available(geom, config) and int(geom.features.get("sample_point_count", 0)) >= 10
    return _result(top_rank, NEUTRAL, evidence, "ok" if informative else "insufficient_shape_geometry", informative)


def evaluate_arch_structural(record, geom, ctx, config):
    geometry_available = _geometry_available(geom, config) and int(geom.features.get("sample_point_count", 0)) >= 10
    opening = float(geom.features["opening_score"]) if geometry_available else NEUTRAL
    geometry = opening
    host_context, host_best = _host_wall_association(ctx, record.instance_id)
    upper_context, upper_best = _upper_support_context(ctx, record.instance_id)
    context, context_informative = availability_aware_geometric_mean([
        (host_context, host_best is not None), (upper_context, upper_best is not None),
    ])
    measured = {"opening_score": opening, "host_support_score": host_context, "upper_load_path_score": upper_context}
    evidence = _base_evidence(record, ctx, measured, "arch bilateral geometry with host support and upper load path")
    related = [edge for edge in (host_best, upper_best) if edge is not None]
    evidence["related_instance_ids"] = list(dict.fromkeys(edge.target_instance_id for edge in related))
    evidence["related_semantic_classes"] = list(dict.fromkeys(edge.target_semantic_class for edge in related))
    status = "ok" if host_best is not None and upper_best is not None else "incomplete_host_or_upper_support"
    return _result(geometry, context, evidence, status, geometry_available or context_informative)


def evaluate_arch_relief(record, geom, ctx, config):
    geometry_available = _geometry_available(geom, config)
    depth = float(geom.features["depth"])
    width = max(float(geom.features["width"]), 1e-9)
    shallow = _clip01(1.0 - depth / width)
    non_opening = 1.0 - float(geom.features["opening_score"])
    geometry = geometric_mean([shallow, non_opening]) if geometry_available else NEUTRAL
    context, best = _host_wall_association(ctx, record.instance_id)
    measured = {"depth": depth, "width": width, "shallow_relief_score": shallow,
                "non_opening_score": non_opening, "wall_attachment": context}
    evidence = _base_evidence(record, ctx, measured, "shallow decorative arch relief on wall surface")
    if best is not None:
        evidence["related_instance_ids"], evidence["related_semantic_classes"] = ([best.target_instance_id], [best.target_semantic_class])
    status = "ok" if best is not None else "no_host_wall"
    return _result(geometry, context, evidence, status, geometry_available or best is not None)


def _fixture_context(ctx: BuildingContext, current_id: int) -> tuple[float, bool]:
    host, host_best = _host_wall_association(ctx, current_id)
    upper, upper_best = _upper_support_context(ctx, current_id)
    return availability_aware_max_merge([
        (host, host_best is not None), (upper, upper_best is not None),
    ])


def _best_device_cluster(record, geom, ctx, config, orientation: str) -> tuple[float | None, dict[str, Any] | None]:
    spacing = max(float(geom.features["estimated_point_spacing"]), 1e-9)
    radius = spacing * float(config["scoring"]["cluster_radius_multiplier"])
    clusters = _protruding_clusters(
        record.sample_points,
        np.asarray(geom.features["obb_center"]),
        np.asarray(geom.features["dominant_plane_normal"]),
        spacing,
        radius,
        float(config["numerical"]["epsilon"]),
    )
    minimum_points = int(config["scoring"]["device_min_points"])
    minimum_compactness = float(config["scoring"]["device_min_compactness"])
    reliable = [
        cluster for cluster in clusters
        if cluster["point_count"] >= minimum_points
        and _clip01(float(np.asarray(cluster["extents"]).min()) / max(float(np.asarray(cluster["extents"]).max()), 1e-9)) >= minimum_compactness
    ]
    if not reliable:
        return None, None
    diagonal = float(geom.features["diagonal"])
    scores = []
    for cluster in reliable:
        small, compact = _cluster_shape_scores(cluster, diagonal)
        outward = np.asarray(cluster["outward_direction"], dtype=np.float64)
        if orientation == "downward":
            orientation_score = _clip01(-float(outward[2]))
        else:
            orientation_score = _clip01(float(np.linalg.norm(outward[:2])))
        total = geometric_mean([small, compact, orientation_score])
        scores.append((total, cluster, small, compact, orientation_score))
    scores.sort(key=lambda item: (-item[0], -item[1]["point_count"]))
    total, cluster, small, compact, orientation_score = scores[0]
    return total, cluster


def evaluate_arch_illumination(record, geom, ctx, config):
    geometry, cluster = _best_device_cluster(record, geom, ctx, config, "downward")
    if geometry is None:
        geometry = NEUTRAL
    context, context_informative = _fixture_context(ctx, record.instance_id)
    if cluster is None:
        context = NEUTRAL
    measured = {
        "candidate_detected": cluster is not None,
        "candidate_point_count": cluster["point_count"] if cluster is not None else 0,
        "candidate_relative_size": None if cluster is None else float(cluster["extent"]) / max(float(geom.features["diagonal"]), 1e-9),
        "candidate_compactness": None if cluster is None else float(np.asarray(cluster["extents"]).min()) / max(float(np.asarray(cluster["extents"]).max()), 1e-9),
        "candidate_normal": None if cluster is None else cluster["normal"],
        "candidate_outward_direction": None if cluster is None else cluster["outward_direction"],
        "surface_attachment": context,
        "elevation_fraction": _elevation_fraction(geom, ctx),
    }
    evidence = _base_evidence(record, ctx, measured,
                              "compact downward-oriented surface-mounted fixture proxy on arch")
    evidence["fixture_candidate"] = cluster
    status = "candidate_found" if cluster is not None else "no_fixture_candidate"
    return _result(geometry, context, evidence, status, cluster is not None)


def evaluate_arch_surveillance(record, geom, ctx, config):
    geometry, cluster = _best_device_cluster(record, geom, ctx, config, "oblique")
    if geometry is None:
        geometry = NEUTRAL
    context, context_informative = _fixture_context(ctx, record.instance_id)
    if cluster is None:
        context = NEUTRAL
    mount_height = _elevation_fraction(geom, ctx)
    context = geometric_mean([context, mount_height])
    measured = {
        "candidate_detected": cluster is not None,
        "candidate_point_count": cluster["point_count"] if cluster is not None else 0,
        "candidate_relative_size": None if cluster is None else float(cluster["extent"]) / max(float(geom.features["diagonal"]), 1e-9),
        "candidate_compactness": None if cluster is None else float(np.asarray(cluster["extents"]).min()) / max(float(np.asarray(cluster["extents"]).max()), 1e-9),
        "candidate_normal": None if cluster is None else cluster["normal"],
        "candidate_outward_direction": None if cluster is None else cluster["outward_direction"],
        "surface_attachment": context,
        "mount_height": mount_height,
    }
    evidence = _base_evidence(record, ctx, measured,
                              "high-mounted oblique camera-like fixture proxy on arch")
    evidence["fixture_candidate"] = cluster
    status = "candidate_found" if cluster is not None else "no_fixture_candidate"
    return _result(geometry, context, evidence, status, cluster is not None)


# ---------------------------------------------------------------------------
# Registry and assignment
# ---------------------------------------------------------------------------


CONDITION_EVALUATORS: dict[str, Evaluator] = {
    "always": evaluate_always,
    "wall_ground_and_upper_support_contact_detected": evaluate_wall_ground_upper_support,
    "vegetation_wall_proximity_and_overlap_detected": evaluate_vegetation_wall_contact,
    "vegetation_instance_rests_on_top_of_wall_upper_surface": evaluate_vegetation_wall_rests_on_top,
    "window_surface_articulation_and_depth_variation_detected": evaluate_window_articulation,
    "window_handle_like_small_protrusion_detected": evaluate_window_handle,
    "blind_window_facade_projection_overlap_detected": evaluate_window_solar_shading,
    "window_operable_evidence_supports_ventilation": evaluate_window_ventilation,
    "door_exterior_wall_relation_detected": evaluate_door_exterior_wall,
    "balcony_wall_or_lower_support_relation_detected": evaluate_balcony_structural,
    "vegetation_balcony_proximity_and_overlap_detected": evaluate_vegetation_balcony_contact,
    "vegetation_instance_rests_on_top_of_balcony_surface": evaluate_vegetation_balcony_rests_on_top,
    "elevated_stair_open_boundary_and_barrier_relation_detected": evaluate_stair_fall_protection,
    "terrain_low_slope_and_access_connection_detected": evaluate_terrain_circulation,
    "roof_lower_support_relation_detected": evaluate_roof_structural,
    "roof_slope_and_drainage_relation_detected": evaluate_roof_drainage,
    "roof_accessible_boundary_and_barrier_relation_detected": evaluate_roof_fall_protection,
    "blind_depth_articulation_relative_to_window_detected": evaluate_blind_operable,
    "blind_window_projection_coverage_detected": evaluate_blind_privacy,
    "column_ground_and_upper_support_contact_detected": evaluate_column_structural,
    "column_top_shape_articulation_detected": evaluate_column_aesthetic,
    "arch_host_support_and_upper_load_path_detected": evaluate_arch_structural,
    "arch_shallow_wall_relief_detected": evaluate_arch_relief,
    "arch_downward_fixture_proxy_detected": evaluate_arch_illumination,
    "arch_high_mounted_oblique_fixture_proxy_detected": evaluate_arch_surveillance,
}


# Explicit attribute-level functional dependencies. Evaluation always
# resolves these before running an attribute, independent of the ontology
# YAML rule order.
ATTRIBUTE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "ventilation": ("operable",),
}


def _dependency_order(by_attribute: dict[str, Any]) -> list[str]:
    """Topological order of attributes honoring ATTRIBUTE_DEPENDENCIES.

    The result is deterministic and independent of the input dict order, so
    swapping the order of ``operable`` / ``ventilation`` in the ontology YAML
    can never change the results.
    """
    order: list[str] = []
    done: set[str] = set()
    pending: list[str] = sorted(by_attribute)
    while pending:
        progressed = False
        for attribute in pending:
            deps = ATTRIBUTE_DEPENDENCIES.get(attribute, ())
            if all(dep in done for dep in deps):
                order.append(attribute)
                done.add(attribute)
                pending.remove(attribute)
                progressed = True
                break
        if not progressed:
            raise ValueError(f"Unresolved attribute dependency cycle: {pending}")
    return order


def _select_rule_payload(rule_payloads: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """Select strongest measured rule; unknown neutral rules never mask it."""
    informative = [item for item in rule_payloads if bool(item.get("informative", False))]
    pool = informative if informative else rule_payloads
    best = max(pool, key=lambda item: (item["confidence"], -item["rule_index"]))
    return best, bool(informative)


def validate_evaluator_registry(conditions: tuple[str, ...] | list[str], exact: bool = True) -> None:
    missing = [condition for condition in conditions if condition not in CONDITION_EVALUATORS]
    if missing:
        raise ValueError(f"Missing explicit condition evaluators: {missing}")
    if exact:
        extra = [condition for condition in CONDITION_EVALUATORS if condition not in conditions]
        if extra:
            raise ValueError(f"Evaluator registry contains conditions absent from used ontology: {extra}")


def _build_boundary_tree(records: list[InstanceRecord], max_points_per_instance: int = 200) -> tuple[Any, np.ndarray] | None:
    clouds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for record in records:
        points = record.sample_points
        if len(points) == 0:
            continue
        if len(points) > max_points_per_instance:
            points = points[np.linspace(0, len(points) - 1, max_points_per_instance, dtype=np.int64)]
        clouds.append(points)
        labels.append(np.full(len(points), record.instance_id, dtype=np.int64))
    if not clouds:
        return None
    all_points = np.concatenate(clouds, axis=0)
    all_labels = np.concatenate(labels, axis=0)
    return cKDTree(all_points), all_labels


def assign_building(
    source_path: str | Path,
    label_map: LabelMap,
    ontology: Ontology,
    config: dict[str, Any],
) -> AssignmentResult:
    validate_evaluator_registry(ontology.unique_conditions)
    runtime_config = {**config, "runtime": {**config.get("runtime", {}), "building_id": Path(source_path).stem}}
    building = extract_instances(source_path, label_map, runtime_config)
    local_frame = estimate_local_frame(building.records, runtime_config)
    geometry = compute_all_geometry(building.records, local_frame, runtime_config)
    graph = build_instance_graph(building.records, geometry, runtime_config)

    count = len(building.records)
    attribute_count = len(ontology.attribute_names)

    tops = [record.xyz_max[2] for record in building.records if len(record.sample_points)]
    building_height = max(
        (max(tops) - local_frame.ground_z) if tops else 0.0,
        float(runtime_config["scoring"]["building_height_floor"]),
    )
    structural_bottoms = [
        record.xyz_min[2] for record in building.records
        if record.semantic_class in ("wall", "column")
    ]
    structural_base_z = min(
        [local_frame.ground_z] + structural_bottoms if structural_bottoms else [local_frame.ground_z]
    )
    local_mins = [geometry[record.instance_id].features["local_bounds_min"] for record in building.records]
    local_maxs = [geometry[record.instance_id].features["local_bounds_max"] for record in building.records]
    building_local_min = np.min(np.stack(local_mins), axis=0) if local_mins else np.zeros(3)
    building_local_max = np.max(np.stack(local_maxs), axis=0) if local_maxs else np.zeros(3)

    class_rank_pools: dict[str, dict[str, list[float]]] = {}
    for record in building.records:
        features = geometry[record.instance_id].features
        pool = class_rank_pools.setdefault(record.semantic_class, {})
        pool.setdefault("depth", []).append(float(features["depth"]))
        pool.setdefault("enlarged_top_ratio", []).append(float(features["enlarged_top_ratio"]))
        pool.setdefault("depth_iqr", []).append(float(features["depth_iqr"]))
        pool.setdefault("normal_dispersion", []).append(float(features["local_normal_dispersion"]))

    ctx = BuildingContext(
        building_id=Path(source_path).stem,
        frame=local_frame,
        graph=graph,
        geometry=geometry,
        records_by_id={record.instance_id: record for record in building.records},
        class_rank_pools=class_rank_pools,
        building_height=building_height,
        structural_base_z=structural_base_z,
        building_local_min=building_local_min,
        building_local_max=building_local_max,
        boundary_tree=_build_boundary_tree(building.records),
        config=runtime_config,
        per_instance_results={},
    )

    applicable = np.zeros((count, attribute_count), dtype=np.uint8)
    prior = np.zeros((count, attribute_count), dtype=np.float64)
    geometry_scores = np.zeros((count, attribute_count), dtype=np.float64)
    context_scores = np.zeros((count, attribute_count), dtype=np.float64)
    confidence = np.zeros((count, attribute_count), dtype=np.float64)
    vector = np.zeros((count, attribute_count), dtype=np.uint8)
    conditions = np.full((count, attribute_count), "", dtype=object)
    sources = np.full((count, attribute_count), "", dtype=object)
    odds_epsilon = float(runtime_config["numerical"]["odds_epsilon"])

    for row, record in enumerate(building.records):
        by_attribute = ontology.rules_by_class_attribute.get(record.semantic_class, {})
        for attribute in _dependency_order(by_attribute):
            attribute_rules = by_attribute[attribute]
            semantic_stable = all(rule.condition == "always" for rule in attribute_rules)
            column = ontology.attribute_index[attribute]
            applicable[row, column] = 1
            rule_payloads: list[dict[str, Any]] = []
            for rule in attribute_rules:
                prior_value = (
                    semantic_prior(record.semantic_class, rule.attribute, record.semantic_purity, runtime_config)
                    if semantic_stable else NEUTRAL
                )
                evaluator = CONDITION_EVALUATORS[rule.condition]
                evaluated = evaluator(record, geometry[record.instance_id], ctx, runtime_config)
                rule_confidence = (
                    odds_fusion(prior_value, evaluated.geometry_score, evaluated.context_score, odds_epsilon)
                    if rule.value
                    else 0.0
                )
                rule_payloads.append({
                    "condition": rule.condition,
                    "source": rule.source,
                    "rule_index": rule.rule_index,
                    "prior": float(prior_value),
                    "geometry_score": float(evaluated.geometry_score),
                    "context_score": float(evaluated.context_score),
                    "confidence": float(rule_confidence),
                    "informative": bool(evaluated.informative),
                })

            best, attribute_informative = _select_rule_payload(rule_payloads)
            prior[row, column] = float(best["prior"])
            geometry_scores[row, column] = float(best["geometry_score"])
            context_scores[row, column] = float(best["context_score"])
            confidence[row, column] = float(best["confidence"])
            vector[row, column] = np.uint8(1 if confidence[row, column] >= 0.5 else 0)
            conditions[row, column] = ";".join(item["condition"] for item in rule_payloads)
            sources[row, column] = ";".join(dict.fromkeys(item["source"] for item in rule_payloads))
            ctx.per_instance_results.setdefault(record.instance_id, {})[attribute] = {
                "geometry_score": float(geometry_scores[row, column]),
                "context_score": float(context_scores[row, column]),
                "confidence": float(confidence[row, column]),
                "informative": attribute_informative,
            }

    validate_score_arrays(prior, geometry_scores, context_scores, confidence)
    if np.any(vector[applicable == 0]) or np.any(confidence[applicable == 0]):
        raise AssertionError("Non-applicable ontology attributes must remain exactly zero")
    return AssignmentResult(
        building, local_frame, geometry, graph, applicable, prior, geometry_scores,
        context_scores, confidence, vector, conditions, sources,
    )
