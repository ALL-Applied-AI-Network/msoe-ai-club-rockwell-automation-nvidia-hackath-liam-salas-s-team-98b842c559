"""Frozen synthetic artifacts for a separated three-vial reference bank."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median

from .paths import REPO


CONFIG_PATH = REPO / "data/twin_vision_dataset_v1.json"
TANK_BASIS = {
    "t200": (0.006, 0.2, 0.3),
    "t400": (0.228, 0.16, 0.155),
    "t100": (0.45, 0.12, 0.01),
}
STAGE_TANKS = (("t200",), ("t200", "t400"), ("t200", "t400", "t100"))
PROFILES = {
    "none": [0.0, 0.0, 0.0],
    "first_only": [0.30, 0.0, 0.0],
    "first_two": [0.30, 0.12, 0.0],
    "all_three": [0.30, 0.18, 0.08],
}
PROFILE_ORDER = ("none", "first_only", "first_two", "all_three")
COHORTS = ("reference_only", "scored_target")
FAMILIES = [
    ("TR", "train", "S0437", "S0117", "after_t200", 310000),
    ("VA", "val", "S0160", "S0380", "after_t400", 1310000),
    ("TE_HIGH", "test", "S0177", "S0431", "after_t100", 2310000),
    ("TE_LOW", "test", "S0147", "S0074", "after_t400", 2320000),
]
STAGES = [
    ("STAGE1", 1, "after_t200", 171, ("t200",), 40.884098916),
    ("STAGE2", 2, "after_t400", 342, ("t200", "t400"), 81.888407622),
    ("STAGE3", 3, "after_t100", 686, ("t200", "t400", "t100"), 122.894103114),
]
SOURCE_HASHES = {
    "data/sim_recipes.csv": "277acc842db31cfe3ee9cbb7c336777f10136a6bcb28726e49c4c3f077ffed45",
    "data/sim_transitions.csv": "914f381b204f7174c52f968a1b00e5899ab6f3214802b050e496aa89a5b8ba9b",
    "data/sim_split_spec.json": "09457230c277866c0caa2e005e049704ebd450f5f28e5545735ba351d1acf97e",
}
LOCAL_RENDER_RUNTIME = {
    "host_name": "spark5",
    "machine": "aarch64",
    "runtime_name": "NVIDIA Isaac Sim",
    "isaac_sim_version": "6.2.0.0",
    "kit_version": "110.4.0",
    "isaac_python": "/home/hackathon5/liamreporoot/isaacsim62-env/.venv/bin/python",
    "environment_profile": "gb10-aarch64-isaac62",
}
LOCAL_ATTEMPT_POLICY = {
    "max_process_attempts": 2,
    "no_output_timeout_seconds": 180,
    "startup_readiness_timeout_seconds": 420,
    "fast_crash_retry_seconds": 180,
    "startup_readiness_marker": "Simulation App Startup Complete",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a non-Boolean number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def clean_cumulative_stage_appearances(programmed_ml: Mapping[str, object]) -> list[list[float]]:
    """Return uncalibrated direct cumulative material appearances."""
    if not isinstance(programmed_ml, Mapping) or set(programmed_ml) != {"t100", "t200", "t400"}:
        raise ValueError("programmed mL must contain exactly t100, t200, and t400")
    amounts = {tank: _finite_number(programmed_ml[tank], f"{tank} programmed mL") for tank in ("t100", "t200", "t400")}
    if any(value < 0 for value in amounts.values()):
        raise ValueError("programmed mL must be nonnegative")
    result = []
    for tanks in STAGE_TANKS:
        total = sum(amounts[tank] for tank in tanks)
        if total <= 0:
            raise ValueError("each cumulative stage must have positive programmed mL")
        result.append([sum(amounts[tank] * TANK_BASIS[tank][channel] for tank in tanks) / total for channel in range(3)])
    return result


def observed_stage_appearances(
    current_programmed_ml: Mapping[str, object],
    previous_programmed_ml: Mapping[str, object],
    appearance_blend_alpha: object,
    onset_id: str,
) -> list[list[float]]:
    """At/after onset, blend current stage toward previous final clean."""
    alpha = _finite_number(appearance_blend_alpha, "appearance blend alpha")
    if not 0 <= alpha <= 1:
        raise ValueError("appearance blend alpha must be in [0, 1]")
    onset = {"after_t200": 0, "after_t400": 1, "after_t100": 2, "none": 3}.get(onset_id)
    if onset is None:
        raise ValueError("unknown synthetic onset")
    current = clean_cumulative_stage_appearances(current_programmed_ml)
    previous_final = clean_cumulative_stage_appearances(previous_programmed_ml)[2]
    return [
        [now + alpha * (before - now) if ordinal >= onset else now for now, before in zip(stage, previous_final, strict=True)]
        for ordinal, stage in enumerate(current)
    ]


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def select_peer_reference(candidates: object, *, target_mask_sha256: object) -> dict:
    """Select the unique largest tight cluster corroborated across batches."""
    invalid = {"validity": "insufficient_evidence", "limit_code": "reference_invalid"}
    if not isinstance(target_mask_sha256, str) or not target_mask_sha256:
        return invalid | {"limit_code": "mask_missing"}
    if not isinstance(candidates, Mapping) or len(candidates) < 2:
        return invalid | {"limit_code": "reference_not_unique_or_too_small"}
    values: dict[str, tuple[float, float, float]] = {}
    metadata: dict[str, dict[str, str]] = {}
    try:
        for peer_id, candidate in candidates.items():
            if not isinstance(peer_id, str) or not isinstance(candidate, Mapping):
                return invalid
            lab = candidate.get("lab")
            if not isinstance(lab, Sequence) or isinstance(lab, (str, bytes)) or len(lab) != 3:
                return invalid
            if candidate.get("mask_sha256") != target_mask_sha256:
                return invalid | {"limit_code": "mask_mismatch"}
            batch_id, origin = candidate.get("batch_id"), candidate.get("origin")
            if not isinstance(batch_id, str) or not batch_id or not isinstance(origin, str) or not origin:
                return invalid
            values[peer_id] = tuple(_finite_number(item, "Lab channel") for item in lab)  # type: ignore[assignment]
            metadata[peer_id] = {"batch_id": batch_id, "origin": origin}
    except (TypeError, ValueError):
        return invalid
    ids = sorted(values)
    accepted: list[tuple[str, ...]] = []
    diameters: dict[tuple[str, ...], float] = {}
    for size in range(len(ids), 1, -1):
        for subset in combinations(ids, size):
            diameter = max((_distance(values[a], values[b]) for a, b in combinations(subset, 2)), default=0.0)
            if diameter <= 2.0:
                accepted.append(subset)
                diameters[subset] = diameter
        if accepted:
            break
    if len(accepted) != 1:
        return invalid | {"limit_code": "reference_not_unique_or_too_small"}
    selected = accepted[0]
    batch_ids = sorted({metadata[peer]["batch_id"] for peer in selected})
    if len(batch_ids) < 2:
        return invalid | {"limit_code": "cross_batch_corroboration_missing"}
    center = [median(values[peer][channel] for peer in selected) for channel in range(3)]
    radii = {peer: _distance(values[peer], center) for peer in selected}
    radius_median = median(radii.values())
    radius_max = max(radii.values())
    radius_mad = median(abs(value - radius_median) for value in radii.values())
    threshold = max(2.0, radius_max + 1.0)
    representative = min(selected, key=lambda peer: (radii[peer], peer))
    raw = {
        "center_lab": center,
        "median_radius_delta_e76": radius_median,
        "max_radius_delta_e76": radius_max,
        "mad_radius_delta_e76": radius_mad,
        "pairwise_diameter_delta_e76": diameters[selected],
        "threshold_delta_e76": threshold,
    }
    return {
        "validity": "valid",
        "limit_code": None,
        "peer_ids": list(selected),
        "peer_origins": [metadata[peer] | {"peer_id": peer} for peer in selected],
        "peer_count": len(selected),
        "peer_batch_ids": batch_ids,
        "peer_batch_count": len(batch_ids),
        "representative_peer_id": representative,
        **raw,
        "serialized_6dp": {
            key: [f"{item:.6f}" for item in value] if isinstance(value, list) else f"{value:.6f}"
            for key, value in raw.items()
        },
    }


def validate_vision_config(config: object) -> None:
    """Reject drift from the frozen separated-bank design."""
    if not isinstance(config, Mapping):
        raise TypeError("vision config must be an object")
    if (config.get("config_status"), config.get("render_status"), config.get("training_eligible")) != ("design_frozen", "unrendered", False):
        raise ValueError("config status must remain frozen, unrendered, and not training eligible")
    if config.get("cohorts") != list(COHORTS):
        raise ValueError("cohort separation drift")
    profiles = config.get("alpha_profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("alpha profiles are required")
    for name, expected in PROFILES.items():
        if profiles.get(name) != expected:
            raise ValueError(f"{name} alpha profile drift")
    expected_families = [
        {"family_id": row[0], "partition": row[1], "previous_sim_recipe_id": row[2], "current_sim_recipe_id": row[3], "onset_id": row[4], "seed_base": row[5]}
        for row in FAMILIES
    ]
    if config.get("recipe_families") != expected_families:
        raise ValueError("recipe family, split, or seed binding drift")
    expected_stages = [
        {"stage_id": row[0], "ordinal": row[1], "onset_id": row[2], "time_code": row[3], "dispensed_tanks": list(row[4]), "measured_geometry_extent_mm": row[5]}
        for row in STAGES
    ]
    if config.get("stage_slots") != expected_stages:
        raise ValueError("stage slot drift")
    if config.get("tank_basis_linear_rgb") != {key: list(value) for key, value in TANK_BASIS.items()}:
        raise ValueError("tank basis drift")
    if config.get("source_sha256") != SOURCE_HASHES:
        raise ValueError("source hash binding drift")
    if config.get("render_runtime") != LOCAL_RENDER_RUNTIME or config.get("process_attempt_policy") != LOCAL_ATTEMPT_POLICY:
        raise ValueError("local Isaac 6.2 runtime or watchdog binding drift")
    future_brev = config.get("future_brev_runtime", {})
    if (future_brev.get("isaac_sim_version"), future_brev.get("environment_profile")) != ("6.0.1.0", "brev-x86_64-isaac601"):
        raise ValueError("future Brev 6.0.1 profile drift")
    if config.get("runtime_mixing_policy") != "one_artifact_set_one_host_and_runtime_profile_never_mix":
        raise ValueError("runtime mixing policy drift")
    if config.get("batch_size_vials") != 3:
        raise ValueError("batch size drift")
    seed = config.get("seed_contract", {})
    if (seed.get("reference_offsets_by_vial"), seed.get("target_offsets_by_vial"), seed.get("profile_changes_seed")) != ([1, 2, 3], [101, 102, 103], False):
        raise ValueError("seed matching contract drift")
    reference = config.get("reference_contract", {})
    if (reference.get("minimum_peer_count"), reference.get("minimum_distinct_batch_ids"), reference.get("primary_mode")) != (2, 2, "all_reference_batches"):
        raise ValueError("cross-batch reference contract drift")
    if reference.get("availability_cases") != ["same_batch_only", "one_reference_batch_x4", "all_reference_batches"]:
        raise ValueError("reference availability cases drift")
    policy = config.get("partition_policy", {})
    if policy.get("test_evaluation_scope") != "reference_assisted_transductive_not_pure_recipe_generalization" or policy.get("test_data_used_for_training_or_tuning") is not False:
        raise ValueError("test transductive boundary drift")
    controls = config.get("render_controls", {})
    required_controls = {"images_per_launch": 3, "launches_per_batch": 3, "subframes": 64, "warmup": 60, "max_tries": 20, "timeout_seconds": 1800, "base_nuisance_only": True}
    if any(controls.get(key) != value for key, value in required_controls.items()):
        raise ValueError("render control drift")
    if config.get("decision_timing") != "retrospective_after_complete_three_vial_batch_no_latency_claim":
        raise ValueError("decision timing drift")
    if config.get("real_data_used") is not False:
        raise ValueError("real data is forbidden")


def _load_source_bindings(repo_root: Path, config: Mapping[str, object]) -> tuple[dict[str, dict], dict[tuple[str, str, str], str]]:
    for relative, expected in SOURCE_HASHES.items():
        if _sha256_path(repo_root / relative) != expected:
            raise ValueError(f"source hash drift: {relative}")
    split = json.loads((repo_root / "data/sim_split_spec.json").read_text(encoding="utf-8"))
    if split.get("version") != "v2":
        raise ValueError("split spec version drift")
    lines = (repo_root / "data/sim_recipes.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    recipes: dict[str, dict] = {}
    for line, row in zip(lines[1:], csv.DictReader(lines), strict=True):
        recipe_id = row["sim_recipe_id"]
        recipes[recipe_id] = {
            "sim_recipe_id": recipe_id,
            "role": row["role"],
            "cell": row["cell"],
            "programmed_ml": {tank: int(row[f"{tank}_ml"]) for tank in ("t100", "t200", "t400")},
            "total_ml": int(row["total_ml"]),
            "row_sha256": _sha256_bytes(line.encode("utf-8")),
        }
    transition_lines = (repo_root / "data/sim_transitions.csv").read_text(encoding="utf-8").splitlines(keepends=True)
    transitions = {
        (row["role"], row["prev_sim_recipe_id"], row["sim_recipe_id"]): _sha256_bytes(line.encode("utf-8"))
        for line, row in zip(transition_lines[1:], csv.DictReader(transition_lines), strict=True)
    }
    selected_recipe_roles: dict[str, str] = {}
    selected_cell_roles: dict[str, str] = {}
    used_seeds: set[int] = set()
    for family in config["recipe_families"]:  # type: ignore[index]
        role = family["partition"]
        transition = (role, family["previous_sim_recipe_id"], family["current_sim_recipe_id"])
        if transition not in transitions:
            raise ValueError(f"missing transition row: {transition}")
        for key in ("previous_sim_recipe_id", "current_sim_recipe_id"):
            recipe_id = family[key]
            recipe = recipes.get(recipe_id)
            if recipe is None or recipe["role"] != role or any(value <= 0 for value in recipe["programmed_ml"].values()):
                raise ValueError(f"recipe role/nonzero binding failed: {recipe_id}")
            total = recipe["total_ml"]
            for reserved in split["real_recipes_reserved"].values():
                reserved_total = sum(reserved.values())
                near_ml = all(abs(recipe["programmed_ml"][tank] - reserved[tank]) <= split["near_rule"]["ml_per_tank"] for tank in TANK_BASIS)
                near_fraction = all(abs(recipe["programmed_ml"][tank] / total - reserved[tank] / reserved_total) <= split["near_rule"]["fraction_per_tank"] for tank in TANK_BASIS)
                if near_ml or near_fraction:
                    raise ValueError(f"selected recipe is reserved-real-near: {recipe_id}")
            if recipe_id in selected_recipe_roles and selected_recipe_roles[recipe_id] != role:
                raise ValueError("recipe cross-role leakage")
            selected_recipe_roles[recipe_id] = role
            if recipe["cell"] in selected_cell_roles and selected_cell_roles[recipe["cell"]] != role:
                raise ValueError("selected recipe cell cross-role leakage")
            selected_cell_roles[recipe["cell"]] = role
        family_seeds = {family["seed_base"] + offset for offset in (1, 2, 3, 101, 102, 103)}
        low, high = split["render_seed_ranges"][role]
        if min(family_seeds) < low or max(family_seeds) > high or used_seeds.intersection(family_seeds):
            raise ValueError("family seeds overlap or leave split range")
        used_seeds.update(family_seeds)
    return recipes, transitions


def _image_id(batch_id: object, vial_id: str, stage_id: str) -> str:
    return f"{batch_id}-{vial_id}-{stage_id}"


def _opaque_batch_id(family_id: str, cohort: str, profile_id: str) -> str:
    """Return a stable identifier whose spelling exposes no profile ordering."""

    bound = (
        "three-vial-separated-reference-bank-v1:"
        f"{family_id}:{cohort}:{profile_id}"
    ).encode("utf-8")
    return f"BATCH_{hashlib.sha256(bound).hexdigest()[:12].upper()}"


def _model_batch(batch: Mapping[str, object], family: Mapping[str, object]) -> dict:
    previous = family["recipe_source_binding"]["previous"]
    current = family["recipe_source_binding"]["current"]
    return {
        "batch_id": batch["batch_id"],
        "family_id": batch["family_id"],
        "cohort": batch["cohort"],
        "partition": batch["partition"],
        "previous_sim_recipe_id": previous["sim_recipe_id"],
        "current_sim_recipe_id": current["sim_recipe_id"],
        "previous_programmed_ml": previous["programmed_ml"],
        "current_programmed_ml": current["programmed_ml"],
        "programmed_dispense_order": ["t200", "t400", "t100"],
        "sequence_scope": "same_recipe_family_batch",
        "vial_ids": ["V1", "V2", "V3"],
        "render_status": "unrendered",
    }


def _candidate(batch: Mapping[str, object], vial_id: str, stage_id: str, origin: str) -> dict:
    return {
        "peer_image_id": _image_id(batch["batch_id"], vial_id, stage_id),
        "image_placeholder": f"unrendered:{_image_id(batch['batch_id'], vial_id, stage_id)}",
        "batch_id": batch["batch_id"],
        "vial_id": vial_id,
        "current_sim_recipe_id": batch["current_sim_recipe_id"],
        "origin": origin,
        "partition": batch["partition"],
    }


def _reference_cases(target: Mapping[str, object], family_batches: Sequence[Mapping[str, object]], vial_ordinal: int, stage_id: str) -> list[dict]:
    later = [_candidate(target, f"V{ordinal}", stage_id, "later_same_target_batch") for ordinal in range(vial_ordinal + 1, 4)]
    reference_batches = sorted((batch for batch in family_batches if batch["cohort"] == "reference_only"), key=lambda batch: str(batch["batch_id"]))
    bank_by_batch = {
        str(batch["batch_id"]): [_candidate(batch, f"V{ordinal}", stage_id, "reference_bank") for ordinal in range(1, 4)]
        for batch in reference_batches
    }

    def case(mode: str, alternative_id: str | None, peers: list[dict], primary: bool) -> dict:
        batch_count = len({peer["batch_id"] for peer in peers})
        eligible = len(peers) >= 2 and batch_count >= 2
        return {
            "case_id": mode if alternative_id is None else f"{mode}:{alternative_id}",
            "mode": mode,
            "alternative_id": alternative_id,
            "candidate_peers": peers,
            "expected_validity": "eligible_if_unique_largest_tight_cross_batch_mode" if eligible else "insufficient_evidence",
            "disposition_source": primary,
            "measurement_status": "unrendered",
            "validity": "invalid",
            "limit_codes": ["unrendered", "missing_rgb_and_mask"],
            "selected_peer_ids": None,
            "representative_peer_id": None,
            "raw_values": None,
            "serialized_6dp": None,
        }

    cases = [case("same_batch_only", None, later, False)]
    cases.extend(case("one_reference_batch", batch_id, later + peers, False) for batch_id, peers in bank_by_batch.items())
    all_bank = [peer for batch_id in sorted(bank_by_batch) for peer in bank_by_batch[batch_id]]
    cases.append(case("all_reference_batches", None, later + all_bank, True))
    return cases


def build_vision_artifacts(config_path: str | Path = CONFIG_PATH, *, repo_root: str | Path = REPO) -> dict:
    """Build separate render, consumer-input and label-provenance plans."""
    path = Path(config_path)
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_vision_config(config)
    root = Path(repo_root)
    recipes, transitions = _load_source_bindings(root, config)
    families: list[dict] = []
    family_lookup: dict[str, dict] = {}
    for family in config["recipe_families"]:
        previous = recipes[family["previous_sim_recipe_id"]]
        current = recipes[family["current_sim_recipe_id"]]
        bound = dict(family) | {
            "recipe_source_binding": {"previous": previous, "current": current},
            "transition_source_binding": {
                "role": family["partition"],
                "previous_sim_recipe_id": previous["sim_recipe_id"],
                "current_sim_recipe_id": current["sim_recipe_id"],
                "row_sha256": transitions[(family["partition"], previous["sim_recipe_id"], current["sim_recipe_id"])],
            },
        }
        families.append(bound)
        family_lookup[family["family_id"]] = bound

    batches: list[dict] = []
    for family in families:
        for cohort in COHORTS:
            offsets = [1, 2, 3] if cohort == "reference_only" else [101, 102, 103]
            for profile_id in PROFILE_ORDER:
                batches.append({
                    "batch_id": _opaque_batch_id(
                        family["family_id"], cohort, profile_id
                    ),
                    "family_id": family["family_id"],
                    "cohort": cohort,
                    "partition": family["partition"],
                    "previous_sim_recipe_id": family["previous_sim_recipe_id"],
                    "current_sim_recipe_id": family["current_sim_recipe_id"],
                    "synthetic_profile_id": profile_id,
                    "appearance_blend_alpha_by_vial": config["alpha_profiles"][profile_id],
                    "synthetic_onset_id": "none" if profile_id == "none" else family["onset_id"],
                    "render_seed_by_vial": [family["seed_base"] + offset for offset in offsets],
                    "nuisance_pairing": "matched_across_profile_counterfactual_batches",
                    "statistical_independence_claim": False,
                    "vial_count": 3,
                })

    launches: list[dict] = []
    images: list[dict] = []
    labels: list[dict] = []
    for batch in batches:
        family = family_lookup[batch["family_id"]]
        previous = family["recipe_source_binding"]["previous"]
        current = family["recipe_source_binding"]["current"]
        clean = clean_cumulative_stage_appearances(current["programmed_ml"])
        for vial_ordinal, (alpha, seed) in enumerate(zip(batch["appearance_blend_alpha_by_vial"], batch["render_seed_by_vial"], strict=True), start=1):
            vial_id = f"V{vial_ordinal}"
            launch_id = f"{batch['batch_id']}-{vial_id}"
            observed = observed_stage_appearances(current["programmed_ml"], previous["programmed_ml"], alpha, batch["synthetic_onset_id"])
            samples = []
            for stage_index, stage in enumerate(config["stage_slots"]):
                image_id = _image_id(batch["batch_id"], vial_id, stage["stage_id"])
                image = {
                    "image_id": image_id,
                    "family_id": batch["family_id"],
                    "batch_id": batch["batch_id"],
                    "cohort": batch["cohort"],
                    "partition": batch["partition"],
                    "current_sim_recipe_id": batch["current_sim_recipe_id"],
                    "vial_id": vial_id,
                    "stage_id": stage["stage_id"],
                    "time_code": stage["time_code"],
                    "measured_geometry_extent_mm": stage["measured_geometry_extent_mm"],
                    "synthetic_profile_id": batch["synthetic_profile_id"],
                    "appearance_blend_alpha": alpha,
                    "synthetic_onset_id": batch["synthetic_onset_id"],
                    "clean_stage_linear_rgb": clean[stage_index],
                    "observed_stage_linear_rgb": observed[stage_index],
                    "semantic_aov": "liquid_required_unrendered",
                    "render_status": "unrendered",
                }
                images.append(image)
                labels.append({
                    "image_id": image_id,
                    "partition": batch["partition"],
                    "synthetic_profile_id": batch["synthetic_profile_id"],
                    "appearance_blend_alpha": alpha,
                    "synthetic_onset_id": batch["synthetic_onset_id"],
                    "clean_stage_linear_rgb": clean[stage_index],
                    "observed_stage_linear_rgb": observed[stage_index],
                    "semantic_aov": "liquid_required_unrendered",
                    "label_source": "render_parameters",
                    "training_eligible": False,
                })
                samples.append({"image_id": image_id, "stage_id": stage["stage_id"], "time_code": stage["time_code"], "observed_stage_linear_rgb": observed[stage_index]})
            launches.append({
                "launch_id": launch_id,
                "family_id": batch["family_id"],
                "batch_id": batch["batch_id"],
                "cohort": batch["cohort"],
                "partition": batch["partition"],
                "vial_id": vial_id,
                "render_seed": seed,
                "render_runtime_profile": config["render_runtime"]["environment_profile"],
                "max_process_attempts": config["process_attempt_policy"]["max_process_attempts"],
                "process_attempt_policy": config["process_attempt_policy"],
                "stage_samples": samples,
                "render_controls": config["render_controls"],
                "attempt_limit": config["process_attempt_policy"]["max_process_attempts"],
                "replacement_attempts": config["process_attempt_policy"]["max_process_attempts"] - 1,
            })

    model_batches = [_model_batch(batch, family_lookup[batch["family_id"]]) for batch in batches]
    reference_model_batches = [batch for batch in model_batches if batch["cohort"] == "reference_only"]
    target_model_batches = [batch for batch in model_batches if batch["cohort"] == "scored_target"]
    model_placeholders = []
    for batch in model_batches:
        for vial_id in ("V1", "V2", "V3"):
            for stage in config["stage_slots"]:
                image_id = _image_id(batch["batch_id"], vial_id, stage["stage_id"])
                model_placeholders.append({
                    "image_id": image_id,
                    "image_placeholder": f"unrendered:{image_id}",
                    "family_id": batch["family_id"],
                    "batch_id": batch["batch_id"],
                    "cohort": batch["cohort"],
                    "partition": batch["partition"],
                    "previous_sim_recipe_id": batch["previous_sim_recipe_id"],
                    "current_sim_recipe_id": batch["current_sim_recipe_id"],
                    "previous_programmed_ml": batch["previous_programmed_ml"],
                    "current_programmed_ml": batch["current_programmed_ml"],
                    "programmed_dispense_order": batch["programmed_dispense_order"],
                    "vial_id": vial_id,
                    "stage_id": stage["stage_id"],
                    "stage_ordinal": stage["ordinal"],
                    "time_code": stage["time_code"],
                    "measured_geometry_extent_mm": stage["measured_geometry_extent_mm"],
                    "render_status": "unrendered",
                })

    decisions = []
    batches_by_family = {family["family_id"]: [batch for batch in model_batches if batch["family_id"] == family["family_id"]] for family in families}
    for target in target_model_batches:
        for vial_ordinal in range(1, 4):
            vial_id = f"V{vial_ordinal}"
            stages = []
            for stage in config["stage_slots"]:
                image_id = _image_id(target["batch_id"], vial_id, stage["stage_id"])
                reference_cases = _reference_cases(target, batches_by_family[target["family_id"]], vial_ordinal, stage["stage_id"])
                reference_terminal_ids = [
                    _image_id(batch["batch_id"], "V3", stage["stage_id"])
                    for batch in batches_by_family[target["family_id"]]
                    if batch["cohort"] == "reference_only"
                ]
                stages.append({
                    "evidence_id": f"EVID-{image_id}",
                    "measurement_id": f"MEAS-{image_id}",
                    "stage_id": stage["stage_id"],
                    "stage_ordinal": stage["ordinal"],
                    "time_code": stage["time_code"],
                    "measured_geometry_extent_mm": stage["measured_geometry_extent_mm"],
                    "target_image_id": image_id,
                    "target_image_placeholder": f"unrendered:{image_id}",
                    "reference_cases": reference_cases,
                    "primary_mode": "all_reference_batches",
                    "optional_terminal_sanity_comparison": {"image_ids": reference_terminal_ids, "selection_use": False, "truth_status": "unverified", "raw_values": None},
                    "target_delta_e76_raw": None,
                    "threshold_delta_e76_raw": None,
                    "unit": "delta_e76",
                    "mask_sha256": None,
                    "stage_outlier": None,
                    "validity": "invalid",
                    "limit_codes": ["unrendered", "missing_rgb_and_mask"],
                })
            decisions.append({
                "decision_id": f"DEC-{target['batch_id']}-{vial_id}",
                "family_id": target["family_id"],
                "batch_id": target["batch_id"],
                "cohort": "scored_target",
                "partition": target["partition"],
                "previous_sim_recipe_id": target["previous_sim_recipe_id"],
                "current_sim_recipe_id": target["current_sim_recipe_id"],
                "previous_programmed_ml": target["previous_programmed_ml"],
                "current_programmed_ml": target["current_programmed_ml"],
                "programmed_dispense_order": target["programmed_dispense_order"],
                "vial_id": vial_id,
                "vial_position": vial_ordinal,
                "stages": stages,
                "disposition": "insufficient_evidence",
                "explanation_status": "UNKNOWN",
                "disposition_source": "deterministic_pipeline_primary_reference_evidence",
                "decision_timing": config["decision_timing"],
                "limit_codes": ["unrendered"],
            })

    conditions = ("numbers_only", "ordered_stage_visual_pairs", "irrelevant_shuffled_images", "deterministic_template")
    attempts = [
        {
            "attempt_id": f"{condition}-{decision['decision_id']}",
            "condition_id": condition,
            "kind": "control" if condition == "deterministic_template" else "model",
            "decision_id": decision["decision_id"],
            "partition": decision["partition"],
            "disposition": decision["disposition"],
            "explanation_status": decision["explanation_status"],
            "disposition_source": "deterministic_pipeline_evidence",
            "requires_multi_image_smoke": condition in {"ordered_stage_visual_pairs", "irrelevant_shuffled_images"},
            "status": "not_run",
        }
        for decision in decisions
        for condition in conditions
    ]

    common_status = {"config_status": "design_frozen", "render_status": "unrendered", "training_eligible": False}
    render_plan = {
        "schema_version": 3,
        "artifact_type": "render_plan",
        "config_id": config["config_id"],
        "config_sha256": _sha256_path(path),
        "status": common_status,
        "source_sha256": config["source_sha256"],
        "claim_boundary": config["claim_boundary"],
        "hypothesis": "A majority tight appearance mode corroborated across a separate same-recipe reference bank can support stage-specific target decisions.",
        "falsifiers": ["reference-bank contamination creates a wrong majority", "equal-size tight modes remain ambiguous", "reference cases remain invalid", "matched controls become outliers", "mask alignment fails"],
        "source_batch_evidence": config["source_batch_evidence"],
        "render_runtime": config["render_runtime"],
        "process_attempt_policy": config["process_attempt_policy"],
        "runtime_contract_reference": config["runtime_contract_reference"],
        "future_brev_runtime": config["future_brev_runtime"],
        "runtime_mixing_policy": config["runtime_mixing_policy"],
        "denominators": {"recipe_families": 4, "total_batches": 32, "reference_only_batches": 16, "scored_target_batches": 16, "vials_and_render_launches": 96, "stage_images_planned": 288},
        "families": families,
        "batches": batches,
        "render_launches": launches,
        "images": images,
        "render_controls": config["render_controls"],
    }
    model_plan = {
        "schema_version": 3,
        "artifact_type": "model_input_plan",
        "config_id": config["config_id"],
        "config_sha256": _sha256_path(path),
        "status": common_status,
        "denominators": {"scored_target_decisions": 48, "reference_availability_target_stage_records": 864, "cosmos_model_calls_scored": 144, "template_controls_scored": 48, "scored_attempts_total": 192, "multi_image_smoke_unscored": 1},
        "partition_policy": config["partition_policy"],
        "decision_timing": config["decision_timing"],
        "reference_policy": {
            "primary_mode": "all_reference_batches",
            "minimum_peer_count": 2,
            "minimum_distinct_batch_ids": 2,
            "tight_cluster_max_diameter_delta_e76": 2.0,
            "unique_largest_cluster_required": True,
            "same_batch_case_is_diagnostic_only": True,
            "one_reference_batch_alternatives": 4,
            "terminal_comparison_is_optional_unverified_and_not_selection": True,
        },
        "reference_batches": reference_model_batches,
        "target_batches": target_model_batches,
        "image_placeholders": model_placeholders,
        "decision_records": decisions,
        "cosmos_attempts": attempts,
        "multi_image_smoke": {"sample_count": 1, "scored": False, "maximum_images": 6, "status": "not_run", "failure_policy": "record_image_conditions_as_failed_or_skipped_without_reducing_frozen_denominators"},
        "model_input_allowlist": config["field_boundary"]["model_input_fields"],
        "real_data_used": False,
    }
    label_plan = {
        "schema_version": 3,
        "artifact_type": "label_provenance_plan",
        "config_id": config["config_id"],
        "config_sha256": _sha256_path(path),
        "status": common_status,
        "oracle_field_allowlist": config["field_boundary"]["label_provenance_oracle_fields"],
        "render_runtime_profile": config["render_runtime"]["environment_profile"],
        "image_labels": labels,
        "claim_boundary": "Synthetic render provenance only; labels are not model inputs and no image has been rendered.",
    }
    return {
        "render-plan.json": render_plan,
        "model-input-plan.json": model_plan,
        "label-provenance-plan.json": label_plan,
    }
