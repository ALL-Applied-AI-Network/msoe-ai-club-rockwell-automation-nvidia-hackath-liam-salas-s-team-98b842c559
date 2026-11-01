"""Authoritative ownership and isolation manifest for the Part A/Part B boundary.

Names in this file are import names or repository-relative POSIX paths.  The sets are
deliberately explicit: adding a production or evaluation entry point requires a reviewed
manifest change instead of relying on filename or keyword inference.
"""

from __future__ import annotations

import ast
from collections import deque
from importlib.abc import MetaPathFinder
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import sys


# Simulation truth, render construction, synthetic labels, and host/runtime provenance.
# Shared byte-loading helpers are not Part A by themselves and therefore do not appear here.
PART_A_ONLY_MODULES = frozenset(
    {
        "twin.render.brev_video_cohort",
        "twin.scene.capseal_structural",
        "fluidmix.core.contracts.sim_metadata",
        "twin.probes.explicit_mass_probe",
        "twin.cohorts.downstream_binding",
        "twin.cohorts.receipt_manifest_publisher",
        "twin.cohorts.stage_cohort_materializer",
        "twin.cohorts.varied_stage_materializer",
        "twin.cohorts.full_transition_manifest",
        "fluidmix.memory.reference_bank.materializer",
        "fluidmix.memory.reference_bank.postrender_producer",
        "fluidmix.memory.reference_bank.bank",
        "fluidmix.storage.artifacts.reference_bank_manifest",
        "twin.cohorts.reference_bank_manifest",
        "twin.physx.isosurface_geometry",
        "twin.physx.isosurface_mesh_proxy",
        "twin.scene.nuisance_authoring",
        "twin.render.palette_cut_reel",
        "twin.physx.paint_ball_emitter",
        "twin.probes.particle_array_probe",
        "twin.physx.particle_diffusion",
        "twin.physx.particle_palette",
        "twin.physx.case_plan",
        "twin.physx.composition_cohort",
        "twin.physx.dispense_residual",
        "twin.cohorts.one_run_manifest_v1",
        "twin.physx.particle_contract",
        "twin.physx.residual_case",
        "twin.physx.residual_pilot",
        "twin.physx.residual_schedule",
        "twin.physx.run_variation",
        "twin.physx.three_pour",
        "twin.cohorts.train_residual_manifest_v2",
        "twin.cohorts.varied_cohort_manifest_v3",
        "twin.cohorts.varied_cohort_manifest_v4",
        "twin.physx.volume_gate",
        "twin.probes.pilot_dataset",
        "twin.probes.parse_registration_probe",
        "twin.render.render_capture",
        "twin.probes.resolution_probe",
        "twin.render.review_contact_sheet",
        "twin.render.review_video",
        "twin.transport.rigid_vial_transport",
        "twin.postrender.run_sequence_dataset",
        "twin.probes.spatial_omnisurface_probe",
        "twin.physx.spatial_surface_color",
        "twin.scene.tank_colour_presets",
        "twin.scene.tank_station",
        "twin.cohorts.test_recipe_generator",
        "twin.probes.timeline_probe",
        "twin.transport.transform_attestation",
        "twin.transport.transport_stage",
        "twin.scene.transmissive_optics",
        "twin.scene.twin_appearance",
        "twin.scene.twin_stage_appearance",
        "twin.render.twin_video",
        "fluidmix.measurement.validation_measurement_binding",
        "twin.transport.vial_transport",
        "twin.physx.volume_height_curve",
        "twin.postrender.wp1_postrender",
        "twin.postrender.wp1_dp3_fallback",
    }
)


# Callable Part B boundaries that expose the reviewed production path.  Synthetic capture,
# model-plan, triage, and direct-mask helpers are evaluation-only below.
PRODUCTION_PART_B_ROOTS = frozenset(
    {
        "fluidmix.pipeline.carryover_pipeline",
        "fluidmix.pipeline.history_population",
        "fluidmix.pipeline.sixth_development_run",
        "fluidmix.perception.cnn.cnn_predictor",
        "fluidmix.core.contracts.evidence",
        "fluidmix.core.policy",
        "fluidmix.perception.baselines.empty_reference_rgb_predictor",
        "fluidmix.perception.baselines.timing_adapter",
        "fluidmix.storage.sqlite.evidence_store",
        "fluidmix.storage.artifacts.part_b_evidence",
        "fluidmix.pipeline.service.accepted_adapter",
        "fluidmix.pipeline.service.http",
        "fluidmix.pipeline.service.manager",
        "fluidmix.pipeline.service.production_factory",
        "fluidmix.perception.materializer",
        "fluidmix.storage.sqlite.predicted_store_bridge",
        "fluidmix.perception.masks",
        "fluidmix.memory.history.sixth_development_run",
        "fluidmix.measurement.vision_measurement",
        "fluidmix.pipeline.rgb_runtime",
        "fluidmix.explanation.vlm.stock_runtime",
        "fluidmix.memory.history.development_reference",
        "fluidmix.decision.vial",
        "fluidmix.pipeline.cohort_preparation",
        "fluidmix.memory.history.history",
        "fluidmix.pipeline.phase_capture",
        "fluidmix.perception.tracker.fill_step_tracker",
        "fluidmix.perception.tracker.fill_step_production",
        "fluidmix.perception.tracker.tracker_runtime",
        "fluidmix.pipeline.tracker_receipt_adapter",
        "fluidmix.decision.vlm",
        "fluidmix.explanation.vlm.production_materializer",
    }
)


# These modules may score saved outputs or run model experiments.  They are never production
# evidence constructors and may not be imported by a production root.
EVALUATION_ONLY_MODULES = frozenset(
    {
        "fluidmix.storage.artifacts.capture_manifest_adapter",
        "fluidmix.perception.cnn.cnn_harness_recovery",
        "fluidmix.perception.cnn.cnn_holdout",
        "fluidmix.perception.cnn.cnn_training",
        "fluidmix.explanation.vlm.cosmos_contract_runner",
        "fluidmix.explanation.vlm.cosmos_nano_contract",
        "fluidmix.explanation.vlm.cosmos_nano_input_audit",
        "fluidmix.explanation.vlm.cosmos_nano_model_view",
        "fluidmix.explanation.vlm.cosmos_nano_runner",
        "evaluation.scoring.cosmos_nano_scoring",
        "fluidmix.explanation.vlm.cosmos_nano_usable_contract",
        "fluidmix.explanation.vlm.cosmos_usable_output",
        "evaluation.scoring.cosmos_usable_scoring",
        "evaluation.scoring.empty_reference_development_scoring",
        "evaluation.scoring.explanation_scoring",
        "fluidmix.explanation.template.explanation_contract",
        "fluidmix.measurement.fill_clips",
        "twin.cohorts.fill_clip_inputs",
        "fluidmix.pipeline.varied_demo",
        "twin.cohorts.varied_image_locator",
        "fluidmix.measurement.varied_stage_measurement",
        "fluidmix.pipeline.model_plan_measurement",
        "fluidmix.pipeline.model_input_plan",
        "fluidmix.pipeline.validation_stage_measurement",
        "fluidmix.pipeline.varied_stage_measurement",
        "fluidmix.decision.triage",
        "evaluation.rca_bench.model_bakeoff_runner",
        "fluidmix.explanation.vlm.model_input_plan_builder",
        "fluidmix.memory.reference_bank.predicted_store_producer",
        "twin.probes.pilot_diagnostic",
        "evaluation.rca_bench.rca_bench",
        "evaluation.rca_bench.rca_bench_controller",
        "evaluation.rca_bench.rca_bench_fatal",
        "evaluation.rca_bench.rca_bench_input_bundle",
        "evaluation.rca_bench.rca_bench_oracle_builder",
        "fluidmix.explanation.vlm.stage_cosmos_contract",
        "fluidmix.explanation.vlm.stage_cosmos_inputs",
        "evaluation.scoring.stage_cosmos_plan_artifact",
        "fluidmix.explanation.vlm.stage_cosmos_runner",
        "evaluation.scoring.stage_cosmos_scoring",
        "fluidmix.decision.stage",
        "fluidmix.measurement.stage_measurement",
        "fluidmix.perception.baselines.spatial_rgb_mask_predictor",
        "evaluation.scoring.synthetic_decision_scoring",
        "fluidmix.perception.baselines.temporal_rgb_mask_baseline",
        "fluidmix.explanation.vlm.cosmos_contract",
        "fluidmix.explanation.vlm.cosmos_inputs",
        "evaluation.scoring.cosmos_plan_artifact",
        "fluidmix.explanation.vlm.cosmos_runtime",
        "evaluation.scoring.cosmos_scoring",
        "fluidmix.measurement.validation_stage_measurement",
        "fluidmix.measurement.fill_step_calibration",
        "fluidmix.explanation.vlm.development_materializer",
        "fluidmix.explanation.vlm.native_roi_rematerializer",
    }
)


# Frozen byte-contract serializers.  They exist for historical regression tests only.
LEGACY_ORACLE_DIAGNOSTIC_MODULES = frozenset(
    {
        "fluidmix.explanation.template.varied_explanation",
        "fluidmix.explanation.template.stage_cosmos_explanation",
    }
)


# Explicitly inventoried common helpers and adjacent measurements that are neither Part A
# truth nor production/evaluation roots for this carryover boundary.
SHARED_SUPPORT_MODULES = frozenset(
    {
        "fluidmix.colour.absorbance",
        "fluidmix.measurement.cap_delay",
        "fluidmix.core.contracts._common",
        "fluidmix.core.io",
        "fluidmix.storage.artifacts.part_boundary_manifest",
        "fluidmix.core.paths",
        "fluidmix.pipeline.service.contract",
        "fluidmix.pipeline.service.device",
        "fluidmix.pipeline.service.models",
        "fluidmix.pipeline.service.sources",
        "fluidmix.storage.artifacts.real_photometric_manifest",
        "fluidmix.measurement.real_photometric_range",
        "twin.render.render_process",
        "fluidmix.memory.history.vision_dataset_plan",
        "fluidmix.explanation.vlm.constrained",
        "fluidmix.explanation.vlm.image_support",
    }
)


# Reviewed production scripts are explicit so the import/read audit includes them.
PRODUCTION_SCRIPT_ENTRY_POINTS: frozenset[str] = frozenset(
    {
        "cli/fluidmix/perception/materialize_empty_reference_predictions.py",
        "cli/fluidmix/memory/history/materialize_sixth_development_run.py",
        "cli/fluidmix/storage/sqlite/populate_predicted_evidence_store.py",
        "cli/fluidmix/pipeline/prepare_wp7_cohort.py",
        "cli/fluidmix/memory/history/prepare_wp7_development_history.py",
        "cli/fluidmix/pipeline/run_carryover_pipeline.py",
    }
)


EVALUATION_SCRIPT_ENTRY_POINTS = frozenset(
    {
        "cli/fluidmix/perception/cnn/audit_cnn_holdout_recovery.py",
        "cli/fluidmix/explanation/vlm/audit_cosmos_nano_inputs.py",
        "cli/fluidmix/explanation/vlm/audit_cosmos_nano_usable_inputs.py",
        "cli/fluidmix/explanation/vlm/audit_stage_cosmos_nano_inputs.py",
        "cli/fluidmix/explanation/vlm/audit_v3_cosmos_nano_inputs.py",
        "cli/evaluation/scoring/build_stage_cosmos_attempt_plan.py",
        "cli/twin/cohorts/build_f2_varied_image_locators.py",
        "cli/fluidmix/perception/cnn/build_cnn_training_inventory.py",
        "cli/twin/cohorts/build_f2_fill_clip_inputs.py",
        "cli/fluidmix/measurement/build_f2_varied_stage_measurements.py",
        "cli/fluidmix/explanation/vlm/build_model_input_plan_from_manifests.py",
        "cli/fluidmix/measurement/build_stage_measured_evidence.py",
        "cli/fluidmix/memory/history/build_twin_vision_plan.py",
        "cli/fluidmix/measurement/build_validation_measurement_inputs.py",
        "cli/fluidmix/perception/baselines/publish_empty_reference_development.py",
        "cli/fluidmix/measurement/publish_wp7_fill_step_calibration.py",
        "cli/evaluation/scoring/build_v3_cosmos_attempt_plan.py",
        "cli/fluidmix/decision/decide_f2_varied_triads.py",
        "cli/twin/probes/evaluate_appearance_pilot.py",
        "cli/fluidmix/measurement/measure_validation_stage_evidence.py",
        "cli/fluidmix/perception/cnn/replay_cnn_holdout_bounds.py",
        "cli/fluidmix/explanation/vlm/run_cosmos_large_eval.py",
        "cli/fluidmix/explanation/vlm/run_cosmos_nano_usable_eval.py",
        "cli/fluidmix/perception/cnn/run_cnn_empty_control_once.py",
        "cli/evaluation/rca_bench/run_model_bakeoff.py",
        "cli/evaluation/rca_bench/run_rca_bench.py",
        "cli/evaluation/rca_bench/build_rca_bench_inputs.py",
        "cli/evaluation/rca_bench/build_rca_bench_oracle.py",
        "cli/fluidmix/explanation/vlm/run_stage_cosmos_nano_eval.py",
        "cli/fluidmix/perception/baselines/run_spatial_rgb_mask_predictor.py",
        "cli/fluidmix/perception/baselines/run_temporal_rgb_mask_baseline.py",
        "cli/fluidmix/explanation/vlm/run_v3_cosmos_nano_eval.py",
        "cli/evaluation/scoring/score_cosmos_nano_matched.py",
        "cli/fluidmix/perception/cnn/score_cnn_holdout_once.py",
        "cli/evaluation/scoring/score_empty_reference_development.py",
        "cli/evaluation/scoring/score_cosmos_nano_usable.py",
        "cli/evaluation/scoring/score_stage_cosmos_nano.py",
        "cli/evaluation/scoring/score_synthetic_stage_decisions.py",
        "cli/evaluation/scoring/score_v3_cosmos_nano.py",
        "cli/fluidmix/explanation/vlm/stage_cosmos_nano_reasoner.py",
        "cli/twin/postrender/decide_wp1_visibility.py",
        "cli/twin/postrender/package_wp1_test_videos.py",
        "cli/twin/postrender/validate_wp1_acceptance.py",
        "cli/twin/postrender/wp1_dp3_fallback.py",
        "cli/fluidmix/perception/cnn/train_cnn_liquid_mask.py",
        "cli/fluidmix/decision/aggregate_wp7_vlm_attempts.py",
        "cli/fluidmix/explanation/vlm/materialize_wp7_vlm_development_bundle.py",
        "cli/fluidmix/pipeline/prepare_wp7_development_reference_route.py",
        "cli/fluidmix/explanation/vlm/rematerialize_wp7_vlm_native_roi.py",
        "cli/fluidmix/explanation/vlm/run_wp7_vlm_model.py",
        "cli/fluidmix/decision/wp7_vlm_decision.py",
    }
)

# Historical validation/appearance measurements consume synthetic renderer bindings and are
# not active model/scoring entry points.  They remain inventoried above but outside this gate.
HISTORICAL_EVALUATION_SCRIPTS = frozenset(
    {
        "cli/fluidmix/measurement/build_validation_measurement_inputs.py",
        "cli/twin/probes/evaluate_appearance_pilot.py",
        "cli/fluidmix/measurement/measure_validation_stage_evidence.py",
        "cli/twin/postrender/decide_wp1_visibility.py",
        "cli/twin/postrender/package_wp1_test_videos.py",
        "cli/twin/postrender/validate_wp1_acceptance.py",
        "cli/twin/postrender/wp1_dp3_fallback.py",
        "cli/fluidmix/measurement/publish_wp7_fill_step_calibration.py",
    }
)
ACTIVE_EVALUATION_MODEL_SCRIPTS = EVALUATION_SCRIPT_ENTRY_POINTS.difference(
    HISTORICAL_EVALUATION_SCRIPTS
)

LEGACY_ORACLE_DIAGNOSTIC_SCRIPT_ENTRY_POINTS = frozenset(
    {"cli/fluidmix/pipeline/build_f2_varied_demo.py"}
)

POST_INFERENCE_LABEL_JOIN_ROOTS = frozenset(
    {
        "evaluation.scoring.empty_reference_development_scoring",
        "evaluation.scoring.synthetic_decision_scoring",
    }
)

# WP7 production root.  Its predictor and explanation backends stay pluggable capabilities.
WP7_FUTURE_ROOTS: frozenset[str] = frozenset({"fluidmix.pipeline.carryover_pipeline"})
WP7_PRODUCTION_PROJECTOR_MODULE = "fluidmix.storage.artifacts.part_b_evidence"
PRESET_CATALOG_RELATIVE_PATH = "configs/tank_colour_presets.json"
PALETTE_CUT_REEL_ARTIFACT_TYPE = "fluidmix_palette_cut_reel_v2"
PART_A_ARTIFACT_TYPES = frozenset({PALETTE_CUT_REEL_ARTIFACT_TYPE})
PART_A_REEL_ROOT_RELATIVE_PATH = "outputs/part-a/palette-cut-reels"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INFERENCE_SEAL_TOKEN = object()


def _local_name(name: str) -> str | None:
    return name if name.split(".", 1)[0] in {"fluidmix", "twin", "evaluation"} else None


def parse_local_imports(
    source: str,
    *,
    module_name: str,
    is_package: bool = False,
) -> frozenset[str]:
    """Return statically named local imports, including constant dynamic calls.

    Dynamic expressions are intentionally ignored: a production entry point needing one
    must use a constant name so the ownership audit can follow it.
    """

    tree = ast.parse(source)
    package = module_name if is_package else module_name.rpartition(".")[0]
    imports: set[str] = set()
    importlib_aliases = {"importlib"}
    import_module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = _local_name(alias.name)
                if local is not None and local not in {"fluidmix", "twin", "evaluation"}:
                    imports.add(local)
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            package_member_import = node.level > 0 and node.module is None
            target = node.module or ""
            if node.level:
                relative = "." * node.level + target
                try:
                    target = importlib.util.resolve_name(relative, package)
                except (ImportError, ValueError):
                    continue
            if target == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        import_module_aliases.add(alias.asname or alias.name)
                continue
            local = _local_name(target)
            if local is not None and local not in {"fluidmix", "twin", "evaluation"}:
                imports.add(local)
            if target.split(".", 1)[0] in {"fluidmix", "twin", "evaluation"} or package_member_import:
                for alias in node.names:
                    if alias.name != "*":
                        imports.add(f"{target}.{alias.name}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        dynamic_import = (
            isinstance(function, ast.Name)
            and function.id in import_module_aliases | {"__import__"}
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id in importlib_aliases
        )
        name_node = node.args[0]
        if not dynamic_import or not isinstance(name_node, ast.Constant) or not isinstance(
            name_node.value, str
        ):
            continue
        target = name_node.value
        if target.startswith("."):
            try:
                target = importlib.util.resolve_name(target, package)
            except (ImportError, ValueError):
                continue
        local = _local_name(target)
        if local is not None and local not in {"fluidmix", "twin", "evaluation"}:
            imports.add(local)

    return frozenset(imports)


def _module_source(repo_root: Path, module_name: str) -> tuple[Path, bool] | None:
    relative = Path("src").joinpath(*module_name.split("."))
    module_file = repo_root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file, False
    package_file = repo_root / relative / "__init__.py"
    if package_file.is_file():
        return package_file, True
    return None


def walk_local_import_graph(
    repo_root: str | Path,
    roots: frozenset[str] | set[str] | tuple[str, ...],
) -> dict[str, frozenset[str]]:
    """Walk all repository-local imports reachable from modules or script paths."""

    root = Path(repo_root).resolve()
    queue = deque(sorted(roots))
    graph: dict[str, frozenset[str]] = {}
    while queue:
        name = queue.popleft()
        if name in graph:
            continue
        if name.startswith("cli/"):
            source_path = root / name
            module_name = "__part_boundary_script__"
            is_package = False
        else:
            located = _module_source(root, name)
            if located is None:
                graph[name] = frozenset()
                continue
            source_path, is_package = located
            module_name = name
        source = source_path.read_text(encoding="utf-8")
        imports = parse_local_imports(
            source,
            module_name=module_name,
            is_package=is_package,
        )
        graph[name] = imports
        for imported in sorted(imports):
            if imported not in graph and _module_source(root, imported) is not None:
                queue.append(imported)
    return graph


def _forbidden_owner(module_name: str, forbidden: frozenset[str]) -> str | None:
    for owner in forbidden:
        if module_name == owner or module_name.startswith(owner + "."):
            return owner
    return None


def _dependency_findings(
    graph: dict[str, frozenset[str]],
    roots: frozenset[str],
    forbidden: frozenset[str],
) -> tuple[str, ...]:
    findings: set[str] = set()
    for root in sorted(roots):
        queue: deque[tuple[str, tuple[str, ...]]] = deque([(root, (root,))])
        visited: set[str] = set()
        while queue:
            owner, chain = queue.popleft()
            if owner in visited:
                continue
            visited.add(owner)
            for imported in sorted(graph.get(owner, ())):
                next_chain = (*chain, imported)
                forbidden_owner = _forbidden_owner(imported, forbidden)
                if forbidden_owner is not None:
                    findings.add(" -> ".join(next_chain))
                    continue
                if imported in graph:
                    queue.append((imported, next_chain))
    return tuple(sorted(findings))


def audit_current_dependency_isolation(repo_root: str | Path) -> tuple[str, ...]:
    """Report forbidden transitive dependencies for every active registered root."""

    production_roots = frozenset(
        PRODUCTION_PART_B_ROOTS | PRODUCTION_SCRIPT_ENTRY_POINTS
    )
    production_graph = walk_local_import_graph(repo_root, production_roots)
    production_forbidden = frozenset(
        PART_A_ONLY_MODULES
        | EVALUATION_ONLY_MODULES
        | LEGACY_ORACLE_DIAGNOSTIC_MODULES
    )
    production = _dependency_findings(
        production_graph,
        production_roots,
        production_forbidden,
    )
    evaluation_graph = walk_local_import_graph(
        repo_root, ACTIVE_EVALUATION_MODEL_SCRIPTS
    )
    evaluation = _dependency_findings(
        evaluation_graph,
        ACTIVE_EVALUATION_MODEL_SCRIPTS,
        frozenset(PART_A_ONLY_MODULES | LEGACY_ORACLE_DIAGNOSTIC_MODULES),
    )
    return tuple(sorted((*production, *evaluation)))


class ForbiddenBoundaryModuleFinder(MetaPathFinder):
    """Fail an import before code in a forbidden ownership domain can execute."""

    def __init__(self, forbidden_modules: frozenset[str]) -> None:
        self.forbidden_modules = forbidden_modules

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if _forbidden_owner(fullname, self.forbidden_modules) is not None:
            raise ModuleNotFoundError(
                f"forbidden Part boundary import: {fullname}"
            )
        return None


def install_forbidden_module_finder(
    forbidden_modules: frozenset[str] | set[str],
) -> ForbiddenBoundaryModuleFinder:
    """Install a fresh-process finder for the exact registered module families."""

    forbidden = frozenset(forbidden_modules)
    already_loaded = sorted(
        name
        for name in sys.modules
        if _forbidden_owner(name, forbidden) is not None
    )
    if already_loaded:
        raise RuntimeError(
            "forbidden modules were loaded before boundary finder installation: "
            + ", ".join(already_loaded)
        )
    finder = ForbiddenBoundaryModuleFinder(forbidden)
    sys.meta_path.insert(0, finder)
    return finder


class ExactPartAReadGuard:
    """Audit hook that denies exact files and descendants of explicit roots."""

    def __init__(
        self,
        exact_files: tuple[Path, ...],
        forbidden_roots: tuple[Path, ...],
    ) -> None:
        self.exact_files = frozenset(exact_files)
        self.forbidden_roots = forbidden_roots

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        if event not in {"open", "fluidmix.stable_read_path"} or not arguments:
            return
        raw_path = arguments[0]
        if isinstance(raw_path, int) or not isinstance(raw_path, (str, bytes)):
            return
        try:
            path = Path(os.fsdecode(raw_path)).resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            return
        if path in self.exact_files or any(
            path == root or path.is_relative_to(root) for root in self.forbidden_roots
        ):
            raise PermissionError(f"forbidden Part A read: {path}")


def install_exact_read_guard(
    *,
    exact_files: tuple[str | Path, ...] | list[str | Path],
    forbidden_roots: tuple[str | Path, ...] | list[str | Path],
) -> ExactPartAReadGuard:
    """Install a process-local audit hook using paths, never filename keywords."""

    exact = tuple(Path(path).resolve(strict=False) for path in exact_files)
    roots = tuple(Path(path).resolve(strict=False) for path in forbidden_roots)
    guard = ExactPartAReadGuard(exact, roots)
    sys.addaudithook(guard)
    return guard


def install_production_part_b_read_guard(
    repo_root: str | Path,
    *,
    sim_metadata_fixture_files: tuple[str | Path, ...] = (),
    part_a_receipt_roots: tuple[str | Path, ...] = (),
    part_a_label_roots: tuple[str | Path, ...] = (),
) -> ExactPartAReadGuard:
    """Protect exact Part-A inputs and classified receipt/label/reel roots."""

    root = Path(repo_root).resolve(strict=False)
    authoritative_reel_root = root / PART_A_REEL_ROOT_RELATIVE_PATH
    return install_exact_read_guard(
        exact_files=(
            root / PRESET_CATALOG_RELATIVE_PATH,
            *sim_metadata_fixture_files,
        ),
        forbidden_roots=(
            *part_a_receipt_roots,
            *part_a_label_roots,
            authoritative_reel_root,
        ),
    )


def validate_part_a_artifact_output(
    output_path: str | Path,
    *,
    repo_root: str | Path,
    artifact_type: str,
) -> tuple[Path, Path]:
    """Require an output below an explicit root for a registered Part-A artifact type."""

    if artifact_type not in PART_A_ARTIFACT_TYPES:
        raise ValueError(f"unregistered Part A artifact type: {artifact_type}")
    repository = Path(repo_root).expanduser().resolve(strict=True)
    root = (repository / PART_A_REEL_ROOT_RELATIVE_PATH).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("classified Part A output root must be a directory")
    output = Path(output_path).expanduser()
    parent = output.parent.resolve(strict=True)
    candidate = parent / output.name
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("reel output must be below its classified Part A output root")
    return candidate, root


class SealedInferenceArtifact:
    """Opaque proof that inference bytes existed and matched a declared digest."""

    __slots__ = ("path", "sha256", "artifact_type", "_factory_token", "_sealed")

    def __init__(
        self,
        path: Path,
        sha256: str,
        artifact_type: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _INFERENCE_SEAL_TOKEN:
            raise TypeError("sealed inference artifacts are factory-only")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", sha256)
        object.__setattr__(self, "artifact_type", artifact_type)
        object.__setattr__(self, "_factory_token", _INFERENCE_SEAL_TOKEN)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("sealed inference artifacts are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("sealed inference artifacts are immutable")
        object.__delattr__(self, name)


def seal_saved_inference_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    artifact_type: str,
) -> SealedInferenceArtifact:
    """Verify saved inference bytes before a scoring-only label join is authorized."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("sealed inference artifact must be a regular file")
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("sealed inference artifact hash must be lowercase SHA-256")
    if not isinstance(artifact_type, str) or not artifact_type:
        raise ValueError("sealed inference artifact type must be nonempty text")
    observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if observed != expected_sha256:
        raise ValueError("sealed inference artifact SHA-256 mismatch")
    return SealedInferenceArtifact(
        candidate.resolve(strict=True),
        observed,
        artifact_type,
        _factory_token=_INFERENCE_SEAL_TOKEN,
    )


def authorize_post_inference_label_join(
    caller_module: str,
    sealed_inference: object,
) -> SealedInferenceArtifact:
    """Authorize label access only for a registered scoring root after sealing."""

    if caller_module not in POST_INFERENCE_LABEL_JOIN_ROOTS:
        raise PermissionError(
            f"Part A label joins are scoring-only: {caller_module}"
        )
    if (
        not isinstance(sealed_inference, SealedInferenceArtifact)
        or sealed_inference._factory_token is not _INFERENCE_SEAL_TOKEN
    ):
        raise TypeError("label join requires a sealed inference artifact")
    path = sealed_inference.path
    if (
        path.is_symlink()
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != sealed_inference.sha256
    ):
        raise ValueError("sealed inference artifact drifted before label join")
    return sealed_inference


def assert_wp7_root_registration(
    root_module: str,
    *,
    direct_dependencies: set[str] | frozenset[str],
) -> None:
    """Require a future WP7 root to register the projector and exclude diagnostics."""

    if (
        root_module not in WP7_FUTURE_ROOTS
        or root_module not in PRODUCTION_PART_B_ROOTS
    ):
        raise ValueError(f"unregistered WP7 root: {root_module}")
    assert_wp7_dependency_policy(direct_dependencies)


def assert_wp7_dependency_policy(
    direct_dependencies: set[str] | frozenset[str],
) -> None:
    """Allow the production projector but reject frozen diagnostic serializers."""

    if WP7_PRODUCTION_PROJECTOR_MODULE not in direct_dependencies:
        raise ValueError("WP7 root must use the production projector")
    if any(
        _forbidden_owner(name, LEGACY_ORACLE_DIAGNOSTIC_MODULES) is not None
        for name in direct_dependencies
    ):
        raise ValueError("WP7 root imports a legacy oracle diagnostic")


_MODULE_OWNERSHIP_CATEGORIES = (
    ("part_a_only", PART_A_ONLY_MODULES),
    ("production_part_b_root", PRODUCTION_PART_B_ROOTS),
    ("evaluation_only", EVALUATION_ONLY_MODULES),
    ("legacy_oracle_diagnostic", LEGACY_ORACLE_DIAGNOSTIC_MODULES),
    ("shared_support", SHARED_SUPPORT_MODULES),
)


def module_ownership(module_name: str) -> str:
    """Return the one reviewed role for a boundary module, failing closed."""

    matches = [role for role, modules in _MODULE_OWNERSHIP_CATEGORIES if module_name in modules]
    if len(matches) != 1:
        raise ValueError(f"unregistered boundary module: {module_name}")
    return matches[0]


def validate_boundary_manifest(repo_root: str | Path) -> tuple[str, ...]:
    """Validate disjoint registration, source existence, and dependency isolation."""

    root = Path(repo_root).resolve()
    findings: list[str] = []
    seen: dict[str, str] = {}
    for role, modules in _MODULE_OWNERSHIP_CATEGORIES:
        for module_name in sorted(modules):
            previous = seen.setdefault(module_name, role)
            if previous != role:
                findings.append(
                    f"module has multiple ownership roles: {module_name}: {previous}, {role}"
                )
            if _module_source(root, module_name) is None:
                findings.append(f"registered module source is missing: {module_name}")
    script_seen: set[str] = set()
    for role, scripts in (
        ("production", PRODUCTION_SCRIPT_ENTRY_POINTS),
        ("evaluation", EVALUATION_SCRIPT_ENTRY_POINTS),
        ("legacy oracle diagnostic", LEGACY_ORACLE_DIAGNOSTIC_SCRIPT_ENTRY_POINTS),
    ):
        for relative in sorted(scripts):
            if relative in script_seen:
                findings.append(f"script has multiple ownership roles: {relative}")
            script_seen.add(relative)
            if not (root / relative).is_file():
                findings.append(f"registered {role} script is missing: {relative}")
    if not ACTIVE_EVALUATION_MODEL_SCRIPTS <= EVALUATION_SCRIPT_ENTRY_POINTS:
        findings.append("active evaluation scripts are not fully registered")
    if not HISTORICAL_EVALUATION_SCRIPTS <= EVALUATION_SCRIPT_ENTRY_POINTS:
        findings.append("historical evaluation scripts are not fully registered")
    if not POST_INFERENCE_LABEL_JOIN_ROOTS <= EVALUATION_ONLY_MODULES:
        findings.append("post-inference join roots are not evaluation-only")
    if not WP7_FUTURE_ROOTS <= PRODUCTION_PART_B_ROOTS:
        findings.append("WP7 future roots are not production-registered")
    if WP7_FUTURE_ROOTS:
        wp7_graph = walk_local_import_graph(root, WP7_FUTURE_ROOTS)
        for wp7_root in sorted(WP7_FUTURE_ROOTS):
            try:
                assert_wp7_root_registration(
                    wp7_root,
                    direct_dependencies=set(wp7_graph.get(wp7_root, ())),
                )
            except ValueError as error:
                findings.append(str(error))
    current_modules = {
        f"{package}."
        + path.relative_to(root / "src" / package)
        .with_suffix("")
        .as_posix()
        .replace("/", ".")
        for package in ("fluidmix", "twin", "evaluation")
        for path in (root / "src" / package).rglob("*.py")
        if path.name != "__init__.py"
    }
    registered_modules = set().union(
        *(modules for _, modules in _MODULE_OWNERSHIP_CATEGORIES)
    )
    for module_name in sorted(current_modules.difference(registered_modules)):
        findings.append(f"current module is not ownership-registered: {module_name}")
    for module_name in sorted(registered_modules.difference(current_modules)):
        findings.append(f"registered module is not current: {module_name}")
    findings.extend(audit_current_dependency_isolation(root))
    return tuple(sorted(findings))


__all__ = [
    "ACTIVE_EVALUATION_MODEL_SCRIPTS",
    "EVALUATION_ONLY_MODULES",
    "EVALUATION_SCRIPT_ENTRY_POINTS",
    "ExactPartAReadGuard",
    "ForbiddenBoundaryModuleFinder",
    "HISTORICAL_EVALUATION_SCRIPTS",
    "LEGACY_ORACLE_DIAGNOSTIC_MODULES",
    "LEGACY_ORACLE_DIAGNOSTIC_SCRIPT_ENTRY_POINTS",
    "PART_A_ONLY_MODULES",
    "PART_A_ARTIFACT_TYPES",
    "PART_A_REEL_ROOT_RELATIVE_PATH",
    "PALETTE_CUT_REEL_ARTIFACT_TYPE",
    "POST_INFERENCE_LABEL_JOIN_ROOTS",
    "PRESET_CATALOG_RELATIVE_PATH",
    "PRODUCTION_PART_B_ROOTS",
    "PRODUCTION_SCRIPT_ENTRY_POINTS",
    "SealedInferenceArtifact",
    "SHARED_SUPPORT_MODULES",
    "WP7_FUTURE_ROOTS",
    "WP7_PRODUCTION_PROJECTOR_MODULE",
    "assert_wp7_dependency_policy",
    "assert_wp7_root_registration",
    "authorize_post_inference_label_join",
    "audit_current_dependency_isolation",
    "install_forbidden_module_finder",
    "install_exact_read_guard",
    "install_production_part_b_read_guard",
    "module_ownership",
    "parse_local_imports",
    "seal_saved_inference_artifact",
    "validate_boundary_manifest",
    "validate_part_a_artifact_output",
    "walk_local_import_graph",
]
