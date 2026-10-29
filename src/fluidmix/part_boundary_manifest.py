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
        "fluidmix.brev_video_cohort",
        "fluidmix.capseal_structural",
        "fluidmix.contracts.sim_metadata",
        "fluidmix.explicit_mass_probe",
        "fluidmix.f2_downstream_binding",
        "fluidmix.f2_receipt_manifest_publisher",
        "fluidmix.f2_stage_cohort_materializer",
        "fluidmix.f2_varied_stage_materializer",
        "fluidmix.full_transition_manifest",
        "fluidmix.isosurface_geometry",
        "fluidmix.isosurface_mesh_proxy",
        "fluidmix.nuisance_authoring",
        "fluidmix.paint_ball_emitter",
        "fluidmix.particle_array_probe",
        "fluidmix.particle_diffusion",
        "fluidmix.particle_palette",
        "fluidmix.physx_case_plan",
        "fluidmix.physx_composition_cohort",
        "fluidmix.physx_dispense_residual",
        "fluidmix.physx_one_run_manifest_v1",
        "fluidmix.physx_particle_contract",
        "fluidmix.physx_residual_case",
        "fluidmix.physx_residual_pilot",
        "fluidmix.physx_residual_schedule",
        "fluidmix.physx_run_variation",
        "fluidmix.physx_three_pour",
        "fluidmix.physx_train_residual_manifest_v2",
        "fluidmix.physx_varied_cohort_manifest_v3",
        "fluidmix.physx_varied_cohort_manifest_v4",
        "fluidmix.physx_volume_gate",
        "fluidmix.pilot_dataset",
        "fluidmix.parse_registration_probe",
        "fluidmix.render_capture",
        "fluidmix.resolution_probe",
        "fluidmix.review_contact_sheet",
        "fluidmix.review_video",
        "fluidmix.rigid_vial_transport",
        "fluidmix.run_sequence_dataset",
        "fluidmix.spatial_omnisurface_probe",
        "fluidmix.spatial_surface_color",
        "fluidmix.tank_colour_presets",
        "fluidmix.tank_station",
        "fluidmix.test_recipe_generator",
        "fluidmix.timeline_probe",
        "fluidmix.transform_attestation",
        "fluidmix.transport_stage",
        "fluidmix.transmissive_optics",
        "fluidmix.twin_appearance",
        "fluidmix.twin_stage_appearance",
        "fluidmix.twin_video",
        "fluidmix.validation_measurement_binding",
        "fluidmix.vial_transport",
        "fluidmix.volume_height_curve",
        "fluidmix.wp1_postrender",
    }
)


# Callable Part B boundaries that expose the reviewed production path.  Synthetic capture,
# model-plan, triage, and direct-mask helpers are evaluation-only below.
PRODUCTION_PART_B_ROOTS = frozenset(
    {
        "fluidmix.contracts.evidence",
        "fluidmix.deploy_input_policy",
        "fluidmix.evidence_store",
        "fluidmix.part_b_evidence",
        "fluidmix.rgb_mask_prediction",
        "fluidmix.vision_measurement",
    }
)


# These modules may score saved outputs or run model experiments.  They are never production
# evidence constructors and may not be imported by a production root.
EVALUATION_ONLY_MODULES = frozenset(
    {
        "fluidmix.capture_manifest_adapter",
        "fluidmix.cosmos_contract_runner",
        "fluidmix.cosmos_nano_contract",
        "fluidmix.cosmos_nano_input_audit",
        "fluidmix.cosmos_nano_model_view",
        "fluidmix.cosmos_nano_runner",
        "fluidmix.cosmos_nano_scoring",
        "fluidmix.cosmos_nano_usable_contract",
        "fluidmix.cosmos_usable_output",
        "fluidmix.cosmos_usable_scoring",
        "fluidmix.explanation_scoring",
        "fluidmix.explanation_contract",
        "fluidmix.fill_clips",
        "fluidmix.f2_varied_demo",
        "fluidmix.f2_varied_image_locator",
        "fluidmix.f2_varied_stage_measurement",
        "fluidmix.f2_varied_triage",
        "fluidmix.model_bakeoff_runner",
        "fluidmix.model_input_plan_builder",
        "fluidmix.pilot_diagnostic",
        "fluidmix.rca_bench",
        "fluidmix.rca_bench_controller",
        "fluidmix.rca_bench_fatal",
        "fluidmix.stage_cosmos_contract",
        "fluidmix.stage_cosmos_inputs",
        "fluidmix.stage_cosmos_plan_artifact",
        "fluidmix.stage_cosmos_runner",
        "fluidmix.stage_cosmos_scoring",
        "fluidmix.stage_decision",
        "fluidmix.stage_measurement",
        "fluidmix.synthetic_decision_scoring",
        "fluidmix.temporal_rgb_mask_baseline",
        "fluidmix.v3_cosmos_contract",
        "fluidmix.v3_cosmos_inputs",
        "fluidmix.v3_cosmos_plan_artifact",
        "fluidmix.v3_cosmos_runtime",
        "fluidmix.v3_cosmos_scoring",
        "fluidmix.validation_stage_measurement",
    }
)


# Frozen byte-contract serializers.  They exist for historical regression tests only.
LEGACY_ORACLE_DIAGNOSTIC_MODULES = frozenset(
    {
        "fluidmix.f2_varied_explanation",
        "fluidmix.stage_cosmos_explanation",
    }
)


# Explicitly inventoried common helpers and adjacent measurements that are neither Part A
# truth nor production/evaluation roots for this carryover boundary.
SHARED_SUPPORT_MODULES = frozenset(
    {
        "fluidmix.absorbance",
        "fluidmix.cap_delay",
        "fluidmix.contracts._common",
        "fluidmix.part_boundary_manifest",
        "fluidmix.paths",
        "fluidmix.real_photometric_manifest",
        "fluidmix.real_photometric_range",
        "fluidmix.render_process",
        "fluidmix.vision_dataset_plan",
    }
)


# INN-58 exposes the new production path as Python APIs only.  The existing scripts bind
# synthetic captures or run scoring/model experiments and are evaluation-only.
PRODUCTION_SCRIPT_ENTRY_POINTS: frozenset[str] = frozenset()


EVALUATION_SCRIPT_ENTRY_POINTS = frozenset(
    {
        "scripts/audit_cosmos_nano_inputs.py",
        "scripts/audit_cosmos_nano_usable_inputs.py",
        "scripts/audit_stage_cosmos_nano_inputs.py",
        "scripts/audit_v3_cosmos_nano_inputs.py",
        "scripts/build_stage_cosmos_attempt_plan.py",
        "scripts/build_f2_varied_image_locators.py",
        "scripts/build_f2_varied_stage_measurements.py",
        "scripts/build_model_input_plan_from_manifests.py",
        "scripts/build_stage_measured_evidence.py",
        "scripts/build_twin_vision_plan.py",
        "scripts/build_validation_measurement_inputs.py",
        "scripts/build_v3_cosmos_attempt_plan.py",
        "scripts/decide_f2_varied_triads.py",
        "scripts/evaluate_appearance_pilot.py",
        "scripts/measure_validation_stage_evidence.py",
        "scripts/run_cosmos_large_eval.py",
        "scripts/run_cosmos_nano_usable_eval.py",
        "scripts/run_model_bakeoff.py",
        "scripts/run_rca_bench.py",
        "scripts/run_stage_cosmos_nano_eval.py",
        "scripts/run_temporal_rgb_mask_baseline.py",
        "scripts/run_v3_cosmos_nano_eval.py",
        "scripts/score_cosmos_nano_matched.py",
        "scripts/score_cosmos_nano_usable.py",
        "scripts/score_stage_cosmos_nano.py",
        "scripts/score_synthetic_stage_decisions.py",
        "scripts/score_v3_cosmos_nano.py",
        "scripts/stage_cosmos_nano_reasoner.py",
    }
)

# Historical validation/appearance measurements consume synthetic renderer bindings and are
# not active model/scoring entry points.  They remain inventoried above but outside this gate.
HISTORICAL_EVALUATION_SCRIPTS = frozenset(
    {
        "scripts/build_validation_measurement_inputs.py",
        "scripts/evaluate_appearance_pilot.py",
        "scripts/measure_validation_stage_evidence.py",
    }
)
ACTIVE_EVALUATION_MODEL_SCRIPTS = EVALUATION_SCRIPT_ENTRY_POINTS.difference(
    HISTORICAL_EVALUATION_SCRIPTS
)

LEGACY_ORACLE_DIAGNOSTIC_SCRIPT_ENTRY_POINTS = frozenset(
    {"scripts/build_f2_varied_demo.py"}
)

POST_INFERENCE_LABEL_JOIN_ROOTS = frozenset(
    {"fluidmix.synthetic_decision_scoring"}
)

# WP7 has no implementation in INN-58.  Its eventual import root must be added here and
# to PRODUCTION_PART_B_ROOTS in the same reviewed change.
WP7_FUTURE_ROOTS: frozenset[str] = frozenset()
WP7_PRODUCTION_PROJECTOR_MODULE = "fluidmix.part_b_evidence"
PRESET_CATALOG_RELATIVE_PATH = "configs/tank_colour_presets.json"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INFERENCE_SEAL_TOKEN = object()


def _local_name(name: str) -> str | None:
    return name if name == "fluidmix" or name.startswith("fluidmix.") else None


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
                if local is not None and local != "fluidmix":
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
            if local is not None and local != "fluidmix":
                imports.add(local)
            if target == "fluidmix" or package_member_import:
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
        if local is not None and local != "fluidmix":
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
        if name.startswith("scripts/"):
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
    """Protect the catalog plus explicitly bound fixture, receipt, and label targets."""

    root = Path(repo_root).resolve(strict=False)
    return install_exact_read_guard(
        exact_files=(
            root / PRESET_CATALOG_RELATIVE_PATH,
            *sim_metadata_fixture_files,
        ),
        forbidden_roots=(*part_a_receipt_roots, *part_a_label_roots),
    )


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
        "fluidmix."
        + path.relative_to(root / "src" / "fluidmix")
        .with_suffix("")
        .as_posix()
        .replace("/", ".")
        for path in (root / "src" / "fluidmix").rglob("*.py")
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
    "walk_local_import_graph",
]
