"""Deterministic liquid-mask colour measurements for frozen synthetic plans."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets

from .deploy_input_policy import validate_deploy_input_fields


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOCATOR_FIELDS = {"rgb_path", "rgb_sha256", "mask_path", "mask_sha256"}
_ORACLE_VALUES = {
    "none",
    "first_only",
    "first_two",
    "all_three",
    "after_t200",
    "after_t400",
    "after_t100",
}
_DECISION_CONTEXT_FIELDS = (
    "decision_id",
    "family_id",
    "batch_id",
    "cohort",
    "partition",
    "previous_sim_recipe_id",
    "current_sim_recipe_id",
    "previous_programmed_ml",
    "current_programmed_ml",
    "programmed_dispense_order",
    "vial_id",
    "vial_position",
    "decision_timing",
)
_TRUSTED_MEASUREMENT_TOKEN = object()


class TrustedMeasuredEvidenceResult(dict):
    """Serializable measured evidence with non-serialized live capabilities."""

    __slots__ = (
        "_trusted_bindings",
        "_trusted_registry",
        "_factory_token",
        "_sealed",
    )

    def __init__(
        self,
        value: Mapping[str, object],
        bindings: tuple["TrustedPredictedMeasurementBindingV1", ...],
        registry: object,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _TRUSTED_MEASUREMENT_TOKEN:
            raise TypeError("trusted measured evidence results are factory-only")
        dict.__init__(self, value)
        object.__setattr__(self, "_trusted_bindings", bindings)
        object.__setattr__(self, "_trusted_registry", registry)
        object.__setattr__(self, "_factory_token", _TRUSTED_MEASUREMENT_TOKEN)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("trusted measured evidence capabilities are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("trusted measured evidence capabilities are immutable")
        object.__delattr__(self, name)


def trusted_measurement_bindings_from_result_v1(
    value: object,
) -> tuple["TrustedPredictedMeasurementBindingV1", ...]:
    """Extract live capabilities; serialized/reloaded dictionaries are rejected."""

    try:
        trusted = (
            isinstance(value, TrustedMeasuredEvidenceResult)
            and value._factory_token is _TRUSTED_MEASUREMENT_TOKEN
        )
        bindings = value._trusted_bindings if trusted else None
    except AttributeError:
        trusted = False
        bindings = None
    if (
        not trusted
        or not isinstance(bindings, tuple)
        or any(
            not is_trusted_predicted_measurement_binding_v1(binding)
            for binding in bindings
        )
    ):
        raise TypeError("measurement bindings require a live trusted adapter result")
    return bindings


def trusted_prediction_registry_from_result_v1(value: object) -> object:
    """Extract the live registry capability from a trusted adapter result."""

    from .rgb_mask_prediction import is_trusted_predicted_mask_registry_v1
    from .wp7_rgb_runtime import is_trusted_real_rgb_mask_registry_v1

    try:
        trusted = (
            isinstance(value, TrustedMeasuredEvidenceResult)
            and value._factory_token is _TRUSTED_MEASUREMENT_TOKEN
        )
        registry = value._trusted_registry if trusted else None
    except AttributeError:
        trusted = False
        registry = None
    if not trusted or not (
        is_trusted_predicted_mask_registry_v1(registry)
        or is_trusted_real_rgb_mask_registry_v1(registry)
    ):
        raise TypeError("prediction registry requires a live trusted adapter result")
    return registry


class TrustedPredictedMeasurementBindingV1:
    """Opaque receipt binding one measurement/evidence ID to a trusted locator."""

    __slots__ = (
        "_image_id",
        "_measurement_id",
        "_evidence_id",
        "_rgb_sha256",
        "_mask_sha256",
        "_entry_sha256",
        "_registry_sha256",
        "_validity",
        "_limit_codes",
        "_median_lab",
        "_mad_lab",
        "_iqr_lab",
        "_measurement_sha256",
        "_factory_token",
        "_sealed",
    )

    def __init__(
        self,
        *,
        image_id: str,
        measurement_id: str,
        evidence_id: str,
        rgb_sha256: str,
        mask_sha256: str,
        entry_sha256: str,
        registry_sha256: str,
        validity: str,
        limit_codes: tuple[str, ...],
        median_lab: tuple[float, float, float] | None,
        mad_lab: tuple[float, float, float] | None,
        iqr_lab: tuple[float, float, float] | None,
        measurement_sha256: str,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _TRUSTED_MEASUREMENT_TOKEN:
            raise TypeError("trusted predicted measurement bindings are factory-only")
        object.__setattr__(self, "_image_id", image_id)
        object.__setattr__(self, "_measurement_id", measurement_id)
        object.__setattr__(self, "_evidence_id", evidence_id)
        object.__setattr__(self, "_rgb_sha256", rgb_sha256)
        object.__setattr__(self, "_mask_sha256", mask_sha256)
        object.__setattr__(self, "_entry_sha256", entry_sha256)
        object.__setattr__(self, "_registry_sha256", registry_sha256)
        object.__setattr__(self, "_validity", validity)
        object.__setattr__(self, "_limit_codes", limit_codes)
        object.__setattr__(self, "_median_lab", median_lab)
        object.__setattr__(self, "_mad_lab", mad_lab)
        object.__setattr__(self, "_iqr_lab", iqr_lab)
        object.__setattr__(self, "_measurement_sha256", measurement_sha256)
        object.__setattr__(self, "_factory_token", _TRUSTED_MEASUREMENT_TOKEN)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("trusted predicted measurement bindings are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("trusted predicted measurement bindings are immutable")
        object.__delattr__(self, name)

    @property
    def image_id(self) -> str:
        return self._image_id

    @property
    def measurement_id(self) -> str:
        return self._measurement_id

    @property
    def evidence_id(self) -> str:
        return self._evidence_id

    @property
    def rgb_sha256(self) -> str:
        return self._rgb_sha256

    @property
    def mask_sha256(self) -> str:
        return self._mask_sha256

    @property
    def entry_sha256(self) -> str:
        return self._entry_sha256

    @property
    def registry_sha256(self) -> str:
        return self._registry_sha256

    @property
    def validity(self) -> str:
        return self._validity

    @property
    def limit_codes(self) -> tuple[str, ...]:
        return self._limit_codes

    @property
    def median_lab(self) -> tuple[float, float, float] | None:
        return self._median_lab

    @property
    def mad_lab(self) -> tuple[float, float, float] | None:
        return self._mad_lab

    @property
    def iqr_lab(self) -> tuple[float, float, float] | None:
        return self._iqr_lab

    @property
    def measurement_sha256(self) -> str:
        return self._measurement_sha256

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": "trusted-predicted-measurement-binding-v1",
            "image_id": self._image_id,
            "measurement_id": self._measurement_id,
            "evidence_id": self._evidence_id,
            "rgb_sha256": self._rgb_sha256,
            "mask_sha256": self._mask_sha256,
            "entry_sha256": self._entry_sha256,
            "registry_sha256": self._registry_sha256,
            "validity": self._validity,
            "limit_codes": list(self._limit_codes),
            "measurement_sha256": self._measurement_sha256,
        }


def is_trusted_predicted_measurement_binding_v1(value: object) -> bool:
    try:
        if (
            not isinstance(value, TrustedPredictedMeasurementBindingV1)
            or value._factory_token is not _TRUSTED_MEASUREMENT_TOKEN
        ):
            return False
        text_values = (value._image_id, value._measurement_id, value._evidence_id)
        hashes = (
            value._rgb_sha256,
            value._mask_sha256,
            value._entry_sha256,
            value._registry_sha256,
            value._measurement_sha256,
        )
        if any(not isinstance(item, str) or not item for item in text_values):
            return False
        if any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in hashes
        ):
            return False
        if value._validity not in {"valid", "invalid"}:
            return False
        if not isinstance(value._limit_codes, tuple) or any(
            not isinstance(code, str) for code in value._limit_codes
        ):
            return False
        lab_values = (value._median_lab, value._mad_lab, value._iqr_lab)
        if any(
            lab is not None
            and (
                not isinstance(lab, tuple)
                or len(lab) != 3
                or any(
                    isinstance(channel, bool)
                    or not isinstance(channel, (int, float))
                    or not math.isfinite(channel)
                    for channel in lab
                )
            )
            for lab in lab_values
        ):
            return False
        return (value._validity == "valid") == all(
            lab is not None for lab in lab_values
        )
    except (AttributeError, OverflowError, TypeError):
        return False


def _invalid_measurement(image_id: str, code: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "validity": "invalid",
        "limit_codes": [code],
        "rgb_sha256": None,
        "mask_sha256": None,
        "liquid_pixel_count": None,
        "mask_coverage_fraction": None,
        "mask_height_px": None,
        "lab_raw": None,
        "lab_6dp": None,
        "unit": "cie_lab_d65_2deg",
    }


def _read_bound_bytes(path_text: str, expected_hash: str, role: str) -> tuple[bytes, str] | str:
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError(f"{role} path must be absolute")
    try:
        data = path.read_bytes()
    except OSError:
        return f"{role}_missing"
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_hash:
        return f"{role}_hash_mismatch"
    return data, digest


def measure_image(
    image_id: object,
    locator: object,
    *,
    _include_dispersion: bool = False,
) -> dict[str, object]:
    """Measure one SHA-bound RGB/mask pair, failing closed on pixel evidence."""

    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a nonempty string")
    if not isinstance(locator, Mapping) or set(locator) != _LOCATOR_FIELDS:
        raise ValueError("image locator fields are malformed")
    for field in ("rgb_path", "mask_path"):
        if not isinstance(locator[field], str) or not locator[field]:
            raise ValueError(f"image locator {field} must be a nonempty string")
    for field in ("rgb_sha256", "mask_sha256"):
        if not isinstance(locator[field], str) or _SHA256.fullmatch(locator[field]) is None:
            raise ValueError(f"image locator {field} must be a lowercase SHA-256")

    rgb_bound = _read_bound_bytes(locator["rgb_path"], locator["rgb_sha256"], "rgb")
    if isinstance(rgb_bound, str):
        return _invalid_measurement(image_id, rgb_bound)
    mask_bound = _read_bound_bytes(
        locator["mask_path"], locator["mask_sha256"], "mask"
    )
    if isinstance(mask_bound, str):
        return _invalid_measurement(image_id, mask_bound)

    from PIL import Image, UnidentifiedImageError
    import numpy as np
    from skimage.color import rgb2lab

    try:
        with Image.open(BytesIO(rgb_bound[0])) as image:
            if image.format != "PNG" or image.mode != "RGB":
                return _invalid_measurement(image_id, "rgb_not_uint8_rgb_png")
            image.load()
            rgb = np.asarray(image, dtype=np.uint8).copy()
        with Image.open(BytesIO(mask_bound[0])) as image:
            if image.format != "PNG" or image.mode not in {"L", "1"}:
                return _invalid_measurement(image_id, "mask_not_binary_png")
            image.load()
            mask = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError, ValueError):
        return _invalid_measurement(image_id, "png_decode_failed")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return _invalid_measurement(image_id, "rgb_shape_invalid")
    if mask.shape != rgb.shape[:2]:
        return _invalid_measurement(image_id, "rgb_mask_shape_mismatch")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        return _invalid_measurement(image_id, "mask_not_binary")
    selected = mask == 255
    liquid_pixels = int(np.count_nonzero(selected))
    if liquid_pixels <= 0:
        return _invalid_measurement(image_id, "mask_empty")
    coverage = liquid_pixels / float(mask.size)
    occupied_rows = np.flatnonzero(np.any(selected, axis=1))
    mask_height_px = int(occupied_rows[-1] - occupied_rows[0] + 1)
    if not math.isfinite(coverage) or coverage <= 0.0 or coverage > 1.0:
        return _invalid_measurement(image_id, "mask_coverage_invalid")
    if mask_height_px <= 0:
        return _invalid_measurement(image_id, "mask_height_quality_invalid")
    lab_image = rgb2lab(rgb.astype(np.float64) / 255.0)
    descriptor = [float(np.median(lab_image[..., channel][selected])) for channel in range(3)]
    if not all(math.isfinite(value) for value in descriptor):
        return _invalid_measurement(image_id, "lab_nonfinite")
    measurement = {
        "image_id": image_id,
        "validity": "valid",
        "limit_codes": [],
        "rgb_sha256": rgb_bound[1],
        "mask_sha256": mask_bound[1],
        "liquid_pixel_count": liquid_pixels,
        "mask_coverage_fraction": coverage,
        "mask_height_px": mask_height_px,
        "lab_raw": descriptor,
        "lab_6dp": [f"{value:.6f}" for value in descriptor],
        "unit": "cie_lab_d65_2deg",
    }
    if _include_dispersion:
        selected_lab = lab_image[selected]
        median_array = np.asarray(descriptor, dtype=np.float64)
        mad_array = np.median(np.abs(selected_lab - median_array), axis=0)
        q25, q75 = np.percentile(selected_lab, [25.0, 75.0], axis=0)
        measurement["median_lab"] = [float(value) for value in median_array]
        measurement["mad_lab"] = [float(value) for value in mad_array]
        measurement["iqr_lab"] = [float(value) for value in q75 - q25]
    return measurement


def measure_trusted_predicted_image(
    entry: object,
    *,
    measurement_id: object,
    evidence_id: object,
) -> tuple[dict[str, object], TrustedPredictedMeasurementBindingV1]:
    """Measure one opaque predicted-mask entry and retain its evidence binding."""

    from .rgb_mask_prediction import is_trusted_predicted_mask_entry_v1

    if not is_trusted_predicted_mask_entry_v1(entry):
        raise TypeError("measurement requires a trusted predicted-mask entry")
    if not isinstance(measurement_id, str) or not measurement_id:
        raise ValueError("trusted predicted measurement_id must be nonempty text")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("trusted predicted evidence_id must be nonempty text")
    measurement = measure_image(
        entry.image_id,
        entry.locator(),
        _include_dispersion=True,
    )
    codes = measurement.get("limit_codes")
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        raise ValueError("trusted predicted measurement limit codes are malformed")
    median_lab: tuple[float, float, float] | None = None
    mad_lab: tuple[float, float, float] | None = None
    iqr_lab: tuple[float, float, float] | None = None
    if measurement.get("validity") == "valid":
        median_lab = tuple(measurement["median_lab"])
        mad_lab = tuple(measurement["mad_lab"])
        iqr_lab = tuple(measurement["iqr_lab"])
    measurement_payload = {
        "image_id": entry.image_id,
        "measurement_id": measurement_id,
        "evidence_id": evidence_id,
        "rgb_sha256": entry.rgb_sha256,
        "mask_sha256": entry.mask_sha256,
        "validity": measurement.get("validity"),
        "limit_codes": list(codes),
        "median_lab": list(median_lab) if median_lab is not None else None,
        "mad_lab": list(mad_lab) if mad_lab is not None else None,
        "iqr_lab": list(iqr_lab) if iqr_lab is not None else None,
    }
    measurement_sha256 = hashlib.sha256(
        json.dumps(
            measurement_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    binding = TrustedPredictedMeasurementBindingV1(
        image_id=entry.image_id,
        measurement_id=measurement_id,
        evidence_id=evidence_id,
        rgb_sha256=entry.rgb_sha256,
        mask_sha256=entry.mask_sha256,
        entry_sha256=entry.entry_sha256,
        registry_sha256=entry.registry_sha256,
        validity=str(measurement.get("validity")),
        limit_codes=tuple(codes),
        median_lab=median_lab,
        mad_lab=mad_lab,
        iqr_lab=iqr_lab,
        measurement_sha256=measurement_sha256,
        _factory_token=_TRUSTED_MEASUREMENT_TOKEN,
    )
    return measurement, binding


def build_trusted_runtime_measured_evidence_v1(
    registry: object,
    measurement_requests: object,
) -> TrustedMeasuredEvidenceResult:
    """Mint the established measurement capability from a reviewed camera registry."""

    from .wp7_rgb_runtime import is_trusted_real_rgb_mask_registry_v1

    if not is_trusted_real_rgb_mask_registry_v1(registry):
        raise TypeError("runtime measurement requires a trusted camera registry")
    if isinstance(measurement_requests, (str, bytes)) or not isinstance(
        measurement_requests, Mapping
    ):
        raise TypeError("runtime measurement requests must be an object")
    bindings: list[TrustedPredictedMeasurementBindingV1] = []
    rows: list[dict[str, object]] = []
    for image_id in sorted(measurement_requests):
        request = measurement_requests[image_id]
        if (
            not isinstance(image_id, str)
            or not isinstance(request, Mapping)
            or set(request) != {"measurement_id", "evidence_id"}
        ):
            raise ValueError("runtime measurement request fields are invalid")
        measurement_id = request.get("measurement_id")
        evidence_id = request.get("evidence_id")
        if not isinstance(measurement_id, str) or not measurement_id:
            raise ValueError("runtime measurement_id must be nonempty text")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("runtime evidence_id must be nonempty text")
        entry = registry.entry(image_id)
        measurement = measure_image(
            image_id,
            entry.measurement_locator(),
            _include_dispersion=True,
        )
        codes = tuple(str(code) for code in measurement.get("limit_codes", []))
        valid = measurement.get("validity") == "valid"
        median = tuple(measurement["median_lab"]) if valid else None
        mad = tuple(measurement["mad_lab"]) if valid else None
        iqr = tuple(measurement["iqr_lab"]) if valid else None
        payload = {
            "image_id": image_id,
            "measurement_id": measurement_id,
            "evidence_id": evidence_id,
            "rgb_sha256": entry.rgb_sha256,
            "mask_sha256": entry.mask_sha256,
            "validity": measurement.get("validity"),
            "limit_codes": list(codes),
            "median_lab": list(median) if median is not None else None,
            "mad_lab": list(mad) if mad is not None else None,
            "iqr_lab": list(iqr) if iqr is not None else None,
        }
        binding = TrustedPredictedMeasurementBindingV1(
            image_id=image_id,
            measurement_id=measurement_id,
            evidence_id=evidence_id,
            rgb_sha256=entry.rgb_sha256,
            mask_sha256=entry.mask_sha256,
            entry_sha256=entry.entry_sha256,
            registry_sha256=registry.registry_sha256,
            validity=str(measurement.get("validity")),
            limit_codes=codes,
            median_lab=median,
            mad_lab=mad,
            iqr_lab=iqr,
            measurement_sha256=hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            _factory_token=_TRUSTED_MEASUREMENT_TOKEN,
        )
        bindings.append(binding)
        rows.append({**payload, "citation_sha256": binding.measurement_sha256})
    value = {
        "schema_version": 1,
        "artifact_type": "runtime_measured_evidence",
        "image_measurements": rows,
        "trusted_predicted_mask_registry_sha256": registry.registry_sha256,
    }
    return TrustedMeasuredEvidenceResult(
        value,
        tuple(bindings),
        registry,
        _factory_token=_TRUSTED_MEASUREMENT_TOKEN,
    )


def _reject_oracles(value: object) -> None:
    try:
        validate_deploy_input_fields(value, require_allowlist=False)
    except ValueError as exc:
        raise ValueError(f"model input contains forbidden oracle key: {exc}") from exc
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_oracles(child)
    elif isinstance(value, list):
        for child in value:
            _reject_oracles(child)
    elif isinstance(value, str) and value in _ORACLE_VALUES:
        raise ValueError(f"model input contains forbidden oracle value: {value}")


def _euclidean(left: object, right: object) -> float:
    if not isinstance(left, list) or not isinstance(right, list):
        raise ValueError("Lab descriptor is malformed")
    if len(left) != 3 or len(right) != 3:
        raise ValueError("Lab descriptor must contain three channels")
    value = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))
    if not math.isfinite(value):
        raise ValueError("delta E76 is nonfinite")
    return value


def _referenced_image_ids(decisions: list[object]) -> set[str]:
    referenced: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping) or not isinstance(decision.get("stages"), list):
            raise ValueError("model input decision records are malformed")
        for stage in decision["stages"]:
            if not isinstance(stage, Mapping):
                raise ValueError("model input stage record is malformed")
            target_id = stage.get("target_image_id")
            cases = stage.get("reference_cases")
            if not isinstance(target_id, str) or not target_id or not isinstance(cases, list):
                raise ValueError("model input stage references are malformed")
            referenced.add(target_id)
            for case in cases:
                if not isinstance(case, Mapping) or not isinstance(
                    case.get("candidate_peers"), list
                ):
                    raise ValueError("model input reference case is malformed")
                for peer in case["candidate_peers"]:
                    if not isinstance(peer, Mapping):
                        raise ValueError("model input candidate peer is malformed")
                    peer_id = peer.get("peer_image_id")
                    if not isinstance(peer_id, str) or not peer_id:
                        raise ValueError("model input candidate image ID is malformed")
                    referenced.add(peer_id)
    return referenced


def _invalid_case(case: Mapping[str, object], codes: list[str]) -> dict[str, object]:
    return {
        "case_id": case.get("case_id"),
        "mode": case.get("mode"),
        "alternative_id": case.get("alternative_id"),
        "disposition_source": case.get("disposition_source") is True,
        "validity": "insufficient_evidence",
        "limit_codes": codes,
        "peer_ids": None,
        "peer_origins": None,
        "peer_count": None,
        "peer_batch_ids": None,
        "peer_batch_count": None,
        "representative_peer_id": None,
        "center_lab_raw": None,
        "center_lab_6dp": None,
        "median_radius_delta_e76_raw": None,
        "max_radius_delta_e76_raw": None,
        "mad_radius_delta_e76_raw": None,
        "pairwise_diameter_delta_e76_raw": None,
        "threshold_delta_e76_raw": None,
        "threshold_delta_e76_6dp": None,
        "target_delta_e76_raw": None,
        "target_delta_e76_6dp": None,
        "stage_outlier": None,
        "unit": "delta_e76",
    }


def _evaluate_case(
    case: Mapping[str, object],
    target: Mapping[str, object],
    measurements: Mapping[str, Mapping[str, object]],
    *,
    target_batch_id: object,
    target_run_id: object,
    decision_config: Mapping[str, object],
) -> dict[str, object]:
    if target["validity"] != "valid":
        return _invalid_case(case, ["target_measurement_invalid"])
    peers = case.get("candidate_peers")
    if not isinstance(peers, list):
        raise ValueError("candidate peers must be a list")
    candidates: dict[str, dict[str, object]] = {}
    invalid_codes: list[str] = []
    seen_peer_rgb_hashes: set[str] = set()
    for peer in peers:
        if not isinstance(peer, Mapping):
            raise ValueError("candidate peer must be an object")
        peer_id = peer.get("peer_image_id")
        batch_id = peer.get("batch_id")
        origin = peer.get("origin")
        run_id = peer.get("run_id")
        if (
            not isinstance(peer_id, str)
            or not peer_id
            or not isinstance(batch_id, str)
            or not batch_id
            or not isinstance(origin, str)
            or not origin
        ):
            raise ValueError("candidate peer identity is malformed")
        if peer_id in candidates:
            raise ValueError("candidate peer image IDs must be unique within a case")
        measured = measurements[peer_id]
        if measured["validity"] != "valid":
            invalid_codes.extend(
                f"candidate_measurement_invalid:{peer_id}:{code}"
                for code in measured["limit_codes"]
            )
            continue
        rgb_sha256 = measured["rgb_sha256"]
        if not isinstance(rgb_sha256, str):
            raise ValueError("valid measurement RGB SHA-256 is malformed")
        if rgb_sha256 == target["rgb_sha256"]:
            invalid_codes.append("target_rgb_bytes_reused_as_reference")
        if rgb_sha256 in seen_peer_rgb_hashes:
            invalid_codes.append("duplicate_reference_rgb_bytes")
        seen_peer_rgb_hashes.add(rgb_sha256)
        candidates[peer_id] = {
            "lab": measured["lab_raw"],
            "mask_sha256": measured["mask_sha256"],
            "batch_id": batch_id,
            "origin": origin,
            "run_id": run_id if isinstance(run_id, str) else batch_id,
        }
    if invalid_codes:
        return _invalid_case(case, list(dict.fromkeys(invalid_codes)))

    from .vision_dataset_plan import select_peer_reference

    reference = select_peer_reference(
        candidates,
        target_mask_sha256=target["mask_sha256"],
        target_batch_id=target_batch_id,
        target_run_id=target_run_id,
        decision_config=decision_config,
    )
    if reference["validity"] != "valid":
        return _invalid_case(case, [str(reference["limit_code"])])
    delta = _euclidean(target["lab_raw"], reference["center_lab"])
    threshold = float(reference["threshold_delta_e76"])
    outlier = delta > threshold
    return {
        "case_id": case.get("case_id"),
        "mode": case.get("mode"),
        "alternative_id": case.get("alternative_id"),
        "disposition_source": case.get("disposition_source") is True,
        "validity": "valid",
        "limit_codes": [],
        "peer_ids": reference["peer_ids"],
        "peer_origins": reference["peer_origins"],
        "peer_count": reference["peer_count"],
        "peer_batch_ids": reference["peer_batch_ids"],
        "peer_batch_count": reference["peer_batch_count"],
        "representative_peer_id": reference["representative_peer_id"],
        "center_lab_raw": reference["center_lab"],
        "center_lab_6dp": reference["serialized_6dp"]["center_lab"],
        "median_radius_delta_e76_raw": reference["median_radius_delta_e76"],
        "max_radius_delta_e76_raw": reference["max_radius_delta_e76"],
        "mad_radius_delta_e76_raw": reference["mad_radius_delta_e76"],
        "pairwise_diameter_delta_e76_raw": reference[
            "pairwise_diameter_delta_e76"
        ],
        "threshold_delta_e76_raw": threshold,
        "threshold_delta_e76_6dp": f"{threshold:.6f}",
        "target_delta_e76_raw": delta,
        "target_delta_e76_6dp": f"{delta:.6f}",
        "stage_outlier": outlier,
        "unit": "delta_e76",
    }


def _evaluate_stage(
    stage: Mapping[str, object],
    measurements: Mapping[str, Mapping[str, object]],
    *,
    target_batch_id: object,
    target_run_id: object,
    decision_config: Mapping[str, object],
) -> dict[str, object]:
    target_id = stage["target_image_id"]
    if not isinstance(target_id, str):
        raise ValueError("target image ID must be a string")
    target = measurements[target_id]
    cases_raw = stage["reference_cases"]
    if not isinstance(cases_raw, list):
        raise ValueError("reference cases must be a list")
    cases = [
        _evaluate_case(
            case,
            target,
            measurements,
            target_batch_id=target_batch_id,
            target_run_id=target_run_id,
            decision_config=decision_config,
        )
        for case in cases_raw
        if isinstance(case, Mapping)
    ]
    if len(cases) != len(cases_raw):
        raise ValueError("reference cases must contain objects")
    primary_mode = stage.get("primary_mode")
    primaries = [
        case
        for case in cases
        if case["mode"] == primary_mode and case["disposition_source"] is True
    ]
    if len(primaries) != 1:
        raise ValueError("stage must contain exactly one primary reference case")
    primary = primaries[0]
    limit_codes: list[str] = []
    if target["validity"] != "valid":
        limit_codes.append("target_measurement_invalid")
        limit_codes.extend(str(code) for code in target["limit_codes"])
    if primary["validity"] != "valid":
        limit_codes.extend(str(code) for code in primary["limit_codes"])
    valid = target["validity"] == "valid" and primary["validity"] == "valid"
    result = {
        "evidence_id": stage.get("evidence_id"),
        "measurement_id": stage.get("measurement_id"),
        "stage_id": stage.get("stage_id"),
        "stage_ordinal": stage.get("stage_ordinal"),
        "target_image_id": target_id,
        "target_measurement": dict(target),
        "reference_cases": cases,
        "primary_mode": primary_mode,
        "validity": "valid" if valid else "insufficient_evidence",
        "limit_codes": [] if valid else list(dict.fromkeys(limit_codes)),
        "target_delta_e76_raw": primary["target_delta_e76_raw"] if valid else None,
        "target_delta_e76_6dp": primary["target_delta_e76_6dp"] if valid else None,
        "threshold_delta_e76_raw": primary["threshold_delta_e76_raw"] if valid else None,
        "threshold_delta_e76_6dp": primary["threshold_delta_e76_6dp"] if valid else None,
        "stage_outlier": primary["stage_outlier"] if valid else None,
        "unit": "delta_e76",
    }
    if "stage_event" in stage:
        result["stage_event"] = dict(stage["stage_event"]) if isinstance(stage["stage_event"], Mapping) else stage["stage_event"]
    else:
        result["time_code"] = stage.get("time_code")
    return result


def _validated_decision_config(model_plan: Mapping[str, object]) -> Mapping[str, object]:
    default = {
        "tight_cluster_max_diameter_delta_e76": 2.0,
        "threshold_floor_delta_e76": 2.0,
        "threshold_margin_delta_e76": 1.0,
    }
    config = model_plan.get("decision_config", default)
    if not isinstance(config, Mapping) or set(config) != set(default):
        raise ValueError("decision config is malformed")
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    declared = model_plan.get("decision_config_sha256")
    if model_plan.get("schema_version") == 2 and declared != digest:
        raise ValueError("decision config hash mismatch")
    if declared is not None and declared != digest:
        raise ValueError("decision config hash mismatch")
    if dict(config) != default:
        raise ValueError("decision config differs from frozen behavior")
    return config


def evaluate_model_plan(model_plan: object, image_locators: object) -> dict[str, object]:
    """Convert allowlisted model inputs and SHA-bound pixels into evidence records."""

    if not isinstance(model_plan, Mapping):
        raise TypeError("model input plan must be an object")
    _reject_oracles(model_plan)
    if model_plan.get("schema_version") == 2:
        validate_deploy_input_fields(model_plan)
    if model_plan.get("artifact_type") != "model_input_plan":
        raise ValueError("only model_input_plan artifacts are accepted")
    if model_plan.get("real_data_used") is not False:
        raise ValueError("this synthetic measurement path requires real_data_used=false")
    decisions_raw = model_plan.get("decision_records")
    if not isinstance(decisions_raw, list) or not decisions_raw:
        raise ValueError("model input plan decision_records must be a nonempty list")
    prediction_producer: dict[str, object] | None = None
    prediction_registry = None
    trusted_prediction_bindings: list[TrustedPredictedMeasurementBindingV1] = []
    if (
        isinstance(image_locators, Mapping)
        and set(image_locators) == {"schema_version", "data_origin", "images"}
        and image_locators.get("schema_version") == 1
        and image_locators.get("data_origin") == "twin"
        and isinstance(image_locators.get("images"), Mapping)
    ):
        locator_images = image_locators["images"]
    elif isinstance(image_locators, Mapping) and image_locators.get("schema_version") == 2:
        from .rgb_mask_prediction import build_trusted_predicted_mask_registry_v1

        prediction_registry = build_trusted_predicted_mask_registry_v1(
            image_locators
        )
        locator_images = {
            image_id: prediction_registry.entry(image_id).locator()
            for image_id in prediction_registry.image_ids
        }
        prediction_producer = prediction_registry.producer
    else:
        raise ValueError("image locator artifact must be a synthetic twin locator map")
    _reject_oracles(image_locators)
    referenced = _referenced_image_ids(decisions_raw)
    assert isinstance(locator_images, Mapping)
    if any(not isinstance(image_id, str) or not image_id for image_id in locator_images):
        raise ValueError("image locator IDs must be nonempty strings")
    if set(locator_images) != referenced:
        missing = sorted(referenced.difference(locator_images))
        extra = sorted(set(locator_images).difference(referenced))
        raise ValueError(f"image locator IDs differ from model plan: missing={missing!r}, extra={extra!r}")
    if model_plan.get("schema_version") == 2:
        paths: dict[str, list[str]] = {"rgb": [], "mask": []}
        for locator in locator_images.values():
            if not isinstance(locator, Mapping):
                raise ValueError("image locator fields are malformed")
            for role in ("rgb", "mask"):
                path = locator.get(f"{role}_path")
                if not isinstance(path, str) or not path:
                    raise ValueError(
                        f"F2 {role} image paths must be nonempty strings"
                    )
                paths[role].append(path)
        for role, values in paths.items():
            if len(set(values)) != len(values):
                raise ValueError(f"F2 {role} image paths must be unique")
    decision_config = _validated_decision_config(model_plan)
    if prediction_registry is None:
        measurements = {
            image_id: measure_image(image_id, locator_images[image_id])
            for image_id in sorted(referenced)
        }
    else:
        measurements = {}
        for image_id in sorted(referenced):
            measurement, trusted_binding = measure_trusted_predicted_image(
                prediction_registry.entry(image_id),
                measurement_id=f"MEAS-{image_id}",
                evidence_id=f"EVID-{image_id}",
            )
            measurement["trusted_predicted_mask_binding"] = trusted_binding.receipt()
            trusted_prediction_bindings.append(trusted_binding)
            measurements[image_id] = measurement

    decisions: list[dict[str, object]] = []
    for decision in decisions_raw:
        assert isinstance(decision, Mapping)
        missing_context = [field for field in _DECISION_CONTEXT_FIELDS if field not in decision]
        if missing_context:
            raise ValueError(f"decision context fields are missing: {missing_context!r}")
        stages_raw = decision["stages"]
        if not isinstance(stages_raw, list) or len(stages_raw) != 3:
            raise ValueError("each vial decision must contain exactly three stages")
        if model_plan.get("schema_version") == 2:
            if not isinstance(decision.get("run_id"), str) or not decision["run_id"]:
                raise ValueError("F2 decision run_id must be source-bound")
            for stage in stages_raw:
                if not isinstance(stage, Mapping):
                    raise ValueError("model input stage records are malformed")
                cases = stage.get("reference_cases")
                if not isinstance(cases, list):
                    raise ValueError("model input reference cases are malformed")
                for case in cases:
                    if not isinstance(case, Mapping) or not isinstance(case.get("candidate_peers"), list):
                        raise ValueError("model input reference cases are malformed")
                    for peer in case["candidate_peers"]:
                        if not isinstance(peer, Mapping) or not isinstance(peer.get("run_id"), str) or not peer["run_id"]:
                            raise ValueError("F2 candidate run_id must be source-bound")
        stages = [
            _evaluate_stage(
                stage,
                measurements,
                target_batch_id=decision.get("batch_id"),
                target_run_id=decision.get("run_id"),
                decision_config=decision_config,
            )
            for stage in stages_raw
            if isinstance(stage, Mapping)
        ]
        if len(stages) != len(stages_raw):
            raise ValueError("model input stage records are malformed")
        if sorted(stage["stage_ordinal"] for stage in stages) != [1, 2, 3]:
            raise ValueError("vial decision stage ordinals must be exactly 1, 2, and 3")
        valid_stages = [stage for stage in stages if stage["validity"] == "valid"]
        if any(stage["stage_outlier"] is True for stage in valid_stages):
            disposition = "discard"
            limit_codes: list[str] = []
        elif len(valid_stages) == 3 and all(
            stage["stage_outlier"] is False for stage in valid_stages
        ):
            disposition = "keep"
            limit_codes = []
        else:
            disposition = "insufficient_evidence"
            limit_codes = list(
                dict.fromkeys(
                    code
                    for stage in stages
                    if stage["validity"] != "valid"
                    for code in stage["limit_codes"]
                )
            )
        context = {field: decision[field] for field in _DECISION_CONTEXT_FIELDS}
        decisions.append(
            {
                **context,
                "stages": stages,
                "disposition": disposition,
                "explanation_status": {
                    "discard": "DIFFERENCE",
                    "keep": "NO_DIFFERENCE",
                    "insufficient_evidence": "UNKNOWN",
                }[disposition],
                "disposition_source": "deterministic_pipeline_primary_reference_evidence",
                "limit_codes": limit_codes,
            }
        )
    measurement_rows = [measurements[key] for key in sorted(measurements)]
    valid_image_measurements = sum(
        row["validity"] == "valid" for row in measurement_rows
    )
    invalid_reasons: dict[str, int] = {}
    for row in measurement_rows:
        if row["validity"] == "valid":
            continue
        for code in row["limit_codes"]:
            reason = str(code)
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    invalid_reasons = dict(sorted(invalid_reasons.items()))
    verified_rgb_hashes = [
        str(row["rgb_sha256"])
        for row in measurement_rows
        if isinstance(row.get("rgb_sha256"), str)
    ]
    verified_mask_hashes = [
        str(row["mask_sha256"])
        for row in measurement_rows
        if isinstance(row.get("mask_sha256"), str)
    ]
    result = {
        "schema_version": 1,
        "artifact_type": "measured_evidence",
        "config_id": model_plan.get("config_id"),
        "decision_config": dict(decision_config),
        "decision_config_sha256": hashlib.sha256(
            json.dumps(decision_config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "measurement_contract": {
            "implementation": "skimage.color.rgb2lab",
            "input_encoding": "uint8_srgb_divide_255",
            "statistic": "coordinatewise_median_lab_under_binary_liquid_mask",
            "distance": "delta_e76",
            "primary_mode": "all_reference_batches",
        },
        "image_measurements": measurement_rows,
        "artifact_content_diagnostics": {
            "rgb_file_sha256_total_count": len(verified_rgb_hashes),
            "rgb_file_sha256_unique_count": len(set(verified_rgb_hashes)),
            "mask_file_sha256_total_count": len(verified_mask_hashes),
            "mask_file_sha256_unique_count": len(set(verified_mask_hashes)),
        },
        "decision_records": decisions,
        "denominators": {
            "image_measurements": len(measurements),
            "valid_image_measurements": valid_image_measurements,
            "invalid_image_measurements": len(measurements)
            - valid_image_measurements,
            "invalid_image_measurement_reasons": invalid_reasons,
            "vial_decisions": len(decisions),
            "valid_vial_decisions": sum(
                decision["disposition"] != "insufficient_evidence"
                for decision in decisions
            ),
            "abstained_vial_decisions": sum(
                decision["disposition"] == "insufficient_evidence"
                for decision in decisions
            ),
        },
        "data_origin": "twin",
        "real_data_used": False,
        "claim_boundary": "Synthetic colour-distance evidence only; no physical carryover, defect, cause, volume, plant-action, or sim-to-real claim.",
    }
    if prediction_producer is not None:
        result["mask_source"] = "rgb_only_prediction"
        result["mask_prediction_producer"] = prediction_producer
        assert prediction_registry is not None
        result["trusted_predicted_mask_registry_sha256"] = (
            prediction_registry.registry_sha256
        )
        return TrustedMeasuredEvidenceResult(
            result,
            tuple(trusted_prediction_bindings),
            prediction_registry,
            _factory_token=_TRUSTED_MEASUREMENT_TOKEN,
        )
    return result


def _load_strict_json(path: Path, role: str) -> tuple[object, str]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{role} is unavailable") from error
    if len(data) > 64 * 1024 * 1024:
        raise ValueError(f"{role} exceeds the size bound")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{role} contains nonfinite JSON value {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not strict UTF-8 JSON") from error
    return payload, hashlib.sha256(data).hexdigest()


def _publish_json_no_clobber(output: Path, payload: object) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"measured evidence output already exists: {output}")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("measured evidence output parent must be a directory")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = parent / f".{output.name}.tmp-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, output, follow_symlinks=False)
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not published:
        raise OSError("measured evidence output was not published")


def write_measured_evidence(
    model_plan_path: str | os.PathLike[str],
    locator_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Evaluate two separate allowlisted artifacts and publish one new record."""

    plan, plan_sha256 = _load_strict_json(Path(model_plan_path), "model input plan")
    locators, locator_sha256 = _load_strict_json(
        Path(locator_path), "image locator artifact"
    )
    result = evaluate_model_plan(plan, locators)
    result["source_model_input_sha256"] = plan_sha256
    result["source_image_locators_sha256"] = locator_sha256
    _publish_json_no_clobber(Path(output_path), result)
    return result
