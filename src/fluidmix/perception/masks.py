"""Strict manifest boundary for RGB-only predicted liquid masks."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType


_TOP_FIELDS = {
    "schema_version",
    "artifact_type",
    "data_origin",
    "mask_source",
    "producer",
    "images",
}
_PRODUCER_FIELDS = {
    "producer_id",
    "producer_kind",
    "model_id",
    "model_sha256",
    "config_id",
    "config_sha256",
    "code_commit",
    "threshold_source",
    "training_data_origin",
    "input_fields",
}
_IMAGE_FIELDS = {"rgb_path", "rgb_sha256", "mask_path", "mask_sha256"}
_TEMPORAL_IMAGE_FIELDS = {
    "rgb_path",
    "rgb_sha256",
    "reference_rgb_path",
    "reference_rgb_sha256",
    "mask_path",
    "mask_sha256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_FORBIDDEN_TOKENS = {"oracle", "label", "labels", "residual", "palette", "composition"}
_TRUST_FACTORY_TOKEN = object()
_REAL_TRUST_TOKEN = object()
ENDPOINT_SELECTION_UNAVAILABLE_CODES = frozenset(
    {
        "dispense_stream_bout_missing",
        "dispense_stream_bout_count_ambiguous",
        "stable_post_bout_endpoint_missing",
        "stable_post_bout_endpoint_ambiguous",
    }
)


class _TrustedRealRGBMaskEntryV1Base:
    """Marker for factory-minted real-camera mask capabilities."""


class _TrustedRealRGBMaskRegistryV1Base:
    """Marker for factory-minted real-camera mask registries."""


def is_trusted_real_rgb_mask_entry_v1(value: object) -> bool:
    return bool(
        isinstance(value, _TrustedRealRGBMaskEntryV1Base)
        and value._factory_token is _REAL_TRUST_TOKEN
        and _SHA256.fullmatch(value.registry_sha256) is not None
        and _SHA256.fullmatch(value.entry_sha256) is not None
    )


def is_trusted_real_rgb_mask_registry_v1(value: object) -> bool:
    return bool(
        isinstance(value, _TrustedRealRGBMaskRegistryV1Base)
        and value._factory_token is _REAL_TRUST_TOKEN
        and _SHA256.fullmatch(value.registry_sha256) is not None
        and all(
            is_trusted_real_rgb_mask_entry_v1(value.entry(image_id))
            and value.entry(image_id).registry_sha256 == value.registry_sha256
            for image_id in value.image_ids
        )
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TrustedPredictedMaskEntryV1:
    """Opaque locator capability created only after schema-v2 validation."""

    __slots__ = (
        "_image_id",
        "_rgb_path",
        "_rgb_sha256",
        "_mask_path",
        "_mask_sha256",
        "_model_sha256",
        "_config_sha256",
        "_code_commit",
        "_threshold_source",
        "_entry_sha256",
        "_registry_sha256",
        "_factory_token",
        "_sealed",
    )

    def __init__(
        self,
        *,
        image_id: str,
        locator: Mapping[str, object],
        producer: Mapping[str, object],
        entry_sha256: str,
        registry_sha256: str,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _TRUST_FACTORY_TOKEN:
            raise TypeError(
                "trusted predicted-mask entries are factory-only; validate a locator artifact"
            )
        self._image_id = image_id
        self._rgb_path = str(locator["rgb_path"])
        self._rgb_sha256 = str(locator["rgb_sha256"])
        self._mask_path = str(locator["mask_path"])
        self._mask_sha256 = str(locator["mask_sha256"])
        self._model_sha256 = str(producer["model_sha256"])
        self._config_sha256 = str(producer["config_sha256"])
        self._code_commit = str(producer["code_commit"])
        self._threshold_source = str(producer["threshold_source"])
        self._entry_sha256 = entry_sha256
        self._registry_sha256 = registry_sha256
        self._factory_token = _TRUST_FACTORY_TOKEN
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("trusted predicted-mask entries are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("trusted predicted-mask entries are immutable")
        object.__delattr__(self, name)

    @property
    def image_id(self) -> str:
        return self._image_id

    @property
    def rgb_path(self) -> str:
        return self._rgb_path

    @property
    def rgb_sha256(self) -> str:
        return self._rgb_sha256

    @property
    def mask_path(self) -> str:
        return self._mask_path

    @property
    def mask_sha256(self) -> str:
        return self._mask_sha256

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def config_sha256(self) -> str:
        return self._config_sha256

    @property
    def code_commit(self) -> str:
        return self._code_commit

    @property
    def threshold_source(self) -> str:
        return self._threshold_source

    @property
    def entry_sha256(self) -> str:
        return self._entry_sha256

    @property
    def registry_sha256(self) -> str:
        return self._registry_sha256

    def locator(self) -> dict[str, str]:
        return {
            "rgb_path": self._rgb_path,
            "rgb_sha256": self._rgb_sha256,
            "mask_path": self._mask_path,
            "mask_sha256": self._mask_sha256,
        }


class TrustedPredictedMaskRegistryV1:
    """Read-only collection of opaque validated predicted-mask entries."""

    __slots__ = (
        "_entries",
        "_producer",
        "_registry_sha256",
        "_factory_token",
        "_sealed",
    )

    def __init__(
        self,
        entries: Mapping[str, TrustedPredictedMaskEntryV1],
        producer: Mapping[str, object],
        registry_sha256: str,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _TRUST_FACTORY_TOKEN:
            raise TypeError(
                "trusted predicted-mask registries are factory-only; validate a locator artifact"
            )
        frozen_producer = copy.deepcopy(dict(producer))
        frozen_producer["input_fields"] = tuple(frozen_producer["input_fields"])
        self._entries = MappingProxyType(dict(entries))
        self._producer = MappingProxyType(frozen_producer)
        self._registry_sha256 = registry_sha256
        self._factory_token = _TRUST_FACTORY_TOKEN
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("trusted predicted-mask registries are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("trusted predicted-mask registries are immutable")
        object.__delattr__(self, name)

    @property
    def registry_sha256(self) -> str:
        return self._registry_sha256

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def producer(self) -> dict[str, object]:
        producer = copy.deepcopy(dict(self._producer))
        producer["input_fields"] = list(producer["input_fields"])
        return producer

    def entry(self, image_id: str) -> TrustedPredictedMaskEntryV1:
        try:
            return self._entries[image_id]
        except KeyError as error:
            raise ValueError(
                f"trusted predicted-mask registry is missing image_id: {image_id}"
            ) from error


def _entry_payload_from_capability(
    entry: TrustedPredictedMaskEntryV1,
) -> dict[str, str]:
    return {
        "image_id": entry._image_id,
        "rgb_path": entry._rgb_path,
        "rgb_sha256": entry._rgb_sha256,
        "mask_path": entry._mask_path,
        "mask_sha256": entry._mask_sha256,
        "model_sha256": entry._model_sha256,
        "config_sha256": entry._config_sha256,
        "code_commit": entry._code_commit,
        "threshold_source": entry._threshold_source,
    }


def is_trusted_predicted_mask_entry_v1(value: object) -> bool:
    """Return whether ``value`` is a live factory-created entry capability."""

    try:
        if (
            not isinstance(value, TrustedPredictedMaskEntryV1)
            or value._factory_token is not _TRUST_FACTORY_TOKEN
        ):
            return False
        if (
            not isinstance(value._registry_sha256, str)
            or _SHA256.fullmatch(value._registry_sha256) is None
        ):
            return False
        return _canonical_sha256(_entry_payload_from_capability(value)) == value._entry_sha256
    except (AttributeError, TypeError, ValueError):
        return False


def is_trusted_predicted_mask_registry_v1(value: object) -> bool:
    """Return whether ``value`` is a live factory-created registry capability."""

    try:
        if (
            not isinstance(value, TrustedPredictedMaskRegistryV1)
            or value._factory_token is not _TRUST_FACTORY_TOKEN
        ):
            return False
        entries = value._entries
        if not isinstance(entries, Mapping) or not entries:
            return False
        entry_payloads = []
        for image_id in sorted(entries):
            entry = entries[image_id]
            if (
                not isinstance(image_id, str)
                or not is_trusted_predicted_mask_entry_v1(entry)
                or entry.image_id != image_id
                or entry.registry_sha256 != value._registry_sha256
            ):
                return False
            entry_payloads.append(_entry_payload_from_capability(entry))
        observed_registry_sha256 = _canonical_sha256(
            {
                "schema_version": "trusted-predicted-mask-registry-v1",
                "entries": entry_payloads,
            }
        )
        return observed_registry_sha256 == value._registry_sha256
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _require_exact_fields(
    value: Mapping[object, object], expected: set[str], role: str
) -> None:
    for key in value:
        if not isinstance(key, str):
            raise ValueError("prediction locator field names must be strings")
    actual = set(value)
    if actual == expected:
        return
    for key in sorted(actual.difference(expected)):
        assert isinstance(key, str)
        normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        if set(normalized.split("_")).intersection(_FORBIDDEN_TOKENS):
            raise ValueError(f"forbidden prediction locator field: {key}")
    raise ValueError(f"{role} fields are invalid")


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"prediction producer {role} must be nonempty text")
    return value


def _sha(value: object, role: str) -> str:
    text = _text(value, role)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"prediction producer {role} must be lowercase SHA-256")
    return text


def validate_rgb_mask_prediction_locators(
    artifact: object,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Validate schema 2 and return its images plus copied producer provenance."""

    if not isinstance(artifact, Mapping):
        raise TypeError("RGB mask prediction locator artifact must be an object")
    _require_exact_fields(artifact, _TOP_FIELDS, "RGB mask prediction locator top-level")
    expected_header = {
        "schema_version": 2,
        "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
    }
    if any(artifact.get(key) != value for key, value in expected_header.items()):
        raise ValueError("RGB mask prediction locator identity drifted")

    producer = artifact.get("producer")
    if not isinstance(producer, Mapping):
        raise ValueError("prediction producer fields are invalid")
    _require_exact_fields(producer, _PRODUCER_FIELDS, "prediction producer")
    for field in ("producer_id", "model_id", "config_id"):
        _text(producer.get(field), field)
    if producer.get("producer_kind") != "rgb_only_liquid_segmenter":
        raise ValueError("prediction producer kind is invalid")
    _sha(producer.get("model_sha256"), "model_sha256")
    _sha(producer.get("config_sha256"), "config_sha256")
    commit = producer.get("code_commit")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("prediction producer code_commit must be 40 lowercase hex")
    if producer.get("threshold_source") != "twin_validation_frozen":
        raise ValueError("prediction producer threshold source is invalid")
    if producer.get("training_data_origin") != "twin":
        raise ValueError("prediction producer training data origin is invalid")
    if producer.get("input_fields") != ["rgb"] or not isinstance(
        producer.get("input_fields"), list
    ):
        raise ValueError("prediction producer input_fields must be exactly ['rgb']")

    images = artifact.get("images")
    if not isinstance(images, Mapping) or not images:
        raise ValueError("RGB mask prediction locator images must be a nonempty object")
    for image_id, locator in images.items():
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("RGB mask prediction image IDs must be nonempty strings")
        if not isinstance(locator, Mapping):
            raise ValueError("RGB mask prediction image locator fields are invalid")
        _require_exact_fields(
            locator, _IMAGE_FIELDS, "RGB mask prediction image locator"
        )
        for role in ("rgb", "mask"):
            path = locator.get(f"{role}_path")
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise ValueError(
                    f"RGB mask prediction {role} path must be nonempty and absolute"
                )
            digest = locator.get(f"{role}_sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(
                    f"RGB mask prediction {role} hash must be lowercase SHA-256"
                )
    return images, copy.deepcopy(dict(producer))


def build_trusted_predicted_mask_registry_v1(
    artifact: object,
) -> TrustedPredictedMaskRegistryV1:
    """Validate one spatial RGB prediction artifact and mint opaque entries."""

    images, producer = validate_rgb_mask_prediction_locators(artifact)
    entry_payloads: dict[str, dict[str, str]] = {}
    for image_id in sorted(images):
        locator = images[image_id]
        assert isinstance(image_id, str)
        assert isinstance(locator, Mapping)
        mask_path = Path(str(locator["mask_path"]))
        try:
            resolved_mask_path = mask_path.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"trusted prediction mask path is not canonical: {image_id}"
            ) from error
        candidate_parts = (
            tuple(part.lower() for part in mask_path.parts),
            tuple(part.lower() for part in resolved_mask_path.parts),
        )
        if any(
            parts[index : index + 2] == ("masks", "oracle")
            for parts in candidate_parts
            for index in range(len(parts) - 1)
        ):
            raise ValueError(
                f"trusted prediction cannot relabel an oracle mask path: {image_id}"
            )
        if resolved_mask_path != mask_path:
            raise ValueError(
                f"trusted prediction mask path must be canonical and symlink-free: {image_id}"
            )
        entry_payloads[image_id] = {
            "image_id": image_id,
            "rgb_path": str(locator["rgb_path"]),
            "rgb_sha256": str(locator["rgb_sha256"]),
            "mask_path": str(locator["mask_path"]),
            "mask_sha256": str(locator["mask_sha256"]),
            "model_sha256": str(producer["model_sha256"]),
            "config_sha256": str(producer["config_sha256"]),
            "code_commit": str(producer["code_commit"]),
            "threshold_source": str(producer["threshold_source"]),
        }
    registry_sha256 = _canonical_sha256(
        {
            "schema_version": "trusted-predicted-mask-registry-v1",
            "entries": [entry_payloads[key] for key in sorted(entry_payloads)],
        }
    )
    entries = {
        image_id: TrustedPredictedMaskEntryV1(
            image_id=image_id,
            locator=images[image_id],
            producer=producer,
            entry_sha256=_canonical_sha256(entry_payloads[image_id]),
            registry_sha256=registry_sha256,
            _factory_token=_TRUST_FACTORY_TOKEN,
        )
        for image_id in sorted(entry_payloads)
    }
    return TrustedPredictedMaskRegistryV1(
        entries,
        producer,
        registry_sha256,
        _factory_token=_TRUST_FACTORY_TOKEN,
    )


def validate_temporal_rgb_mask_prediction_locators(
    artifact: object,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Validate schema 3 temporal RGB predictions and return bound locators."""

    if not isinstance(artifact, Mapping):
        raise TypeError("temporal RGB mask prediction locator artifact must be an object")
    _require_exact_fields(
        artifact, _TOP_FIELDS, "temporal RGB mask prediction locator top-level"
    )
    expected_header = {
        "schema_version": 3,
        "artifact_type": "rgb_temporal_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
    }
    if any(artifact.get(key) != value for key, value in expected_header.items()):
        raise ValueError("temporal RGB mask prediction locator identity drifted")

    producer = artifact.get("producer")
    if not isinstance(producer, Mapping):
        raise ValueError("prediction producer fields are invalid")
    _require_exact_fields(producer, _PRODUCER_FIELDS, "prediction producer")
    for field in ("producer_id", "model_id", "config_id"):
        _text(producer.get(field), field)
    if producer.get("producer_kind") != "rgb_only_liquid_segmenter":
        raise ValueError("prediction producer kind is invalid")
    _sha(producer.get("model_sha256"), "model_sha256")
    _sha(producer.get("config_sha256"), "config_sha256")
    commit = producer.get("code_commit")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ValueError("prediction producer code_commit must be 40 lowercase hex")
    if producer.get("threshold_source") != "twin_validation_frozen":
        raise ValueError("prediction producer threshold source is invalid")
    if producer.get("training_data_origin") != "twin":
        raise ValueError("prediction producer training data origin is invalid")
    if producer.get("input_fields") != ["rgb"] or not isinstance(
        producer.get("input_fields"), list
    ):
        raise ValueError("prediction producer input_fields must be exactly ['rgb']")

    images = artifact.get("images")
    if not isinstance(images, Mapping) or not images:
        raise ValueError(
            "temporal RGB mask prediction locator images must be a nonempty object"
        )
    for image_id, locator in images.items():
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(
                "temporal RGB mask prediction image IDs must be nonempty strings"
            )
        if not isinstance(locator, Mapping):
            raise ValueError(
                "temporal RGB mask prediction image locator fields are invalid"
            )
        _require_exact_fields(
            locator,
            _TEMPORAL_IMAGE_FIELDS,
            "temporal RGB mask prediction image locator",
        )
        for role in ("rgb", "reference_rgb", "mask"):
            path = locator.get(f"{role}_path")
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise ValueError(
                    f"temporal RGB mask prediction {role} path must be nonempty and absolute"
                )
            digest = locator.get(f"{role}_sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(
                    f"temporal RGB mask prediction {role} hash must be lowercase SHA-256"
                )
    return images, copy.deepcopy(dict(producer))
