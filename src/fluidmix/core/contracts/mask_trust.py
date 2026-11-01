"""Ring-neutral verification contract for live mask capabilities."""

from __future__ import annotations

import hashlib
import json
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PREDICTED_MASK_TRUST_TOKEN = object()
_REAL_RGB_MASK_TRUST_TOKEN = object()


class _TrustedPredictedMaskEntryV1Base:
    """Marker for factory-minted predicted-mask entry capabilities."""


class _TrustedPredictedMaskRegistryV1Base:
    """Marker for factory-minted predicted-mask registry capabilities."""


class _TrustedRealRGBMaskEntryV1Base:
    """Marker for factory-minted real-camera mask capabilities."""


class _TrustedRealRGBMaskRegistryV1Base:
    """Marker for factory-minted real-camera mask registries."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _predicted_entry_payload(value: object) -> dict[str, str]:
    locator = value.locator()
    return {
        "image_id": value.image_id,
        "rgb_path": locator["rgb_path"],
        "rgb_sha256": locator["rgb_sha256"],
        "mask_path": locator["mask_path"],
        "mask_sha256": locator["mask_sha256"],
        "model_sha256": value.model_sha256,
        "config_sha256": value.config_sha256,
        "code_commit": value.code_commit,
        "threshold_source": value.threshold_source,
    }


def is_trusted_predicted_mask_entry_v1(value: object) -> bool:
    """Return whether a value is an intact factory-minted prediction entry."""

    try:
        return bool(
            isinstance(value, _TrustedPredictedMaskEntryV1Base)
            and value._factory_token is _PREDICTED_MASK_TRUST_TOKEN
            and _SHA256.fullmatch(value.registry_sha256) is not None
            and _canonical_sha256(_predicted_entry_payload(value))
            == value.entry_sha256
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def is_trusted_predicted_mask_registry_v1(value: object) -> bool:
    """Return whether a value is an intact factory-minted prediction registry."""

    try:
        if (
            not isinstance(value, _TrustedPredictedMaskRegistryV1Base)
            or value._factory_token is not _PREDICTED_MASK_TRUST_TOKEN
            or _SHA256.fullmatch(value.registry_sha256) is None
        ):
            return False
        entry_payloads = []
        for image_id in sorted(value.image_ids):
            entry = value.entry(image_id)
            if (
                not isinstance(image_id, str)
                or not is_trusted_predicted_mask_entry_v1(entry)
                or entry.image_id != image_id
                or entry.registry_sha256 != value.registry_sha256
            ):
                return False
            entry_payloads.append(_predicted_entry_payload(entry))
        if not entry_payloads:
            return False
        return _canonical_sha256(
            {
                "schema_version": "trusted-predicted-mask-registry-v1",
                "entries": entry_payloads,
            }
        ) == value.registry_sha256
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def is_trusted_real_rgb_mask_entry_v1(value: object) -> bool:
    """Return whether a value is a live real-camera mask entry."""

    try:
        return bool(
            isinstance(value, _TrustedRealRGBMaskEntryV1Base)
            and value._factory_token is _REAL_RGB_MASK_TRUST_TOKEN
            and _SHA256.fullmatch(value.registry_sha256) is not None
            and _SHA256.fullmatch(value.entry_sha256) is not None
        )
    except (AttributeError, TypeError, ValueError):
        return False


def is_trusted_real_rgb_mask_registry_v1(value: object) -> bool:
    """Return whether a value is a live real-camera mask registry."""

    try:
        return bool(
            isinstance(value, _TrustedRealRGBMaskRegistryV1Base)
            and value._factory_token is _REAL_RGB_MASK_TRUST_TOKEN
            and _SHA256.fullmatch(value.registry_sha256) is not None
            and all(
                is_trusted_real_rgb_mask_entry_v1(value.entry(image_id))
                and value.entry(image_id).registry_sha256 == value.registry_sha256
                for image_id in value.image_ids
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False


__all__ = [
    "is_trusted_predicted_mask_entry_v1",
    "is_trusted_predicted_mask_registry_v1",
    "is_trusted_real_rgb_mask_entry_v1",
    "is_trusted_real_rgb_mask_registry_v1",
]
