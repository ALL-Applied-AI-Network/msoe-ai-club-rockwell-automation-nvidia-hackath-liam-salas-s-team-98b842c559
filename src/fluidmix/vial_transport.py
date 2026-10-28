"""Fixed-camera rigid vial/carrier transport for accepted FillCam frames."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class TransportSpec:
    frame_count: int = 21
    fps: float = 10.0
    outgoing_shift_px: int = -400
    incoming_shift_px: int = -180
    inpaint_radius_px: float = 5.0
    feather_radius_px: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count < 2
        ):
            raise ValueError("frame_count must be an integer of at least two")
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(float(self.fps))
            or float(self.fps) <= 0.0
        ):
            raise ValueError("fps must be finite and positive")
        for name, value in (
            ("outgoing_shift_px", self.outgoing_shift_px),
            ("incoming_shift_px", self.incoming_shift_px),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value >= 0:
                raise ValueError(f"{name} must be a negative integer")
            if abs(value) < self.frame_count - 1:
                raise ValueError(
                    f"{name} must move at least one pixel per transport frame"
                )
        if (
            isinstance(self.inpaint_radius_px, bool)
            or not isinstance(self.inpaint_radius_px, (int, float))
            or not math.isfinite(float(self.inpaint_radius_px))
            or float(self.inpaint_radius_px) <= 0.0
        ):
            raise ValueError("inpaint_radius_px must be finite and positive")
        if (
            isinstance(self.feather_radius_px, bool)
            or not isinstance(self.feather_radius_px, int)
            or self.feather_radius_px < 0
        ):
            raise ValueError("feather_radius_px must be a non-negative integer")


def transport_offsets(spec: TransportSpec) -> tuple[tuple[int, int], ...]:
    """Return smooth monotonic `(outgoing_x, incoming_x)` pixel offsets."""

    if not isinstance(spec, TransportSpec):
        raise TypeError("spec must be a TransportSpec")
    result = []
    for index in range(spec.frame_count):
        fraction = index / (spec.frame_count - 1)
        eased = fraction * fraction * (3.0 - 2.0 * fraction)
        result.append(
            (
                round(spec.outgoing_shift_px * eased),
                round(spec.incoming_shift_px * eased),
            )
        )
    if len(set(result)) != len(result):
        raise ValueError("transport offsets must be byte-distinct at this frame count")
    return tuple(result)


def _validated_frame(frame_rgb: object) -> np.ndarray:
    frame = np.asarray(frame_rgb)
    if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise ValueError("frame_rgb must be a uint8 640x480 RGB image")
    return np.ascontiguousarray(frame)


def _validated_mask(mask_value: object, name: str) -> np.ndarray:
    raw = np.asarray(mask_value)
    if raw.shape != (480, 640) or raw.dtype.kind not in "bui":
        raise ValueError(f"{name} must be a 640x480 binary mask")
    mask = np.where(raw > 0, np.uint8(255), np.uint8(0))
    if not np.any(mask):
        raise ValueError(f"{name} must not be empty")
    return np.ascontiguousarray(mask)


def _grabcut_region(
    frame_rgb: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    *,
    definite_foreground: np.ndarray | None = None,
) -> np.ndarray:
    """Refine one fixed FillCam mover region without transforming background."""

    x0, y0, x1, y1 = bbox_xyxy
    mask = np.full(frame_rgb.shape[:2], cv2.GC_BGD, dtype=np.uint8)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD

    region = frame_rgb[y0:y1, x0:x1].astype(np.int16)
    background_samples = np.concatenate(
        (
            frame_rgb[0:40].reshape(-1, 3),
            frame_rgb[:, 0:30].reshape(-1, 3),
        ),
        axis=0,
    )
    background_color = np.median(background_samples, axis=0)
    color_distance = np.linalg.norm(region - background_color, axis=2)
    hard = np.zeros(frame_rgb.shape[:2], dtype=bool)
    hard[y0:y1, x0:x1] = color_distance >= 35.0
    if definite_foreground is not None:
        hard |= definite_foreground > 0
    if not np.any(hard):
        raise ValueError("FillCam mover region has no foreground evidence")
    mask[hard] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        mask,
        None,
        background_model,
        foreground_model,
        3,
        cv2.GC_INIT_WITH_MASK,
    )
    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        np.uint8(255),
        np.uint8(0),
    )
    result[:y0] = 0
    result[y1:] = 0
    result[:, :x0] = 0
    result[:, x1:] = 0
    result[hard] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    return result


def extract_fillcam_mover_masks(
    frame_rgb: object, liquid_mask: object
) -> tuple[np.ndarray, np.ndarray]:
    """Extract outgoing and next-empty mover masks in the frozen FillCam view.

    The bounds correspond to the qualified 640x480 camera and the three authored
    vial/carrier positions.  The accepted liquid mask is a hard foreground seed
    for the outgoing vial; no synthetic class or oracle is added to model input.
    """

    frame = _validated_frame(frame_rgb)
    liquid = _validated_mask(liquid_mask, "liquid_mask")
    outgoing = _grabcut_region(
        frame,
        (245, 50, 400, 480),
        definite_foreground=liquid,
    )
    incoming = _grabcut_region(frame, (405, 60, 585, 480))
    outgoing[liquid > 0] = 255
    overlap = (outgoing > 0) & (incoming > 0)
    if np.any(overlap):
        incoming[overlap] = 0
    if not np.any(incoming):
        raise ValueError("incoming FillCam mover mask is empty")
    return outgoing, incoming


def compose_transport_frames(
    frame_rgb: object,
    *,
    outgoing_mask: object,
    incoming_mask: object,
    spec: TransportSpec = TransportSpec(),
) -> tuple[np.ndarray, ...]:
    """Translate two rigid foregrounds left over a fixed inpainted background."""

    frame = _validated_frame(frame_rgb)
    outgoing = _validated_mask(outgoing_mask, "outgoing_mask")
    incoming = _validated_mask(incoming_mask, "incoming_mask")
    if np.any((outgoing > 0) & (incoming > 0)):
        raise ValueError("outgoing_mask and incoming_mask must not overlap")
    if not isinstance(spec, TransportSpec):
        raise TypeError("spec must be a TransportSpec")

    union = cv2.bitwise_or(outgoing, incoming)
    background = cv2.inpaint(
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        union,
        float(spec.inpaint_radius_px),
        cv2.INPAINT_TELEA,
    )
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

    def alpha(mask: np.ndarray) -> np.ndarray:
        if spec.feather_radius_px == 0:
            return (mask.astype(np.float32) / 255.0)[..., None]
        radius = spec.feather_radius_px
        blurred = cv2.GaussianBlur(
            mask,
            (2 * radius + 1, 2 * radius + 1),
            sigmaX=max(radius / 2.0, 0.5),
        )
        return (blurred.astype(np.float32) / 255.0)[..., None]

    outgoing_alpha = alpha(outgoing)
    incoming_alpha = alpha(incoming)

    def translated(layer: np.ndarray, dx: int) -> np.ndarray:
        matrix = np.asarray(((1.0, 0.0, float(dx)), (0.0, 1.0, 0.0)))
        return cv2.warpAffine(
            layer,
            matrix,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    frames: list[np.ndarray] = []
    for index, (outgoing_dx, incoming_dx) in enumerate(transport_offsets(spec)):
        if index == 0:
            frames.append(frame.copy())
            continue
        canvas = background.astype(np.float32)
        for mask_alpha, dx in (
            (incoming_alpha, incoming_dx),
            (outgoing_alpha, outgoing_dx),
        ):
            moved_alpha = translated(mask_alpha, dx)
            if moved_alpha.ndim == 2:
                moved_alpha = moved_alpha[..., None]
            moved_layer = translated(frame, dx).astype(np.float32)
            canvas = moved_layer * moved_alpha + canvas * (1.0 - moved_alpha)
        frames.append(np.clip(np.rint(canvas), 0, 255).astype(np.uint8))
    return tuple(frames)


__all__ = [
    "TransportSpec",
    "compose_transport_frames",
    "extract_fillcam_mover_masks",
    "transport_offsets",
]
