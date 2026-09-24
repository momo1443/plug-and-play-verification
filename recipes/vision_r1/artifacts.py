"""Immutable visual artifacts for replayable crop certificates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image

CERTIFICATE_SCHEMA_VERSION = "vision-r1-visual-certificate-v1"


def image_sha256(image: Image.Image) -> str:
    """Hash decoded RGB pixels and dimensions, independent of file encoding."""

    rgb = image.convert("RGB")
    header = f"RGB\0{rgb.width}\0{rgb.height}\0".encode("ascii")
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def normalize_bbox(
    raw_bbox: Any,
    *,
    width: int,
    height: int,
    min_edge: int = 28,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.90,
) -> tuple[int, int, int, int]:
    """Validate a non-trivial in-bounds pixel crop without silent clipping."""

    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)) or len(raw_bbox) != 4:
        raise ValueError("bbox_2d must contain exactly four integer coordinates")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_bbox):
        raise ValueError("bbox_2d coordinates must be integers")
    left, top, right, bottom = (int(value) for value in raw_bbox)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("bbox_2d must be ordered and inside the source image")
    crop_width = right - left
    crop_height = bottom - top
    if crop_width < min_edge or crop_height < min_edge:
        raise ValueError(f"crop edges must each be at least {min_edge} pixels")
    area_ratio = (crop_width * crop_height) / float(width * height)
    if not min_area_ratio <= area_ratio <= max_area_ratio:
        raise ValueError(f"crop area ratio must be in [{min_area_ratio:.2f}, {max_area_ratio:.2f}]")
    return left, top, right, bottom


@dataclass(frozen=True)
class VisualArtifact:
    artifact_id: str
    sha256: str
    width: int
    height: int
    image: Image.Image
    parent_artifact_id: str | None = None
    parent_sha256: str | None = None
    bbox_2d: tuple[int, int, int, int] | None = None
    created_step: int = 0

    @classmethod
    def root(cls, image: Image.Image) -> VisualArtifact:
        rgb = image.convert("RGB")
        digest = image_sha256(rgb)
        return cls(
            artifact_id=f"image:root:{digest[:12]}",
            sha256=digest,
            width=rgb.width,
            height=rgb.height,
            image=rgb,
        )

    @classmethod
    def crop(
        cls,
        *,
        ordinal: int,
        parent: VisualArtifact,
        bbox_2d: Any,
        created_step: int,
    ) -> VisualArtifact:
        bbox = normalize_bbox(bbox_2d, width=parent.width, height=parent.height)
        image = parent.image.crop(bbox).convert("RGB")
        digest = image_sha256(image)
        return cls(
            artifact_id=f"image:crop:{ordinal}:{digest[:12]}",
            sha256=digest,
            width=image.width,
            height=image.height,
            image=image,
            parent_artifact_id=parent.artifact_id,
            parent_sha256=parent.sha256,
            bbox_2d=bbox,
            created_step=created_step,
        )

    def record(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "parent_artifact_id": self.parent_artifact_id,
            "parent_sha256": self.parent_sha256,
            "bbox_2d": list(self.bbox_2d) if self.bbox_2d is not None else None,
            "created_step": self.created_step,
        }


@dataclass(frozen=True)
class VisualCertificate:
    schema_version: str
    evidence_artifact_ids: tuple[str, ...]
    claim: str


def parse_visual_certificate(value: Any) -> tuple[VisualCertificate | None, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return None, ("certificate_not_object",)
    required = {"schema_version", "evidence_artifact_ids", "claim"}
    if set(value) != required:
        return None, ("certificate_fields_invalid",)
    errors: list[str] = []
    schema_version = value.get("schema_version")
    evidence_ids = value.get("evidence_artifact_ids")
    claim = value.get("claim")
    if schema_version != CERTIFICATE_SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if not isinstance(evidence_ids, list) or not evidence_ids:
        errors.append("evidence_artifact_ids_empty")
    elif len(evidence_ids) > 2:
        errors.append("too_many_evidence_artifacts")
    elif len(set(evidence_ids)) != len(evidence_ids) or not all(
        isinstance(item, str) and item.startswith("image:crop:") for item in evidence_ids
    ):
        errors.append("evidence_artifact_ids_invalid")
    if not isinstance(claim, str) or not claim.strip() or len(claim) > 500:
        errors.append("claim_invalid")
    if errors:
        return None, tuple(errors)
    return VisualCertificate(
        schema_version=schema_version,
        evidence_artifact_ids=tuple(evidence_ids),
        claim=claim.strip(),
    ), ()
