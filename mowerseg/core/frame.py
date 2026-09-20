from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class Frame:
    """Camera / image / video frame entering the perception pipeline."""

    image: Image.Image
    frame_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_image(
        cls,
        image: Image.Image | str | Path,
        *,
        frame_id: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Frame:
        if isinstance(image, (str, Path)):
            path = Path(image)
            rgb = Image.open(path).convert("RGB")
            return cls(
                image=rgb,
                frame_id=frame_id or path.stem,
                source=source or str(path),
                metadata=metadata or {},
            )
        return cls(
            image=image.convert("RGB"),
            frame_id=frame_id,
            source=source,
            metadata=metadata or {},
        )
