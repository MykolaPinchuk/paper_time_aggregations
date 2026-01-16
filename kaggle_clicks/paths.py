from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def runs(self) -> Path:
        return self.root / "runs"


def get_paths() -> Paths:
    return Paths(root=Path(__file__).resolve().parents[1])

