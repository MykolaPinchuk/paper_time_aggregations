from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the prebuilt paper.pdf if present.
    prebuilt = out_dir / "paper.pdf"
    if not prebuilt.exists():
        print(f"Warning: missing {prebuilt} (bundle will contain results/figures only).")

    _copytree(artifacts / "plots", out_dir / "plots")
    _copytree(artifacts / "10pct_paper_grid", out_dir / "tables" / "10pct_paper_grid")
    _copytree(artifacts / "10pct_paper_grid_noTE", out_dir / "tables" / "10pct_paper_grid_noTE")
    _copytree(artifacts / "10pct_te_lift", out_dir / "tables" / "10pct_te_lift")
    _copytree(artifacts / "combined", out_dir / "tables" / "combined")
    _copytree(artifacts / "figures_main", out_dir / "figures")

    (out_dir / "BUNDLE_README.txt").write_text(
        "This folder is an assembled upload bundle (paper PDF + regenerated tables/figures).\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
