from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-src", required=True, help="Directory containing main.tex")
    ap.add_argument("--out-pdf", required=True)
    args = ap.parse_args()

    paper_src = Path(args.paper_src)
    main_tex = paper_src / "main.tex"
    if not main_tex.exists():
        raise SystemExit(f"Missing {main_tex}. Place LaTeX sources under {paper_src}/")

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise SystemExit("latexmk not found on PATH. Install a LaTeX toolchain or build in a container.")

    build_dir = paper_src / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(
        [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={build_dir}",
            str(main_tex),
        ]
    )
    produced = build_dir / "main.pdf"
    if not produced.exists():
        raise SystemExit(f"Build succeeded but {produced} not found.")

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

