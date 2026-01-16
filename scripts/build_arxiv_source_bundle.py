from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create an arXiv TeX source upload bundle.")
    ap.add_argument("--paper-src", required=True, help="Directory containing main.tex, sections/, figures/, tables/")
    ap.add_argument("--out-dir", required=True, help="Output directory for the source bundle")
    ap.add_argument(
        "--write-tar-gz",
        action="store_true",
        help="Also write <out-dir>.tar.gz suitable for arXiv upload.",
    )
    args = ap.parse_args()

    paper_src = Path(args.paper_src)
    main_tex = paper_src / "main.tex"
    if not main_tex.exists():
        raise SystemExit(f"Missing {main_tex}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Core sources
    shutil.copy2(main_tex, out_dir / "main.tex")
    bib = paper_src / "references.bib"
    if bib.exists():
        shutil.copy2(bib, out_dir / "references.bib")

    _copytree(paper_src / "sections", out_dir / "sections")
    _copytree(paper_src / "figures", out_dir / "figures")
    _copytree(paper_src / "tables", out_dir / "tables")

    # Keep build products out of the upload.
    for rel in ("_build", ".latexmkrc"):
        p = out_dir / rel
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    (out_dir / "ARXIV_BUILD_INSTRUCTIONS.txt").write_text(
        "Build locally (from this folder):\n"
        "  pdflatex main.tex\n"
        "  bibtex main\n"
        "  pdflatex main.tex\n"
        "  pdflatex main.tex\n",
        encoding="utf-8",
    )

    if args.write_tar_gz:
        tar_path = out_dir.with_suffix(out_dir.suffix + ".tar.gz") if out_dir.suffix else Path(str(out_dir) + ".tar.gz")
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as tf:
            # Only add files; adding directories recursively would duplicate entries.
            for p in sorted(out_dir.rglob("*")):
                if p.is_dir():
                    continue
                tf.add(p, arcname=str(p.relative_to(out_dir)))
        print(f"Wrote {tar_path}")

    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
