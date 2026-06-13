from __future__ import annotations

import argparse
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

RUNTIME_PATHS = (
    Path("data/processed"),
    Path("artifacts/search"),
    Path("artifacts/recsys"),
    Path("artifacts/reports"),
    Path("artifacts/registry"),
)


@dataclass(frozen=True)
class BundlePlan:
    files: tuple[Path, ...]
    total_bytes: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Package the generated MARS runtime data/artifacts for a fresh-machine "
            "Docker Compose review."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/mars_runtime_bundle.zip"),
        help="Output zip path. Extract this zip at the repository root on another PC.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help=(
            "Also include the 50K H&M product image files referenced by "
            "data/processed/products.parquet. This is recommended for dashboard "
            "image previews, but makes the bundle much larger."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the file count and size without writing a zip.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    plan = build_plan(root, include_images=args.include_images)
    size_gb = plan.total_bytes / (1024**3)
    print(f"Runtime bundle plan: {len(plan.files):,} files, {size_gb:.2f} GiB")
    if args.dry_run:
        for path in plan.files[:20]:
            print(path.as_posix())
        if len(plan.files) > 20:
            print(f"... {len(plan.files) - 20:,} more files")
        return

    output = args.output
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for relative_path in plan.files:
            archive.write(root / relative_path, relative_path.as_posix())
    print(f"Wrote {output}")


def build_plan(root: Path, *, include_images: bool = False) -> BundlePlan:
    files: set[Path] = set()
    for relative_path in RUNTIME_PATHS:
        files.update(_iter_existing_files(root, relative_path))
    if include_images:
        files.update(_iter_product_images(root))
    sorted_files = tuple(sorted(files, key=lambda path: path.as_posix()))
    total_bytes = sum((root / path).stat().st_size for path in sorted_files)
    return BundlePlan(files=sorted_files, total_bytes=total_bytes)


def _iter_existing_files(root: Path, relative_path: Path) -> Iterable[Path]:
    path = root / relative_path
    if path.is_file():
        yield relative_path
        return
    if not path.is_dir():
        return
    for child in path.rglob("*"):
        if child.is_file():
            yield child.relative_to(root)


def _iter_product_images(root: Path) -> Iterable[Path]:
    products_path = root / "data/processed/products.parquet"
    if not products_path.exists():
        raise FileNotFoundError(
            "data/processed/products.parquet is required when --include-images is used"
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read products.parquet") from exc

    products = pd.read_parquet(products_path, columns=["image_path"])
    for raw_path in products["image_path"].dropna().astype(str).unique():
        relative_path = Path(raw_path)
        if relative_path.is_absolute():
            try:
                relative_path = relative_path.relative_to(root)
            except ValueError:
                continue
        image_path = root / relative_path
        if image_path.is_file():
            yield relative_path


if __name__ == "__main__":
    main()
