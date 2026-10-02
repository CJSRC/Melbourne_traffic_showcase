from __future__ import annotations

from pathlib import Path


def resolve_hf_snapshot_path(model_id: str) -> str:
    candidate = Path(model_id)
    if candidate.exists():
        return str(candidate)
    if "/" not in model_id:
        return model_id
    org, name = model_id.split("/", 1)
    external_dir = Path(__file__).resolve().parent / "external" / "hf_models" / f"{org}--{name}"
    if external_dir.exists():
        return str(external_dir)
    snapshot_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{name}" / "snapshots"
    if not snapshot_root.exists():
        return model_id
    snapshots = sorted((path for path in snapshot_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not snapshots:
        return model_id
    return str(snapshots[-1])


def local_files_only_default(model_id: str) -> bool:
    return Path(resolve_hf_snapshot_path(model_id)).exists()


def rebase_output_root(path_str: str, outputs_root: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    basename = path.name
    rebased = outputs_root / basename
    if rebased.exists():
        return rebased
    parent_basename = path.parent.name
    if parent_basename:
        rebased = outputs_root / parent_basename
        if rebased.exists():
            return rebased
    return path
