"""
Remove contaminated examples from training datasets based on search results.
"""

import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader
from datasets import Dataset, DatasetDict, DatasetInfo, load_from_disk
from datasets.splits import SplitDict, SplitInfo
from datasets.utils.metadata import DatasetCardData, DatasetInfosDict, MetadataConfigs
from huggingface_hub import HfApi, hf_hub_download

from decontamination.util import read_dataset
from decontamination.util import TypedDataFrame

from .config import BenchmarkConfig, DatasetConfig, Result

log = logging.getLogger(__name__)

SUBSET_FOR_HUB_CONFIG_FILENAME = "subset_for_hub_config.json"
_PARQUET_SHARD_SUFFIX_RE = re.compile(r"-\d+-of-\d+$")


def strip_namespace(path: str) -> str:
    """Strip HuggingFace namespace (user/org) from path. E.g. nvidia/ds -> ds."""
    if "/" in path:
        return path.split("/", 1)[1]
    return path


def build_dataset_name(path: str) -> str:
    """Get decontaminated dataset name: path without namespace + '-decontaminated'."""
    return f"{strip_namespace(path)}-decontaminated"


def validate_results(
    results: list[Result],
    datasets: list[DatasetConfig],
    benchmarks: list[BenchmarkConfig],
) -> None:
    """Check that all dataset-benchmark pairs exist in results. Exit with error if not."""
    result_keys = {
        (
            r.dataset_path,
            r.dataset_subset,
            r.dataset_split,
            r.benchmark_path,
            r.benchmark_subset,
            r.benchmark_split,
        )
        for r in results
    }
    missing = []
    for ds in datasets:
        for bench in benchmarks:
            key = (
                ds.path,
                ds.subset,
                ds.split,
                bench.path,
                bench.subset,
                bench.split,
            )
            if key not in result_keys:
                missing.append(
                    f"  - dataset {ds.path} (subset={ds.subset}, split={ds.split}) "
                    f"x benchmark {bench.path} (subset={bench.subset}, split={bench.split})"
                )
    if missing:
        log.warning(
            "Missing results for the following dataset-benchmark pairs:\n%s\n"
            "Please run 'ds index' and 'ds search' first, then retry.",
            "\n".join(missing),
        )
        sys.exit(1)


def result_matches_benchmarks(r: Result, benchmarks: list[BenchmarkConfig]) -> bool:
    """Check if a result matches any of the given benchmarks."""
    return any(
        r.benchmark_path == b.path
        and r.benchmark_subset == b.subset
        and r.benchmark_split == b.split
        for b in benchmarks
    )


def find_matching_result(
    results: list[Result],
    path: str,
    subset: str | None,
    split: str,
    benchmarks: list[BenchmarkConfig],
) -> Result | None:
    """Return the first Result matching (path, subset, split) and benchmarks, or None."""
    for r in results:
        if (
            r.dataset_path == path
            and r.dataset_subset == subset
            and r.dataset_split == split
            and result_matches_benchmarks(r, benchmarks)
        ):
            return r
    return None


def make_contamination_stat(
    subset: str | None,
    split: str,
    total: int,
    removed: int,
) -> dict:
    """Build a contamination stat dict for README/JSON."""
    return {
        "subset": subset,
        "split": split,
        "total": total,
        "removed": removed,
    }


def collect_contaminated_ids(
    results: list[Result],
    path: str,
    subset: str | None,
    split: str,
    benchmarks: list[BenchmarkConfig],
) -> set[int]:
    """Union contaminated_ids keys for given (path, subset, split) across matching benchmarks."""
    ids_to_remove: set[int] = set()
    for r in results:
        if (
            r.dataset_path == path
            and r.dataset_subset == subset
            and r.dataset_split == split
            and result_matches_benchmarks(r, benchmarks)
        ):
            ids_to_remove.update(r.contaminated_ids.keys())
    return ids_to_remove


def hub_config_name(subset: str | None) -> str:
    """
    Hub dataset configuration name for a source subset (``None`` → ``default``).
    """
    name = "default" if subset is None else subset
    if name == "data":
        raise ValueError(
            "Hub config name 'data' is reserved by the datasets library; "
            "use a different subset name in datasets.yaml."
        )
    return name


def load_subset_for_hub_config_map(
    output_path: Path,
) -> dict[str, str | None] | None:
    """Load subset→Hub config map written alongside multi-config exports."""
    p = output_path / SUBSET_FOR_HUB_CONFIG_FILENAME
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        out: dict[str, str | None] = {}
        for k, v in raw.items():
            out[str(k)] = None if v is None else str(v)
        return out
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not read %s: %s", p, e)
        return None


def save_subset_for_hub_config_map(
    output_path: Path, mapping: dict[str, str | None]
) -> None:
    """Persist which source subset each Hub config_name corresponds to (for result lookup)."""
    p = output_path / SUBSET_FOR_HUB_CONFIG_FILENAME
    with open(p, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")


def load_existing_decontaminated_local(
    output_path: Path,
) -> dict[str, DatasetDict] | None:
    """
    Load existing decontaminated data: one ``DatasetDict`` per Hub config subdirectory.
    Legacy single-folder layout (``dataset_dict.json`` at output root) is not loaded.
    """
    if not output_path.exists():
        return None
    if (output_path / "dataset_dict.json").is_file():
        log.debug(
            "Ignoring legacy decontaminated layout at %s (expected per-config subdirs)",
            output_path,
        )
        return None
    subdirs = sorted(
        p for p in output_path.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not subdirs:
        return None
    out: dict[str, DatasetDict] = {}
    for sub in subdirs:
        if not (sub / "dataset_dict.json").is_file():
            continue
        try:
            loaded = load_from_disk(str(sub))
            if isinstance(loaded, DatasetDict):
                out[sub.name] = loaded
        except Exception as e:
            log.warning(
                "Could not load existing config %s from %s: %s", sub.name, sub, e
            )
    return out if out else None


def hub_splits(repo_id: str) -> set[str] | None:
    """
    Return split stem keys seen in Parquet paths on the hub (e.g. ``train`` from
    ``data/train-*.parquet`` or ``{config}/train-*.parquet``).
    Returns None if the repo does not exist.
    """
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return None
    splits: set[str] = set()
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        if len(parts) != 2:
            continue
        _prefix, fname = parts
        if not fname.endswith(".parquet"):
            continue
        stem = fname[: -len(".parquet")]
        stem = _PARQUET_SHARD_SUFFIX_RE.sub("", stem)
        splits.add(stem)
    return splits


def hf_repo_exists(huggingface_id: str, dataset_name: str) -> bool:
    """Check if the HF dataset repo exists."""
    repo_id = f"{huggingface_id}/{dataset_name}"
    try:
        HfApi().repo_info(repo_id=repo_id, repo_type="dataset")
        return True
    except Exception:
        return False


def split_key(subset: str | None, split: str) -> str:
    """Flatten subset+split to a single key for DatasetDict (HF-compatible: word chars and dots)."""
    if subset is not None:
        return f"{subset}.{split}"
    return split


def merge_into_config_dict(
    existing: dict[str, DatasetDict] | None,
    subset_for_config: dict[str, str | None],
    subset: str | None,
    split: str,
    dataset: Dataset,
) -> dict[str, DatasetDict]:
    """Merge a filtered split into ``existing[hub_config_name(subset)][split]``."""
    cfg = hub_config_name(subset)
    if cfg not in subset_for_config:
        subset_for_config[cfg] = subset
    elif subset_for_config[cfg] != subset:
        log.warning(
            "Hub config %r already tied to subset %r; also got subset %r for split %r (last write wins)",
            cfg,
            subset_for_config[cfg],
            subset,
            split,
        )
    if existing is None:
        return {cfg: DatasetDict({split: dataset})}
    if cfg not in existing:
        existing[cfg] = DatasetDict({split: dataset})
    else:
        existing[cfg][split] = dataset
    return existing


def build_decontamination_info(
    source_path: str,
    benchmarks: list[BenchmarkConfig],
    decontamination_stats: list[dict],
    results: list[Result],
) -> dict:
    """Build decontamination info dict for JSON export."""
    info: dict = {
        "source_dataset": source_path,
        "benchmarks": [
            {
                "display_name": b.display_name,
                "path": b.path,
                "subset": b.subset,
                "split": b.split,
            }
            for b in benchmarks
        ],
        "contamination_stats": list(decontamination_stats),
        "splits": [
            split_key(s.get("subset"), s["split"]) for s in decontamination_stats
        ],
    }
    # Add per-split search params from results
    for stat in info["contamination_stats"]:
        subset, split = stat.get("subset"), stat["split"]
        r = find_matching_result(results, source_path, subset, split, benchmarks)
        if r is not None:
            stat["ngram_size"] = r.ngram_size
            stat["match_threshold"] = r.match_threshold
    return info


def _benchmark_display_name(
    benchmarks: list[BenchmarkConfig],
    benchmark_path: str,
    benchmark_subset: str | None,
    benchmark_split: str,
) -> str:
    """Resolve benchmark display name from config."""
    for b in benchmarks:
        if (
            b.path == benchmark_path
            and b.subset == benchmark_subset
            and b.split == benchmark_split
        ):
            return b.display_name
    return f"{benchmark_path} ({benchmark_subset or 'default'}/{benchmark_split})"


def _format_contamination_rate(fraction: float) -> str:
    """Format contamination fraction as percentage string."""
    pct = fraction * 100
    return f"{pct:.4f}%" if pct < 1 else f"{pct:.2f}%"


def build_readme_context(
    source_path: str,
    benchmarks: list[BenchmarkConfig],
    decontamination_stats: list[dict],
    results: list[Result],
) -> dict:
    """Build template context for the decontamination README section."""
    matching_results = [
        r
        for r in results
        if r.dataset_path == source_path and result_matches_benchmarks(r, benchmarks)
    ]

    benchmark_order = {(b.path, b.subset, b.split): i for i, b in enumerate(benchmarks)}

    def sort_key(r: Result) -> tuple:
        subset = r.dataset_subset or ""
        split = r.dataset_split
        bench_key = (r.benchmark_path, r.benchmark_subset, r.benchmark_split)
        bench_idx = benchmark_order.get(bench_key, 999)
        return (subset, split, bench_idx)

    matching_results.sort(key=sort_key)

    # Table 1: Settings rows
    seen_settings: set[tuple[int, float]] = set()
    settings_rows: list[dict] = []
    for r in matching_results:
        key = (r.ngram_size, r.match_threshold)
        if key not in seen_settings:
            seen_settings.add(key)
            settings_rows.append({"param": "N-gram size", "value": r.ngram_size})
            settings_rows.append(
                {"param": "Match threshold", "value": r.match_threshold}
            )

    # Table 2: Split-benchmark rows with rowspan
    subset_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in matching_results:
        sub = r.dataset_subset if r.dataset_subset is not None else "-"
        spl = r.dataset_split
        subset_counts[sub] += 1
        split_counts[(sub, spl)] += 1

    seen_subset: set[str] = set()
    seen_split: set[tuple[str, str]] = set()
    split_benchmark_rows: list[dict] = []
    for r in matching_results:
        sub = r.dataset_subset if r.dataset_subset is not None else "-"
        spl = r.dataset_split
        subset_rowspan = subset_counts[sub] if sub not in seen_subset else 0
        split_rowspan = split_counts[(sub, spl)] if (sub, spl) not in seen_split else 0
        if subset_rowspan > 0:
            seen_subset.add(sub)
        if split_rowspan > 0:
            seen_split.add((sub, spl))

        split_benchmark_rows.append(
            {
                "subset": sub,
                "split": spl,
                "subset_rowspan": subset_rowspan,
                "split_rowspan": split_rowspan,
                "benchmark_display_name": _benchmark_display_name(
                    benchmarks, r.benchmark_path, r.benchmark_subset, r.benchmark_split
                ),
                "dataset_num_docs": r.dataset_num_docs,
                "dataset_num_docs_fmt": f"{r.dataset_num_docs:,}",
                "dataset_num_contaminated_docs": r.dataset_num_contaminated_docs,
                "contamination_rate_pct": _format_contamination_rate(
                    r.dataset_contamination_fraction
                ),
                "benchmark_num_docs": r.benchmark_num_docs,
                "benchmark_num_contaminated_docs": r.benchmark_num_contaminated_docs,
                "benchmark_contamination_rate_pct": _format_contamination_rate(
                    r.benchmark_contamination_fraction
                ),
            }
        )

    # Dataset summary (whole-dataset aggregates from decontamination_stats)
    total_original = sum(s["total"] for s in decontamination_stats)
    total_removed = sum(s["removed"] for s in decontamination_stats)
    total_remaining = total_original - total_removed
    overall_rate = (
        _format_contamination_rate(total_removed / total_original)
        if total_original > 0
        else "0%"
    )
    summary_row = {
        "total_original": total_original,
        "total_original_fmt": f"{total_original:,}",
        "total_removed": total_removed,
        "total_removed_fmt": f"{total_removed:,}",
        "total_remaining": total_remaining,
        "total_remaining_fmt": f"{total_remaining:,}",
        "overall_rate": overall_rate,
    }

    # Benchmarks list for template
    benchmarks_data = [
        {
            "display_name": b.display_name,
            "path": b.path,
            "subset": b.subset,
            "split": b.split,
            "data_files": b.data_files,
        }
        for b in benchmarks
    ]

    return {
        "source_path": source_path,
        "source_url": f"https://huggingface.co/datasets/{source_path}",
        "benchmarks": benchmarks_data,
        "settings_rows": settings_rows,
        "split_benchmark_rows": split_benchmark_rows,
        "summary_row": summary_row,
    }


def extract_yaml_and_rest(content: str) -> tuple[str, str]:
    """Extract YAML block (between first --- and second ---) and rest of content."""
    if not content.strip().startswith("---"):
        return "", content
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[1].strip(), parts[2].strip()
    return "", content


def _dataset_info_and_data_files_for_config(
    merged: DatasetDict, config_name: str
) -> tuple[DatasetInfo, list[dict[str, str]]]:
    """One Hub config: ``DatasetInfo`` and ``data_files`` entries for README metadata."""
    data_dir = config_name if config_name != "default" else "data"
    merged._check_values_type()
    merged._check_values_features()
    split_names = sorted(merged.keys(), key=str)
    first = merged[split_names[0]]
    features = first.features
    splits = SplitDict()
    data_files: list[dict[str, str]] = []
    total_dataset_size = 0
    total_download_size = 0
    for split_name in split_names:
        ds = merged[split_name]
        nbytes = ds._data.nbytes
        splits.add(
            SplitInfo(
                name=str(split_name),
                num_bytes=nbytes,
                num_examples=len(ds),
            )
        )
        total_dataset_size += nbytes
        total_download_size += nbytes
        data_files.append(
            {
                "split": str(split_name),
                "path": f"{data_dir}/{split_name}-*",
            }
        )
    info = DatasetInfo(
        config_name=config_name,
        features=features,
        splits=splits,
        download_size=total_download_size,
        dataset_size=total_dataset_size,
    )
    return info, data_files


def build_dataset_card_metadata_dict(
    merged_by_config: dict[str, DatasetDict],
) -> dict[str, Any]:
    """
    Build README ``dataset_info`` and ``configs`` for all Hub configs so they align
    with per-config ``DatasetDict.push_to_hub`` (``data/{split}-*`` vs
    ``{config}/{split}-*``). Byte sizes use in-memory Arrow tables.
    """
    card = DatasetCardData()
    infos: dict[str, DatasetInfo] = {}
    metas: dict[str, dict] = {}
    for cfg in sorted(str(k) for k in merged_by_config.keys()):
        dd = merged_by_config[cfg]
        info, data_files = _dataset_info_and_data_files_for_config(dd, cfg)
        infos[cfg] = info
        metas[cfg] = {"data_files": data_files}
    DatasetInfosDict(infos).to_dataset_card_data(card)
    MetadataConfigs(metas).to_dataset_card_data(card)
    return card.to_dict()


def _render_decontamination_section(
    source_path: str,
    benchmarks: list[BenchmarkConfig],
    decontamination_stats: list[dict],
    results: list[Result],
) -> str:
    """Render the Decontamination section using Jinja2 template."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("decontamination_section.md.j2")
    context = build_readme_context(
        source_path, benchmarks, decontamination_stats, results
    )
    return template.render(**context)


def build_readme(
    source_path: str,
    benchmarks: list[BenchmarkConfig],
    decontamination_stats: list[dict],
    results: list[Result],
    merged_by_config: dict[str, DatasetDict],
) -> str:
    """Build README with YAML metadata at top (per HF rules), then Decontamination section, then original content."""
    decontam_section = _render_decontamination_section(
        source_path, benchmarks, decontamination_stats, results
    )
    try:
        readme_path = hf_hub_download(
            repo_id=source_path,
            filename="README.md",
            repo_type="dataset",
        )
        with open(readme_path) as f:
            original = f.read()
    except Exception as e:
        log.warning("Could not fetch original README from %s: %s", source_path, e)
        original = ""

    yaml_block, rest = extract_yaml_and_rest(original)
    yaml_dict: dict = {}
    if yaml_block:
        try:
            yaml_dict = yaml.safe_load(yaml_block) or {}
        except yaml.YAMLError:
            yaml_dict = {}

    card_meta = build_dataset_card_metadata_dict(merged_by_config)
    yaml_dict["dataset_info"] = card_meta["dataset_info"]
    yaml_dict["configs"] = card_meta["configs"]

    # Add decontamination metadata to YAML
    yaml_dict["decontamination"] = {
        "source_dataset": source_path,
        "benchmarks": [
            {"path": b.path, "subset": b.subset, "split": b.split} for b in benchmarks
        ],
        "contamination_stats": decontamination_stats,
    }

    # Write: YAML at top (per HF rules), then Decontamination section, then rest
    yaml_str = yaml.dump(
        yaml_dict, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    return (
        f"---\n{yaml_str.rstrip()}\n---\n\n{decontam_section}\n\n---\n\n{rest}".strip()
    )


def upload_metadata_to_hf(api: HfApi, repo_id: str, output_path: Path) -> None:
    """Upload README.md and decontamination_info.json from output_path to repo_id."""
    readme_path = output_path / "README.md"
    if readme_path.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            log.warning("Could not upload README to HF: %s", e)
    info_path = output_path / "decontamination_info.json"
    if info_path.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(info_path),
                path_in_repo="decontamination_info.json",
                repo_id=repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            log.warning("Could not upload decontamination_info to HF: %s", e)


def _result_subsets_for_config(
    config_name: str, subset_for_config: dict[str, str | None]
) -> list[str | None]:
    """Source ``subset`` values to try when matching ``Result`` rows for this Hub config."""
    if config_name in subset_for_config:
        return [subset_for_config[config_name]]
    if config_name == "default":
        return [None, "default"]
    return [config_name]


def ensure_stats_for_merged(
    results: list[Result],
    path: str,
    benchmarks: list[BenchmarkConfig],
    merged_by_config: dict[str, DatasetDict],
    subset_for_config: dict[str, str | None],
    decontamination_stats: list[dict],
) -> None:
    """Add stats for splits present on disk that are missing from decontamination_stats."""
    stats_keys = {(s["subset"], s["split"]) for s in decontamination_stats}
    for config_name in sorted(str(k) for k in merged_by_config.keys()):
        dd = merged_by_config[config_name]
        for split in sorted(str(k) for k in dd.keys()):
            resolved_subset: str | None = None
            r: Result | None = None
            for cand in _result_subsets_for_config(config_name, subset_for_config):
                r = find_matching_result(results, path, cand, split, benchmarks)
                if r is not None:
                    resolved_subset = cand
                    break
            if r is None or (resolved_subset, split) in stats_keys:
                continue
            ids_to_remove = collect_contaminated_ids(
                results, path, resolved_subset, split, benchmarks
            )
            decontamination_stats.append(
                make_contamination_stat(
                    resolved_subset, split, r.dataset_num_docs, len(ids_to_remove)
                )
            )
            stats_keys.add((resolved_subset, split))


def _unique_replace_backup_path(output_path: Path) -> Path:
    """Return a non-existent sibling path for moving output_path aside during replace."""
    parent = output_path.parent
    base = output_path.name + ".replace-backup"
    pid = os.getpid()
    candidate = parent / f"{base}.{pid}"
    if not candidate.exists():
        return candidate
    counter = 0
    while True:
        candidate = parent / f"{base}.{pid}.{counter}"
        counter += 1
        if not candidate.exists():
            return candidate


def save_datasetdict_replacing(merged: DatasetDict, output_path: Path) -> None:
    """
    Save merged to output_path, replacing any existing directory.

    Writes to a sibling temp directory first so save_to_disk never targets the
    same path as load_from_disk (HF datasets forbid that self-overwrite).
    """
    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = tempfile.mkdtemp(
        dir=parent, prefix=f".{output_path.name}.tmp."
    )
    try:
        merged.save_to_disk(tmp_path)
        if not output_path.exists():
            os.rename(tmp_path, output_path)
            tmp_path = None
            return

        backup = _unique_replace_backup_path(output_path)
        os.rename(output_path, backup)
        try:
            assert tmp_path is not None
            os.rename(tmp_path, output_path)
        except Exception:
            try:
                if backup.exists():
                    os.rename(backup, output_path)
            except OSError as e2:
                log.warning(
                    "Failed to restore previous dataset at %s from backup: %s",
                    output_path,
                    e2,
                )
            raise
        else:
            shutil.rmtree(backup)
            tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)


def save_multi_config_replacing(
    merged_by_config: dict[str, DatasetDict],
    subset_for_config: dict[str, str | None],
    output_path: Path,
) -> None:
    """
    Save each Hub config under ``output_path/<config_name>/`` and write
    ``subset_for_hub_config.json`` at ``output_path`` root.
    """
    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_root: str | None = tempfile.mkdtemp(
        dir=parent, prefix=f".{output_path.name}.multi.tmp."
    )
    try:
        tmp_path_obj = Path(tmp_root)
        for cfg in sorted(str(k) for k in merged_by_config.keys()):
            save_datasetdict_replacing(merged_by_config[cfg], tmp_path_obj / cfg)
        save_subset_for_hub_config_map(tmp_path_obj, subset_for_config)
        if not output_path.exists():
            os.rename(tmp_root, output_path)
            tmp_root = None
            return

        backup = _unique_replace_backup_path(output_path)
        os.rename(output_path, backup)
        try:
            assert tmp_root is not None
            os.rename(tmp_root, output_path)
        except Exception:
            try:
                if backup.exists():
                    os.rename(backup, output_path)
            except OSError as e2:
                log.warning(
                    "Failed to restore previous dataset at %s from backup: %s",
                    output_path,
                    e2,
                )
            raise
        else:
            shutil.rmtree(backup)
            tmp_root = None
    finally:
        if tmp_root is not None and os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)


def remove(
    datasets: list[DatasetConfig],
    benchmarks: list[BenchmarkConfig],
    results_path: Path = Path("outputs/results.csv"),
    output_dir: Path = Path("outputs"),
    huggingface_id: str | None = None,
) -> None:
    """
    Remove contaminated examples from training datasets based on search results.

    Reads results from the CSV, validates all dataset-benchmark pairs exist,
    filters out contaminated rows, merges subsets/splits of the same main dataset
    (one Hub config per subset, split names as in the source dataset),
    saves locally under per-config subdirectories, and optionally pushes to Hugging Face.

    Args:
        datasets: List of DatasetConfig specifying datasets to decontaminate.
        benchmarks: List of BenchmarkConfig specifying benchmarks used for search.
        results_path: Path to results.csv (default: outputs/results.csv).
        output_dir: Directory for local output (default: outputs).
        huggingface_id: Optional HF namespace for upload (e.g. 'myorg').
    """
    if not results_path.exists():
        log.error(
            "Results file not found: %s. Run 'ds index' and 'ds search' first.",
            results_path,
        )
        sys.exit(1)

    df = pd.read_csv(results_path)
    df_results = TypedDataFrame(df, Result)
    results = list(df_results)

    validate_results(results, datasets, benchmarks)

    os.makedirs(output_dir, exist_ok=True)

    # Group datasets by path only
    by_path: dict[str, list[DatasetConfig]] = {}
    for ds in datasets:
        by_path.setdefault(ds.path, []).append(ds)

    for path, path_datasets in by_path.items():
        dataset_name = build_dataset_name(path)
        output_path = output_dir / dataset_name

        existing_local = load_existing_decontaminated_local(output_path)
        subset_for_config: dict[str, str | None] = dict(
            load_subset_for_hub_config_map(output_path) or {}
        )

        decontamination_stats: list[dict] = []
        merged_by_config: dict[str, DatasetDict] | None = (
            {k: v for k, v in existing_local.items()} if existing_local else None
        )

        for ds_config in path_datasets:
            cfg = hub_config_name(ds_config.subset)
            if (
                merged_by_config is not None
                and cfg in merged_by_config
                and ds_config.split in merged_by_config[cfg]
            ):
                log.info(
                    "Skipping %s (subset=%s, split=%s): already decontaminated",
                    path,
                    ds_config.subset,
                    ds_config.split,
                )
                ids_to_remove = collect_contaminated_ids(
                    results, path, ds_config.subset, ds_config.split, benchmarks
                )
                r = find_matching_result(
                    results, path, ds_config.subset, ds_config.split, benchmarks
                )
                if r is not None:
                    decontamination_stats.append(
                        make_contamination_stat(
                            ds_config.subset,
                            ds_config.split,
                            r.dataset_num_docs,
                            len(ids_to_remove),
                        )
                    )
                continue

            ids_to_remove = collect_contaminated_ids(
                results,
                path,
                ds_config.subset,
                ds_config.split,
                benchmarks,
            )
            log.info(
                "Loading %s (subset=%s, split=%s), removing %d contaminated examples",
                path,
                ds_config.subset,
                ds_config.split,
                len(ids_to_remove),
            )
            full_ds = read_dataset(
                path,
                ds_config.subset,
                split=ds_config.split,
            )
            filtered = full_ds.filter(
                lambda _, idx: idx not in ids_to_remove,
                with_indices=True,
            )
            decontamination_stats.append(
                make_contamination_stat(
                    ds_config.subset,
                    ds_config.split,
                    len(full_ds),
                    len(ids_to_remove),
                )
            )
            merged_by_config = merge_into_config_dict(
                merged_by_config,
                subset_for_config,
                ds_config.subset,
                ds_config.split,
                filtered,
            )

        if merged_by_config is None:
            continue

        ensure_stats_for_merged(
            results,
            path,
            benchmarks,
            merged_by_config,
            subset_for_config,
            decontamination_stats,
        )

        log.info("Saving decontaminated dataset to %s", output_path)
        save_multi_config_replacing(merged_by_config, subset_for_config, output_path)

        readme_content = build_readme(
            path, benchmarks, decontamination_stats, results, merged_by_config
        )
        readme_path = output_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        decontamination_info = build_decontamination_info(
            path, benchmarks, decontamination_stats, results
        )
        info_path = output_path / "decontamination_info.json"
        with open(info_path, "w") as f:
            json.dump(decontamination_info, f, indent=2)

        if huggingface_id:
            repo_id = f"{huggingface_id}/{dataset_name}"
            log.info("Pushing to Hugging Face: %s", repo_id)
            for cfg in sorted(str(k) for k in merged_by_config.keys()):
                merged_by_config[cfg].push_to_hub(repo_id, config_name=cfg)
            upload_metadata_to_hf(HfApi(), repo_id, output_path)
