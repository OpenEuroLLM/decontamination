"""
Remove contaminated examples from training datasets based on search results.
"""
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download

from decontamination.util import TypedDataFrame

from .config import BenchmarkConfig, DatasetConfig, Result

log = logging.getLogger(__name__)


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
        "fraction": removed / total if total else 0,
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


def normalize_dataset_dict_keys(dd: DatasetDict) -> DatasetDict:
    """Normalize keys: 'subset--split' -> 'subset.split' for HF compatibility."""
    return DatasetDict({str(k).replace("--", "."): v for k, v in dd.items()})


def load_existing_decontaminated_local(output_path: Path) -> DatasetDict | None:
    """
    Load existing decontaminated dataset from local disk only.
    Does not download from HF - use hub_splits() to check what is on the hub.
    """
    if not output_path.exists():
        return None
    try:
        loaded = load_from_disk(str(output_path))
        if isinstance(loaded, DatasetDict):
            return normalize_dataset_dict_keys(loaded)
    except Exception as e:
        log.warning("Could not load existing from %s: %s", output_path, e)
    return None


def hub_splits(repo_id: str) -> set[str] | None:
    """
    Return the set of split keys (e.g. 'train', 'default.chat') that exist on the hub.
    Returns None if the repo does not exist.
    """
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return None
    splits: set[str] = set()
    for f in files:
        if f.startswith("data/") and f.endswith(".parquet"):
            # data/train.parquet -> train, data/default.chat.parquet -> default.chat
            name = f[len("data/") : -len(".parquet")]
            splits.add(name)
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


def parse_split_key(key: str) -> tuple[str | None, str]:
    """Parse key back to (subset, split). 'subset.split' -> (subset, split), 'split' -> (None, split)."""
    if "." in key:
        parts = key.split(".", 1)
        return parts[0], parts[1]
    return None, key


def merge_into_dict(
    existing: DatasetDict | None,
    subset: str | None,
    split: str,
    dataset: Dataset,
) -> DatasetDict:
    """Merge a filtered dataset into existing DatasetDict. Uses flat keys like 'subset--split'."""
    key = split_key(subset, split)
    if existing is None:
        return DatasetDict({key: dataset})
    existing[key] = dataset
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
        "splits": [split_key(s.get("subset"), s["split"]) for s in decontamination_stats],
    }
    # Add per-split search params from results
    for stat in info["contamination_stats"]:
        subset, split = stat.get("subset"), stat["split"]
        r = find_matching_result(results, source_path, subset, split, benchmarks)
        if r is not None:
            stat["ngram_size"] = r.ngram_size
            stat["match_threshold"] = r.match_threshold
    return info


def extract_yaml_and_rest(content: str) -> tuple[str, str]:
    """Extract YAML block (between first --- and second ---) and rest of content."""
    if not content.strip().startswith("---"):
        return "", content
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[1].strip(), parts[2].strip()
    return "", content


def build_readme(
    source_path: str,
    benchmarks: list[BenchmarkConfig],
    decontamination_stats: list[dict],
) -> str:
    """Build README with YAML metadata at top (per HF rules), then Decontamination section, then original content."""
    decontam_section = format_decontamination_section(
        source_path, benchmarks, decontamination_stats
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

    # Add decontamination metadata to YAML
    yaml_dict["decontamination"] = {
        "source_dataset": source_path,
        "benchmarks": [
            {"path": b.path, "subset": b.subset, "split": b.split}
            for b in benchmarks
        ],
        "contamination_stats": decontamination_stats,
    }

    # Write: YAML at top (per HF rules), then Decontamination section, then rest
    yaml_str = yaml.dump(yaml_dict, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return f"---\n{yaml_str.rstrip()}\n---\n\n{decontam_section}\n\n---\n\n{rest}".strip()


def format_decontamination_section(
    source_path: str,
    benchmarks: list[BenchmarkConfig],
    stats: list[dict],
) -> str:
    """Format the Decontamination section for the README."""
    lines = [
        "## Decontamination",
        "",
        f"This dataset is a decontaminated version of [{source_path}](https://huggingface.co/datasets/{source_path}).",
        "",
        "### Benchmarks used",
        "",
    ]
    for b in benchmarks:
        lines.append(f"- **{b.display_name}**: `{b.path}` (subset={b.subset}, split={b.split})")
    lines.extend(["", "### Contamination statistics", ""])
    for s in stats:
        subset_split = f"{s['subset']}/{s['split']}" if s.get("subset") else s["split"]
        lines.append(
            f"- **{subset_split}**: {s['removed']} examples removed "
            f"({s['fraction']:.6f} of {s['total']} total)"
        )
    return "\n".join(lines)


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


def ensure_stats_for_merged(
    results: list[Result],
    path: str,
    benchmarks: list[BenchmarkConfig],
    merged: DatasetDict,
    decontamination_stats: list[dict],
) -> None:
    """Add stats for splits in merged that are missing from decontamination_stats."""
    stats_keys = {(s["subset"], s["split"]) for s in decontamination_stats}
    for key in merged.keys():
        subset, split = parse_split_key(str(key))
        if (subset, split) not in stats_keys:
            ids_to_remove = collect_contaminated_ids(
                results, path, subset, split, benchmarks
            )
            r = find_matching_result(results, path, subset, split, benchmarks)
            if r is not None:
                total = r.dataset_num_docs
                removed = len(ids_to_remove)
                decontamination_stats.append(
                    make_contamination_stat(subset, split, total, removed)
                )
                stats_keys.add((subset, split))


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
    filters out contaminated rows, merges subsets/splits of the same main dataset,
    saves locally, and optionally pushes to Hugging Face.

    Args:
        datasets: List of DatasetConfig specifying datasets to decontaminate.
        benchmarks: List of BenchmarkConfig specifying benchmarks used for search.
        results_path: Path to results.csv (default: outputs/results.csv).
        output_dir: Directory for local output (default: outputs).
        huggingface_id: Optional HF namespace for upload (e.g. 'myorg').
    """
    if not results_path.exists():
        log.error("Results file not found: %s. Run 'ds index' and 'ds search' first.", results_path)
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
        existing = existing_local

        decontamination_stats: list[dict] = []
        merged: DatasetDict | None = existing

        for ds_config in path_datasets:
            key = split_key(ds_config.subset, ds_config.split)
            if existing is not None and key in existing:
                log.info(
                    "Skipping %s (subset=%s, split=%s): already decontaminated",
                    path,
                    ds_config.subset,
                    ds_config.split,
                )
                # Add stats from results for README (use first matching result for total)
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
            full_ds = load_dataset(
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
            merged = merge_into_dict(
                merged,
                ds_config.subset,
                ds_config.split,
                filtered,
            )

        if merged is None:
            continue

        # Ensure README has stats for all splits in merged (including from previous runs)
        ensure_stats_for_merged(
            results, path, benchmarks, merged, decontamination_stats
        )

        # Save and upload all decontaminated splits (existing + newly processed)
        log.info("Saving decontaminated dataset to %s", output_path)
        merged.save_to_disk(str(output_path))

        readme_content = build_readme(path, benchmarks, decontamination_stats)
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
            merged.push_to_hub(repo_id)
            upload_metadata_to_hf(HfApi(), repo_id, output_path)
