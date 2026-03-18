import logging
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy
from datasets import Dataset, load_dataset
from tqdm import tqdm

from decontamination.util import TypedDataFrame
from elasticsearch import Elasticsearch

from .config import BenchmarkConfig, DatasetConfig, Result
from .util import build_default_index_name, create_es_connection

log = logging.getLogger(__name__)


def get_ngram_mapping(
    language: spacy.Language,
    text: str,
    n: int,
) -> dict[str, list[int]]:
    doc = language(text)
    ngram_docs = [doc[i : i + n] for i in range(len(doc) - n + 1)]
    return {
        ngram_doc.text: [token.i for token in ngram_doc] for ngram_doc in ngram_docs
    }


def ngram_match(
    es: Elasticsearch,
    index_name: str,
    query_dataset: Dataset,
    fields: list[str],
    ngram_size: int,
    search_size: int,
) -> tuple[list[float], list[dict], dict[int, float]]:
    spacy_model = spacy.load("en_core_web_lg")

    match_scores = []
    output_data = []
    # Maps ids in the HF dataset ("original_id") to the list of matching scores with test instances,
    # so that we can compute the max score for decontamination.
    all_train_id_scores = defaultdict(list)
    for datum in tqdm(
        query_dataset, desc="Running n-gram match search on benchmark queries"
    ):
        query_strings = [datum[field] for field in fields]
        if any([s is None for s in query_strings]):
            continue
        query_string_match_scores = []
        query_string_match_tokens = defaultdict(list)
        matching_doc_ids = set()
        doc_id_source_mapping = {}
        match_info = None
        for query_string in query_strings:
            # We compute the match score for each query string for ngram matches as follows:
            # For each token in the query string, we retrieve the training documents that
            # contain ngrams from the query string the token belongs to. Then we compute the
            # match score as the ratio of the tokens in the query string that match that training document.
            query_string_tokens = [d.text for d in spacy_model(query_string)]
            query_string_length = len(query_string_tokens)
            ngram_mapping = get_ngram_mapping(
                spacy_model,
                query_string,
                ngram_size,
            )
            train_doc_matches = defaultdict(set)
            for ngram, tokens in ngram_mapping.items():
                search_output = es.search(
                    index=index_name,
                    search_type="query_then_fetch",
                    rest_total_hits_as_int=True,
                    size=search_size,
                    query={"bool": {"filter": [{"match_phrase": {"text": ngram}}]}},
                )
                for hit_info in search_output["hits"]["hits"]:
                    doc_id = hit_info["_id"]
                    doc = hit_info["_source"]
                    train_doc_matches[doc_id].update(tokens)
                    matching_doc_ids.add(doc_id)
                    doc_id_source_mapping[doc_id] = doc

            query_string_match_scores.append(
                {
                    doc_id: len(matching_tokens) / query_string_length
                    for doc_id, matching_tokens in train_doc_matches.items()
                }
            )
            for doc_id, matching_tokens in train_doc_matches.items():
                query_string_match_tokens[doc_id].append(
                    [query_string_tokens[t] for t in matching_tokens]
                )

        if matching_doc_ids:
            # Averaging the match scores of training documents over all query strings.
            aggregated_match_scores = {
                doc_id: sum([x.get(doc_id, 0.0) for x in query_string_match_scores])
                / len(query_strings)
                for doc_id in matching_doc_ids
            }
            sorted_matches = sorted(
                aggregated_match_scores.items(), key=lambda x: x[1], reverse=True
            )
            match_info = []
            for doc_id, score in sorted_matches:
                match_info.append(
                    {
                        "doc_id": doc_id,
                        "source": doc_id_source_mapping[doc_id],
                        "matching_tokens": query_string_match_tokens[doc_id],
                        "score": score,
                    }
                )
                all_train_id_scores[
                    doc_id_source_mapping[doc_id]["original_id"]
                ].append(score)
            match_score = sorted_matches[0][1]
            match_scores.append(match_score)
            output_data.append(
                {"query": query_strings, "matches": match_info, "score": match_score}
            )
        else:
            match_scores.append(0)
    max_train_match_scores = {
        _id: max(scores) for _id, scores in all_train_id_scores.items()
    }
    return match_scores, output_data, max_train_match_scores


def read_dataset(
    name: str,
    subset: str | None,
    split: str,
    query_fields: list[str] = ["question"],
) -> Dataset:
    log.info("Loading dataset: %s, subset: %s, split: %s", name, subset, split)
    ds = load_dataset(name, subset, split=split)
    log.info("Loaded dataset: %s", ds)
    log.info("\tColumns: %s", ds.column_names)
    log.info("\tNumber of examples: %d", len(ds))

    return ds.remove_columns([c for c in ds.column_names if c not in query_fields])


def search(
    datasets: list[DatasetConfig],
    benchmarks: list[BenchmarkConfig],
    ngram_size: int = 8,
    match_threshold: float = 0.5,
    top_k: int = 10,
    output_dir: Path = Path("outputs"),
) -> None:
    """
    Search for benchmark queries in the indexed datasets and save results to a CSV file.

    Args:
        datasets: A list of DatasetConfig objects specifying the datasets to search.
        benchmarks: A list of BenchmarkConfig objects specifying the benchmarks to search for.
        ngram_size: The size of n-grams to use for matching (default: 8).
        match_threshold: The threshold for considering a match as contamination (default: 0.5).
        top_k: The number of top matches to retrieve for each query (default: 10).
        output_dir: The directory to save the results CSV file (default: "outputs").
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_results = (
        TypedDataFrame(pd.read_csv(output_dir / "results.csv"), Result)
        if (output_dir / "results.csv").exists()
        else TypedDataFrame.new(Result)
    )

    es = create_es_connection()

    for dataset in datasets:
        log.info(
            "Processing dataset: %s (subset: %s, split: %s)...",
            dataset.path,
            dataset.subset,
            dataset.split,
        )
        if not dataset.index_name:
            dataset.index_name = build_default_index_name(
                dataset.path, dataset.subset, dataset.split
            )
        log.info("\tUsing index name: %s", dataset.index_name)
        dataset_num_docs = es.count(index=dataset.index_name)["count"]
        log.info("\tNumber of docs: %d\n", dataset_num_docs)

        for benchmark in benchmarks:
            log.info(
                "\tProcessing benchmark: %s (subset: %s, split: %s)...",
                benchmark.path,
                benchmark.subset,
                benchmark.split,
            )

            result = Result(
                dataset_path=dataset.path,
                dataset_subset=dataset.subset,
                dataset_split=dataset.split,
                dataset_query_filter=dataset.query_filter,
                dataset_query_field=dataset.query_field,
                benchmark_path=benchmark.path,
                benchmark_subset=benchmark.subset,
                benchmark_split=benchmark.split,
                benchmark_query_fields=benchmark.query_fields,
                dataset_num_docs=dataset_num_docs,
                dataset_num_contaminated_docs=0,
                dataset_contamination_fraction=0.0,
                benchmark_num_docs=0,
                benchmark_num_contaminated_docs=0,
                benchmark_contamination_fraction=0.0,
                ngram_size=ngram_size,
                match_threshold=match_threshold,
                contaminated_ids={},
            )

            if any(result.is_same_setup(other) for other in df_results):
                log.info(
                    "\tResults already exist for this dataset and benchmark; skipping..."
                )
                continue

            benchamrk_dataset = read_dataset(
                benchmark.path,
                benchmark.subset,
                benchmark.split,
                query_fields=benchmark.query_fields,
            )
            result.benchmark_num_docs = len(benchamrk_dataset)

            match_scores, output_data, train_id_scores = ngram_match(
                es,
                dataset.index_name,
                benchamrk_dataset,
                benchmark.query_fields,
                ngram_size,
                top_k,
            )
            dataset_contaminated_docs = list(
                _id for _id, score in train_id_scores.items() if score > match_threshold
            )
            result.dataset_num_contaminated_docs = len(dataset_contaminated_docs)
            result.dataset_contamination_fraction = (
                len(dataset_contaminated_docs) / dataset_num_docs
            )
            result.contaminated_ids = {id: set() for id in dataset_contaminated_docs}
            result.benchmark_num_contaminated_docs = len(
                [s for s in match_scores if s > match_threshold]
            )
            result.benchmark_contamination_fraction = (
                result.benchmark_num_contaminated_docs / result.benchmark_num_docs
            )
            log.info(
                f"\tBenchmark match score: {result.benchmark_contamination_fraction:.4f}"
                f" ({result.benchmark_num_contaminated_docs} contaminated examples)"
            )

            df_results.add(result)
            df_results.df.to_csv(output_dir / "results.csv", index=False)
