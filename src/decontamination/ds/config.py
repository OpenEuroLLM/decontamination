from dataclasses import dataclass

import pandas as pd

from decontamination.util import TypedRow


@dataclass
class DatasetConfig:
    """
    Configuration for a dataset to be indexed and searched in Elasticsearch.

    Args:
        display_name: A human-readable name for the dataset (optional).
        description: A description of the dataset (optional).
        path: The path to the dataset.
        subset: The subset of the dataset (optional).
        split: The split of the dataset (e.g., "train", "test").
        messages_field: The field in the dataset that contains the text to be
                        indexed and searched (default: "messages").
        query_filter: A tuple of (field, value) to filter the dataset for the
                      benchmark queries (default: ("role", "user")).
        query_field: The field in the dataset that contains the text to be
                     used for querying (default: "content").
        index_name: The name of the Elasticsearch index to use for this dataset
                    (optional; if not provided, a default name will be generated
                    based on the path, subset, and split).
    """

    display_name: str | None
    description: str | None
    path: str
    subset: str | None
    split: str
    messages_field: str = "messages"
    query_filter: tuple[str, str] = ("role", "user")
    query_field: str = "content"
    index_name: str | None = None


@dataclass
class BenchmarkConfig:
    """
    Configuration for a benchmark to be searched for contamination.

    Args:
        display_name: A human-readable name for the benchmark.
        description: A description of the benchmark (optional).
        path: The path to the benchmark dataset.
        subset: The subset of the benchmark dataset (optional).
        split: The split of the benchmark dataset (e.g., "train", "test").
        query_fields: The fields in the benchmark dataset to use for querying.
    """

    display_name: str
    description: str | None
    path: str
    subset: str | None
    split: str
    query_fields: list[str]


@dataclass
class Result(TypedRow["Result"]):
    """
    A result of a contamination search.

    Args:
        dataset_path: The path to the dataset that was searched.
        dataset_subset: The subset of the dataset that was searched (if any).
        dataset_split: The split of the dataset that was searched.
        dataset_query_filter: The query filter used for the dataset.
        dataset_query_field: The query field used for the dataset.
        benchmark_path: The path to the benchmark that was searched.
        benchmark_subset: The subset of the benchmark that was searched (if any).
        benchmark_split: The split of the benchmark that was searched.
        benchmark_query_fields: The query fields used for the benchmark.
        dataset_num_docs: The total number of documents in the dataset.
        dataset_num_contaminated_docs: The number of contaminated documents found in
                                      the dataset.
        dataset_contamination_fraction: The fraction of contaminated documents in
                                      the dataset.
        benchmark_num_docs: The total number of documents in the benchmark.
        benchmark_num_contaminated_docs: The number of contaminated documents found
                                        in the benchmark.
        benchmark_contamination_fraction: The fraction of contaminated documents in
                                        the benchmark.
        ngram_size: The n-gram size used for matching.
        match_threshold: The similarity threshold used for matching.
        contaminated_ids: A dictionary mapping document IDs in the dataset to sets
                         of matching document IDs in the benchmark.
    """

    dataset_path: str
    dataset_subset: str | None
    dataset_split: str
    dataset_query_filter: tuple[str, str]
    dataset_query_field: str

    benchmark_path: str
    benchmark_subset: str | None
    benchmark_split: str
    benchmark_query_fields: list[str]

    dataset_num_docs: int
    dataset_num_contaminated_docs: int
    dataset_contamination_fraction: float

    benchmark_num_docs: int
    benchmark_num_contaminated_docs: int
    benchmark_contamination_fraction: float

    ngram_size: int
    match_threshold: float

    contaminated_ids: dict[int, set[int]]

    def from_row(row: pd.Series) -> "Result":
        dataset_query_filter = eval(row.pop("dataset_query_filter"))
        benchmark_query_fields = eval(row.pop("benchmark_query_fields"))
        contaminated_ids = eval(row.pop("contaminated_ids"))
        return Result(
            dataset_query_filter=dataset_query_filter,
            benchmark_query_fields=benchmark_query_fields,
            contaminated_ids=contaminated_ids,
            **row.to_dict(),
        )

    def is_same_setup(self, other: "Result") -> bool:
        """
        Check if this result is from the same dataset and benchmark setup as another result.

        Args:
            other: The other result to compare against.

        Returns:
            True if the results are from the same setup, False otherwise.
        """
        return (
            self.dataset_path == other.dataset_path
            and self.dataset_subset == other.dataset_subset
            and self.dataset_split == other.dataset_split
            and self.dataset_query_filter == other.dataset_query_filter
            and self.dataset_query_field == other.dataset_query_field
            and self.benchmark_path == other.benchmark_path
            and self.benchmark_subset == other.benchmark_subset
            and self.benchmark_split == other.benchmark_split
            and self.benchmark_query_fields == other.benchmark_query_fields
            and self.ngram_size == other.ngram_size
            and self.match_threshold == other.match_threshold
        )
