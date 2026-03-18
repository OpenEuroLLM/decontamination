import logging
from functools import partial

from datasets import Dataset
from datasets.load import load_dataset
from tqdm import tqdm

from decontamination.util import NUM_PROCESSES
from elasticsearch import Elasticsearch, helpers

from .config import DatasetConfig
from .util import build_default_index_name, create_es_connection

log = logging.getLogger(__name__)


def create_index(
    es: Elasticsearch,
    name: str,
) -> None:
    log.info("Creating a new text index: %s", name)
    mappings = {
        "properties": {
            "text": {"type": "text", "index": True},
            "original_id": {"type": "integer"},
        }
    }
    # The default analyzer is a "standard" analyzer which lowercases and splits tokens on all punctuation. This is not a great choice for math and
    # coding datasets where we would lose math operators, equations get split, etc. The following custom analyzer uses a regex pattern that splits on
    # fewer characters. This is not perfect either, but is a better choice across evals.
    settings = {
        "analysis": {
            "analyzer": {
                "tulu_analyzer": {
                    "type": "pattern",
                    "pattern": '[ ,.?!:;()"-]|\\n|\\\\',
                    "lowercase": True,
                }
            }
        }
    }
    es.indices.create(index=name, mappings=mappings, settings=settings)
    log.info("Created a new text index: %s", name)


def read_dataset(
    name: str,
    subset: str | None,
    split: str,
    messages_field: str = "messages",
    query_filter: tuple[str, str] = ("role", "user"),
    query_field: str = "content",
) -> Dataset:
    log.info("Loading dataset: %s, subset: %s, split: %s", name, subset, split)
    ds = load_dataset(name, subset, split=split)
    log.info("Loaded dataset: %s", ds)
    log.info("\tColumns: %s", ds.column_names)
    log.info("\tNumber of examples: %d", len(ds))

    return ds.map(
        partial(
            map_dataset_batched,
            messages_field=messages_field,
            query_filter=query_filter,
            query_field=query_field,
        ),
        with_indices=True,
        batched=True,
        batch_size=10,
        remove_columns=ds.column_names,
        num_proc=NUM_PROCESSES,
        desc="Processing dataset to extract text and original_id",
    )


def map_dataset_batched(
    batch: dict,
    indices: list[int],
    messages_field: str = "messages",
    query_filter: tuple[str, str] = ("role", "user"),
    query_field: str = "content",
) -> dict:
    new_indices = []
    new_batch = []
    for idx, example in zip(indices, batch[messages_field]):
        for message in example:
            if message[query_filter[0]] == query_filter[1]:
                new_indices.append(idx)
                new_batch.append(message[query_field])
    return {"original_id": new_indices, "text": new_batch}


def index_dataset(
    es: Elasticsearch,
    dataset: Dataset,
    index_name: str,
    batch_size: int = 1_000,
) -> None:
    log.info("Indexing dataset text into Elasticsearch index: %s", index_name)
    index_stats = es.indices.stats(index=index_name)
    index_size = index_stats["indices"][index_name]["total"]["docs"]["count"]
    log.info("Total number of documents in dataset (%s): %d", index_name, len(dataset))
    log.info("Current number of documents in index (before indexing): %d", index_size)
    log.info("Remaining documents to index: %d", len(dataset) - index_size)

    if index_size >= len(dataset):
        log.info(
            "Index already contains all documents from the dataset. Skipping indexing."
        )
        return

    with tqdm(total=len(dataset) - index_size, desc="Indexing dataset text") as pbar:
        for i in range(index_size, len(dataset), batch_size):
            batch = dataset[i : min(i + batch_size, len(dataset))]
            actions = [
                {"_source": {"text": text, "original_id": idx}}
                for idx, text in zip(batch["original_id"], batch["text"])
            ]
            helpers.bulk(es, actions, index=index_name)
            pbar.update(len(actions))


def index(
    datasets: list[DatasetConfig],
    batch_size: int = 1_000,
) -> None:
    """
    Index datasets into Elasticsearch. For each dataset, if the index does not
    already exist, it will be created. Then the dataset will be read and
    indexed in batches.

    Args:
        datasets: A list of DatasetConfig objects specifying the datasets to index.
        batch_size: The number of documents to index in each batch.
    """
    es = create_es_connection()

    for dataset in datasets:
        if not dataset.index_name:
            dataset.index_name = build_default_index_name(
                dataset.path, dataset.subset, dataset.split
            )

        if not es.indices.exists(index=dataset.index_name):
            create_index(es, dataset.index_name)

        ds = read_dataset(
            dataset.path,
            dataset.subset,
            dataset.split,
            dataset.messages_field,
            tuple(dataset.query_filter),
            dataset.query_field,
        )

        index_dataset(
            es,
            ds,
            dataset.index_name,
            batch_size,
        )
