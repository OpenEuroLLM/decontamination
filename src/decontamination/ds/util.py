from elasticsearch import Elasticsearch


def build_default_index_name(
    path: str,
    subset: str | None,
    split: str,
) -> str:
    """
    Build a default index name based on the dataset path, subset, and split.
    The index name is constructed as follows:
    - Start with the dataset path, replacing "/" with "_".
    - If a subset is provided, append "_{subset}" to the path.
    - Append "_{split}" to the path.
    - Convert the entire string to lowercase.

    Args:
        path (str): The dataset path.
        subset (str | None): The dataset subset (e.g., "small", "large"), or None if not applicable.
        split (str): The dataset split (e.g., "train", "test", "validation").

    Returns:
        str: The constructed index name.
    """
    subset = f"_{subset}" if subset else ""
    return f"{path}{subset}_{split}".replace("/", "_").lower()


def create_es_connection(
    host: str = "localhost",
    port: int = 9200,
    scheme: str = "http",
) -> Elasticsearch:
    """
    Create a connection to an Elasticsearch instance.

    Args:
        host (str): The hostname of the Elasticsearch server.
        port (int): The port number of the Elasticsearch server.
        scheme (str): The connection scheme ("http" or "https").

    Returns:
        Elasticsearch: An instance of the Elasticsearch client connected to the specified server.
    """
    return Elasticsearch(f"{scheme}://{host}:{port}")
