import logging

from decontamination.ds import index, remove, search

from .util import CLI

es_logger = logging.getLogger("elasticsearch")
es_logger.setLevel(logging.WARNING)
transport_logger = logging.getLogger("elastic_transport")
transport_logger.setLevel(logging.WARNING)
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.WARNING)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    app = CLI("Dataset indexing, benchmark matching, and decontamination.")
    app.command(index, default_config="configs/ds/index.yaml")
    app.command(search, default_config="configs/ds/search.yaml")
    app.command(remove, default_config="configs/ds/remove.yaml")
    app()


if __name__ == "__main__":
    main()
