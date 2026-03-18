import logging

from decontamination.es import build, prepare, run

from .util import CLI


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    app = CLI("Elasticsearch management.")
    app.command(build, default_config="configs/es/build.yaml")
    app.command(prepare, default_config="configs/es/prepare.yaml")
    app.command(run, default_config="configs/es/run.yaml")
    app()


if __name__ == "__main__":
    main()
