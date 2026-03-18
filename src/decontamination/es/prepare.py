import logging
from pathlib import Path

from decontamination.util import run_command_streaming

from .util import MOUNTS, select_image

log = logging.getLogger(__name__)


def prepare(
    image: str | None = None,
    mounts: dict[Path, Path] = MOUNTS,
) -> None:
    """
    Prepares the local environment for running an Elasticsearch
    Apptainer container by creating necessary directories, extracting
    default configuration files, and modifying them for single-node operation.
    
    Args:
        image: The path to the Apptainer image to use for extracting
               configuration files. If None, the user will be prompted to
               select an image from the images directory.
        mounts: A dictionary mapping local paths to container paths for
                mounting. Defaults to the standard Elasticsearch config, data,
                and logs directories.
    """
    log.info("Preparing mounts for Elasticsearch...")
    for local_path, _ in mounts.items():
        local_path.mkdir(parents=True, exist_ok=True)
        if not local_path.is_dir():
            log.error(f"Failed to create directory {local_path}.")
            raise RuntimeError(f"Failed to create directory {local_path}.")
        log.info(f"Created directory {local_path}.")
    log.info("Successfully created mount folders.\n")

    log.info("Extracting default Elasticsearch configuration files...")
    if image is None:
        image = select_image(Path("images"))
    command = [
        "apptainer",
        "exec",
        image,
        "cp",
        "-r",
        "/usr/share/elasticsearch/config/.",
        "./elasticsearch/config/",
    ]
    log.info(f"Running command: {' '.join(command)}")
    returncode, output = run_command_streaming(command)
    if returncode != 0:
        log.error(f"Failed to extract configuration files: {output}")
        raise RuntimeError(f"Failed to extract configuration files: {output}")
    log.info("Successfully extracted configuration files.\n")

    log.info("Modifying configuration files...")
    config_file = Path("elasticsearch/config/elasticsearch.yml")
    if not config_file.exists():
        log.error(f"Configuration file not found: {config_file}")
        raise RuntimeError(f"Configuration file not found: {config_file}")
    with config_file.open("a") as f:
        f.write("\ndiscovery.type: single-node\nxpack.security.enabled: false\n")
    log.info("Successfully modified configuration files.\n")
