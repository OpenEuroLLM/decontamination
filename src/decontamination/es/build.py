import logging
from pathlib import Path

from decontamination.util import run_command_streaming

log = logging.getLogger(__name__)

def build(
    image: str,
    output_dir: Path,
) -> str:
    """
    Builds an Apptainer image from the specified Docker
    image and returns the path to the built image.

    Args:
        image: The Docker image to build the Apptainer image from.
               Defaults to the official Elasticsearch image for
               the current architecture.
        output_dir: The directory to save the built Apptainer image.

    Returns:
        The path to the built Apptainer image.
    """
    log.info(f"Building Apptainer image from {image}:")
    apptainer_image_name = image.split("/")[-1].replace(":", "-") + ".sif"

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    apptainer_image_path = output_dir / apptainer_image_name
    if apptainer_image_path.exists():
        log.info(
            f"Apptainer image {apptainer_image_path} already exists."
            " Skipping build."
        )
        return str(apptainer_image_path)

    command = [
        "apptainer",
        "pull",
        str(output_dir / apptainer_image_name),
        f"docker://{image}",
    ]

    log.info(f"Running command: {' '.join(command)}")
    returncode, output = run_command_streaming(command)
    if returncode != 0:
        log.error(f"Failed to build Apptainer image: {output}")
        raise RuntimeError(f"Failed to build Apptainer image: {output}")

    log.info(f"Successfully built Apptainer image: {apptainer_image_path}")
    return str(apptainer_image_path)
