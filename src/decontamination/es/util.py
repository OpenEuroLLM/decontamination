from pathlib import Path

MOUNTS = {
    Path("elasticsearch") / folder: Path("/usr/share/elasticsearch") / folder
    for folder in [
        "config",
        "data",
        "logs",
    ]
}


def select_image(
    images_dir: Path = Path("images"),
) -> str:
    """
    Prompt the user to select an Apptainer .sif image from
    the images directory.

    Args:
        images_dir: The directory to search for .sif images.

    Returns:
        The path to the selected .sif image.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise RuntimeError(f"Images directory not found: {images_dir}")

    sif_files = sorted(images_dir.glob("*.sif"))
    if not sif_files:
        raise RuntimeError(f"No .sif images found in {images_dir}")
    if len(sif_files) == 1:
        return str(sif_files[0])

    print("Multiple Apptainer images found; choose one:")
    for i, p in enumerate(sif_files, start=1):
        print(f"  {i}: {p.name}")

    while True:
        try:
            choice = input(f"Select image [1-{len(sif_files)}]: ").strip()
        except EOFError, KeyboardInterrupt:
            raise RuntimeError("No image selected (input interrupted)")

        if not choice:
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sif_files):
                return str(sif_files[idx - 1])

        for p in sif_files:
            if p.name == choice:
                return str(p)

        print(f"Invalid selection: {choice}")
